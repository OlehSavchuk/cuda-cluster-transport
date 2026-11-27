#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"
#include "physics/mean_field.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/hydro_io.cuh"
#include "sim/particles.cuh"

namespace transport {

// Self-describing binary 3D Baryon Stress-Energy Tensor & Eckart-Frame header for HCGF ("HCGF", 88 bytes)
// Coarse-grained strictly over BARYONS (B > 0; mesons/pions with B = 0 are excluded!).
// Followed immediately by 24 contiguous float32 3D arrays of length nx * ny * nz:
//   [0..9]   T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz  [GeV / fm^3] (baryon kinetic + mean-field)
//   [10..13] JB0, JBx, JBy, JBz                                [fm^-3]      (baryon 4-current J_B^mu)
//   [14..17] JQ0, JQx, JQy, JQz                                [fm^-3]      (baryonic charge 4-current)
//   [18..23] eps_E [GeV/fm^3], P_E [GeV/fm^3], nB_E [fm^-3], vx_E [c], vy_E [c], vz_E [c] (Eckart frame!)
#pragma pack(push, 1)
struct HcgfTmunuHeader {
    std::uint32_t magic = 0x46474348U;  // "HCGF"
    std::int32_t version = 1;
    std::int32_t step = 0;
    std::int32_t nx = 0;
    std::int32_t ny = 0;
    std::int32_t nz = 0;
    std::int32_t n_fields = 24;
    std::int32_t reserved = 0;
    double t_fm = 0.0;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
};
#pragma pack(pop)

struct HcgfGridSnapshot {
    std::vector<float> fields[24];
    // 3x3 Eckart rest-frame baryon kinetic stress tensor [Pxx*, Pxy*, Pxz*, Pyy*, Pyz*, Pzz*] [GeV/fm^3]
    std::vector<float> pkin_eckart[6];
    bool has_frozen_particles = false;
    bool has_backflow_candidates = false;
};

namespace hcgf_detail {

// Exact Lorentz boost projection P*_ij = L^i_mu L^j_nu T^munu into rest frame moving at v = (vx, vy, vz)
template <typename T>
HD_INLINE void project_stress_to_rest_frame(T vx, T vy, T vz,
                                            T T00, T T0x, T T0y, T T0z,
                                            T Txx, T Txy, T Txz,
                                            T Tyy, T Tyz, T Tzz,
                                            T Pstar[6]) {
    const T v2_raw = vx * vx + vy * vy + vz * vz;
    const T v2 = (v2_raw < T(0.9990)) ? v2_raw : T(0.9990);
    const T u0 = T(1) / std::sqrt(T(1) - v2);
    const T ux = u0 * vx, uy = u0 * vy, uz = u0 * vz;
    const T ig = T(1) / (u0 + T(1));
    const T L[3][4] = {
        {-ux, T(1) + ux * ux * ig,       ux * uy * ig,       ux * uz * ig},
        {-uy,       uy * ux * ig, T(1) + uy * uy * ig,       uy * uz * ig},
        {-uz,       uz * ux * ig,       uz * uy * ig, T(1) + uz * uz * ig}
    };
    const T Tm[4][4] = {
        {T00, T0x, T0y, T0z},
        {T0x, Txx, Txy, Txz},
        {T0y, Txy, Tyy, Tyz},
        {T0z, Txz, Tyz, Tzz}
    };
    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            T s = T(0);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) s += L[i][a] * L[j][b] * Tm[a][b];
            Pstar[idx++] = s;
        }
    }
}

// Extract Eckart collective 4-velocity u^mu = J_B^mu / n_B = gamma_E * (1, v_E) from baryon 4-current,
// and project baryon stress-energy tensor T_B^munu into the Eckart rest frame:
//   eps_E = u_mu T_B^munu u_nu,   P_E = (eps_E - g_munu T_B^munu) / 3
template <typename T>
HD_INLINE void eval_eckart_frame(T JB0, T JBx, T JBy, T JBz,
                                 T T00, T T0x, T T0y, T T0z,
                                 T Txx, T Txy, T Txz,
                                 T Tyy, T Tyz, T Tzz,
                                 T& nB_e, T& eps_e, T& P_e,
                                 T& vx_e, T& vy_e, T& vz_e) {
    if (JB0 <= T(1.0e-12) || T00 <= T(1.0e-12)) {
        nB_e = T(0); eps_e = T(0); P_e = T(0);
        vx_e = T(0); vy_e = T(0); vz_e = T(0);
        return;
    }
    vx_e = JBx / JB0;
    vy_e = JBy / JB0;
    vz_e = JBz / JB0;
    const T v2 = vx_e * vx_e + vy_e * vy_e + vz_e * vz_e;
    if (v2 > T(0.9990)) {
        const T sc = T(0.9995) / std::sqrt(v2);
        vx_e *= sc; vy_e *= sc; vz_e *= sc;
    }
    const T v2_clamped = vx_e * vx_e + vy_e * vy_e + vz_e * vz_e;
    const T u0 = T(1) / std::sqrt(T(1) - v2_clamped);
    const T ux = u0 * vx_e, uy = u0 * vy_e, uz = u0 * vz_e;
    nB_e = JB0 / u0;

    // Exact covariant Eckart energy density eps_E = u_mu T_B^munu u_nu with u_mu = (u0, -ux, -uy, -uz)
    eps_e = u0 * u0 * T00
            - T(2) * u0 * (ux * T0x + uy * T0y + uz * T0z)
            + (ux * ux * Txx + uy * uy * Tyy + uz * uz * Tzz)
            + T(2) * (ux * uy * Txy + ux * uz * Txz + uy * uz * Tyz);
    if (eps_e < T(0)) eps_e = T(0);

    const T trace_gT = T00 - (Txx + Tyy + Tzz);
    P_e = (eps_e - trace_gT) / T(3);
}

// Pratt-Torrieri symmetric positive-definite linear momentum transformation p*_i = Lambda_ij q_j
// with exact volume preservation det(Lambda) == 1 matching 3x3 Eckart pressure tensor P*_ij:
//   Lambda = R * diag(alpha_1, alpha_2, alpha_3) * R^T,   alpha_m^2 proportional to eig_m(P*_ij)
inline void compute_pratt_torrieri_lambda(const double Pstar[6], double Lambda[9], double alpha_out[3]) {
    Lambda[0] = 1.0; Lambda[1] = 0.0; Lambda[2] = 0.0;
    Lambda[3] = 0.0; Lambda[4] = 1.0; Lambda[5] = 0.0;
    Lambda[6] = 0.0; Lambda[7] = 0.0; Lambda[8] = 1.0;
    alpha_out[0] = alpha_out[1] = alpha_out[2] = 1.0;

    const double p_mean = (Pstar[0] + Pstar[3] + Pstar[5]) / 3.0;
    if (p_mean <= 1.0e-14) return;

    double A[3][3] = {
        {Pstar[0], Pstar[1], Pstar[2]},
        {Pstar[1], Pstar[3], Pstar[4]},
        {Pstar[2], Pstar[4], Pstar[5]}
    };
    double R[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int sw = 0; sw < 12; ++sw) {
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const double apq = A[p][q];
                if (std::fabs(apq) <= 1.0e-15 * p_mean) continue;
                const double tau = (A[q][q] - A[p][p]) / (2.0 * apq);
                const double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                              : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                A[p][p] -= t * apq; A[q][q] += t * apq; A[p][q] = A[q][p] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    if (r != p && r != q) {
                        const double arp = A[r][p], arq = A[r][q];
                        A[r][p] = A[p][r] = c * arp - s * arq;
                        A[r][q] = A[q][r] = s * arp + c * arq;
                    }
                    const double rrp = R[r][p], rrq = R[r][q];
                    R[r][p] = c * rrp - s * rrq;
                    R[r][q] = s * rrp + c * rrq;
                }
            }
        }
    }
    double a[3];
    for (int m = 0; m < 3; ++m) a[m] = std::sqrt(std::max(A[m][m], 0.04 * p_mean) + 0.01 * p_mean);
    const double g1 = std::cbrt(a[0] * a[1] * a[2]);
    for (int m = 0; m < 3; ++m) a[m] = std::clamp(a[m] / g1, 0.40, 2.50);
    const double g2 = std::cbrt(a[0] * a[1] * a[2]);
    for (int m = 0; m < 3; ++m) alpha_out[m] = a[m] / g2;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Lambda[3 * i + j] = alpha_out[0] * R[i][0] * R[j][0] +
                                alpha_out[1] * R[i][1] * R[j][1] +
                                alpha_out[2] * R[i][2] * R[j][2];
}

// Exact Skyrme + Lorentzian Momentum-Dependent potential energy density V_pot(r) [GeV/fm^3]
// entering T^00 (matching MeanFieldGrid::compute_potential_energy_test_gev identically!)
template <typename T>
HD_INLINE T eval_skyrme_md_t00_pot_density(T rho_b_lab, T delta, T psi_md_val,
                                           const MeanFieldConfig& mf_cfg) {
    if (!mf_cfg.enabled || rho_b_lab <= T(1.0e-12)) return T(0);
    const T rho0 = static_cast<T>(mf_cfg.rho0_fm3);
    const T u = rho_b_lab / rho0;
    const T sig_p1 = static_cast<T>(mf_cfg.sigma_exp + 1.0);
    const T gam_p1 = static_cast<T>(mf_cfg.gamma_sym + 1.0);
    return T(0.5) * static_cast<T>(mf_cfg.A_sk_gev) * rho0 * u * u +
           (static_cast<T>(mf_cfg.B_sk_gev) / sig_p1) * rho0 * std::pow(u, sig_p1) +
           static_cast<T>(mf_cfg.S_pot_gev) * rho0 * std::pow(u, gam_p1) * delta * delta +
           T(0.5) * static_cast<T>(mf_cfg.C_md_gev) * rho_b_lab * psi_md_val;
}

#if TRANSPORT_HAS_CUDA
// 20 raw CIC fields packed contiguously as [20][nc]:
//   0..9: t00, t0x, t0y, t0z, txx, txy, txz, tyy, tyz, tzz
//   10..11: rp, rn | 12..15: jb0, jbx, jby, jbz | 16..19: jq0, jqx, jqy, jqz
template <int = 0>
__global__ void deposit_hcgf_cic_kernel(ParticleSoAView p, GridConfig gcfg, int nc,
                                        float inv_n_dV, bool exclude_frozen,
                                        float* d_raw, int* d_has_frozen) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n || !p.active[i]) return;
    if (p.freezeout_tag[i] != 0) {
        if (d_has_frozen && *d_has_frozen == 0) *d_has_frozen = 1;
        if (exclude_frozen) return;
    }
    const int sp_id = p.species[i];
    const SpeciesInfo sp = get_species_info(sp_id);
    if (sp.baryon_num <= 0) return;

    int i0, j0, k0;
    Real fx_r, fy_r, fz_r;
    if (!mf_detail::cic_weights(p.x[i], p.y[i], p.z[i], gcfg, i0, j0, k0, fx_r, fy_r, fz_r)) return;

    const float fx = static_cast<float>(fx_r), fy = static_cast<float>(fy_r), fz = static_cast<float>(fz_r);
    const float inv_fo = (p.freezeout_tag[i] > 1) ? (1.0f / static_cast<float>(p.freezeout_tag[i])) : 1.0f;
    const float w_scale = inv_n_dV * inv_fo;
    const float E = static_cast<float>(p.E[i]);
    const float px = static_cast<float>(p.px[i]);
    const float py = static_cast<float>(p.py[i]);
    const float pz = static_cast<float>(p.pz[i]);
    const float inv_E = 1.0f / E;
    const float vx = px * inv_E, vy = py * inv_E, vz = pz * inv_E;
    const float B = static_cast<float>(sp.baryon_num);
    const float Q = static_cast<float>(sp.charge);
    float wp_f = 0.0f, wn_f = 0.0f;
    get_effective_pn_weights(sp_id, wp_f, wn_f);
    const float vals[20] = {
        E, px, py, pz,
        px * vx, px * vy, px * vz, py * vy, py * vz, pz * vz,
        wp_f, wn_f,
        B, B * vx, B * vy, B * vz,
        Q, Q * vx, Q * vy, Q * vz
    };

    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
        const float wz = (dz == 0) ? (1.0f - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
            const float wy = (dy == 0) ? (1.0f - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                const float w = (((dx == 0) ? (1.0f - fx) : fx) * wy * wz) * w_scale;
                const int c = mf_detail::cell_flat(ix, iy, iz, gcfg);
                for (int f = 0; f < 20; ++f) {
                    atomicAdd(&d_raw[f * nc + c], w * vals[f]);
                }
            }
        }
    }
}

template <int = 0>
__global__ void smooth_hcgf_batch_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                         int nc, int n_fields, int axis, GridConfig gcfg) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * n_fields) return;
    const int f = idx / nc;
    const int c = idx - f * nc;
    mf_detail::smooth_axis_cell(src + f * nc, dst + f * nc, c, axis, gcfg);
}

template <int = 0>
__global__ void deposit_hcgf_psi_kernel(ParticleSoAView p, GridConfig gcfg, int nc,
                                        float inv_n_dV_rho0, float inv_lam2, bool exclude_frozen,
                                        const float* __restrict__ d_raw, float* d_psi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n || !p.active[i]) return;
    if (exclude_frozen && p.freezeout_tag[i] != 0) return;
    const SpeciesInfo sp = get_species_info(p.species[i]);
    if (sp.baryon_num <= 0) return;

    int i0, j0, k0;
    Real fx_r, fy_r, fz_r;
    if (!mf_detail::cic_weights(p.x[i], p.y[i], p.z[i], gcfg, i0, j0, k0, fx_r, fy_r, fz_r)) return;
    const float fx = static_cast<float>(fx_r), fy = static_cast<float>(fy_r), fz = static_cast<float>(fz_r);

    const float* rp  = d_raw + 10 * nc;
    const float* rn  = d_raw + 11 * nc;
    const float* t0x = d_raw + 1 * nc;
    const float* t0y = d_raw + 2 * nc;
    const float* t0z = d_raw + 3 * nc;

    float r_tot = 0.0f, pfx = 0.0f, pfy = 0.0f, pfz = 0.0f;
    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
        const float wz = (dz == 0) ? (1.0f - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
            const float wy = (dy == 0) ? (1.0f - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                const float w = ((dx == 0) ? (1.0f - fx) : fx) * wy * wz;
                const int c = mf_detail::cell_flat(ix, iy, iz, gcfg);
                r_tot += w * (rp[c] + rn[c]);
                pfx += w * t0x[c]; pfy += w * t0y[c]; pfz += w * t0z[c];
            }
        }
    }
    if (r_tot > 1.0e-10f) { const float ir = 1.0f / r_tot; pfx *= ir; pfy *= ir; pfz *= ir; }
    const float inv_fo = (p.freezeout_tag[i] > 1) ? (1.0f / static_cast<float>(p.freezeout_tag[i])) : 1.0f;
    const float inv_A = 1.0f / static_cast<float>(sp.baryon_num);
    const float qx = static_cast<float>(p.px[i]) * inv_A - pfx;
    const float qy = static_cast<float>(p.py[i]) * inv_A - pfy;
    const float qz = static_cast<float>(p.pz[i]) * inv_A - pfz;
    const float contrib = (static_cast<float>(sp.baryon_num) * inv_n_dV_rho0 * inv_fo) /
                          (1.0f + (qx * qx + qy * qy + qz * qz) * inv_lam2);
    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
        const float wz = (dz == 0) ? (1.0f - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
            const float wy = (dy == 0) ? (1.0f - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                const float w = ((dx == 0) ? (1.0f - fx) : fx) * wy * wz;
                atomicAdd(&d_psi[mf_detail::cell_flat(ix, iy, iz, gcfg)], w * contrib);
            }
        }
    }
}

template <int = 0>
__global__ void finalize_hcgf_eckart_kernel(const float* __restrict__ d_raw,
                                            const float* __restrict__ d_psi,
                                            float* __restrict__ d_out,
                                            int nc, MeanFieldConfig mf_cfg) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;

    const float t00 = d_raw[0 * nc + c], t0x = d_raw[1 * nc + c], t0y = d_raw[2 * nc + c], t0z = d_raw[3 * nc + c];
    const float txx = d_raw[4 * nc + c], txy = d_raw[5 * nc + c], txz = d_raw[6 * nc + c];
    const float tyy = d_raw[7 * nc + c], tyz = d_raw[8 * nc + c], tzz = d_raw[9 * nc + c];
    const float rp  = d_raw[10 * nc + c], rn = d_raw[11 * nc + c];
    const float jb0 = d_raw[12 * nc + c], jbx = d_raw[13 * nc + c], jby = d_raw[14 * nc + c], jbz = d_raw[15 * nc + c];
    const float jq0 = d_raw[16 * nc + c], jqx = d_raw[17 * nc + c], jqy = d_raw[18 * nc + c], jqz = d_raw[19 * nc + c];

    float T00 = t00;
    const float rb_eff = rp + rn;
    if (mf_cfg.enabled && rb_eff > 1.0e-12f) {
        const float delta = (rn - rp) / rb_eff;
        const float v_pot = eval_skyrme_md_t00_pot_density(rb_eff, delta, d_psi ? d_psi[c] : 0.0f, mf_cfg);
        T00 = (T00 + v_pot > 1.0e-12f) ? (T00 + v_pot) : 1.0e-12f;
    }

    float nb_e = 0.0f, eps_e = 0.0f, P_e = 0.0f, vx_e = 0.0f, vy_e = 0.0f, vz_e = 0.0f;
    eval_eckart_frame(jb0, jbx, jby, jbz,
                      T00, t0x, t0y, t0z, txx, txy, txz, tyy, tyz, tzz,
                      nb_e, eps_e, P_e, vx_e, vy_e, vz_e);

    float pkin_star[6] = {};
    if (nb_e > 1.0e-12f) {
        project_stress_to_rest_frame(vx_e, vy_e, vz_e,
                                     t00, t0x, t0y, t0z, txx, txy, txz, tyy, tyz, tzz, pkin_star);
    }

    const float vals[24] = {
        T00, t0x, t0y, t0z, txx, txy, txz, tyy, tyz, tzz,
        jb0, jbx, jby, jbz, jq0, jqx, jqy, jqz,
        eps_e, P_e, nb_e, vx_e, vy_e, vz_e
    };
    for (int f = 0; f < 24; ++f) d_out[f * nc + c] = vals[f];
    for (int s = 0; s < 6; ++s)  d_out[(24 + s) * nc + c] = pkin_star[s];
}

template <int = 0>
__global__ void check_frozen_backflow_kernel(ParticleSoAView p, GridConfig gcfg, int nc,
                                             const float* __restrict__ d_nb, float n_melt,
                                             int* d_has_backflow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n || !p.active[i] || p.freezeout_tag[i] == 0) return;
    if (*d_has_backflow != 0) return;
    if (get_species_info(p.species[i]).baryon_num <= 0) return;
    const Real xmin = static_cast<Real>(gcfg.xmin), ymin = static_cast<Real>(gcfg.ymin), zmin = static_cast<Real>(gcfg.zmin);
    const Real inv_dx = static_cast<Real>(gcfg.nx) / static_cast<Real>(gcfg.xmax - gcfg.xmin);
    const Real inv_dy = static_cast<Real>(gcfg.ny) / static_cast<Real>(gcfg.ymax - gcfg.ymin);
    const Real inv_dz = static_cast<Real>(gcfg.nz) / static_cast<Real>(gcfg.zmax - gcfg.zmin);
    const int ix = static_cast<int>(std::floor((p.x[i] - xmin) * inv_dx));
    const int iy = static_cast<int>(std::floor((p.y[i] - ymin) * inv_dy));
    const int iz = static_cast<int>(std::floor((p.z[i] - zmin) * inv_dz));
    if (ix < 0 || ix >= gcfg.nx || iy < 0 || iy >= gcfg.ny || iz < 0 || iz >= gcfg.nz) return;
    const int c = ix + gcfg.nx * (iy + gcfg.ny * iz);
    if (c >= 0 && c < nc && d_nb[c] > n_melt) {
        *d_has_backflow = 1;
    }
}
#endif

}  // namespace hcgf_detail

inline HcgfGridSnapshot compute_hcgf_tmunu_snapshot(int n_test,
                                                    const GridConfig& gcfg,
                                                    const MeanFieldConfig& mf_cfg,
                                                    const std::vector<HostParticle>& host_p,
                                                    bool exclude_frozen = false) {
    HcgfGridSnapshot snap{};
    const int nc = gcfg.num_cells();
    if (nc <= 0) return snap;
    for (int f = 0; f < 24; ++f) snap.fields[f].assign(nc, 0.0f);
    for (int s = 0; s < 6; ++s) snap.pkin_eckart[s].assign(nc, 0.0f);

    const double inv_n_dV = 1.0 / (static_cast<double>(n_test) * gcfg.cell_volume_fm3());
    std::vector<double> t00(nc, 0.0), t0x(nc, 0.0), t0y(nc, 0.0), t0z(nc, 0.0);
    std::vector<double> txx(nc, 0.0), txy(nc, 0.0), txz(nc, 0.0);
    std::vector<double> tyy(nc, 0.0), tyz(nc, 0.0), tzz(nc, 0.0);
    std::vector<double> rp(nc, 0.0), rn(nc, 0.0), jb0(nc, 0.0), jbx(nc, 0.0), jby(nc, 0.0), jbz(nc, 0.0);
    std::vector<double> jq0(nc, 0.0), jqx(nc, 0.0), jqy(nc, 0.0), jqz(nc, 0.0);
    std::vector<double> tmp(nc, 0.0);

    for (const auto& hp : host_p) {
        if (hp.freezeout_tag != 0) {
            snap.has_frozen_particles = true;
            if (exclude_frozen) continue;
        }
        const SpeciesInfo sp = get_species_info(hp.species);
        if (sp.baryon_num <= 0) continue;  // Coarse-grain ONLY baryons (exclude B=0 pions!)

        int i0, j0, k0;
        double fx, fy, fz;
        if (!mf_detail::cic_weights(hp.x, hp.y, hp.z, gcfg, i0, j0, k0, fx, fy, fz)) continue;

        const double inv_fo = (hp.freezeout_tag > 1) ? (1.0 / static_cast<double>(hp.freezeout_tag)) : 1.0;
        const double w_scale = inv_n_dV * inv_fo;
        const double B = static_cast<double>(sp.baryon_num);
        const double Q = static_cast<double>(sp.charge);
        double wp = 0.0, wn = 0.0;
        get_effective_pn_weights(hp.species, wp, wn);
        const double inv_E = 1.0 / hp.E;
        const double vx = hp.px * inv_E, vy = hp.py * inv_E, vz = hp.pz * inv_E;

        for (int dz = 0; dz <= 1; ++dz) {
            const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
            const double wz = (dz == 0) ? (1.0 - fz) : fz;
            for (int dy = 0; dy <= 1; ++dy) {
                const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
                const double wy = (dy == 0) ? (1.0 - fy) : fy;
                for (int dx = 0; dx <= 1; ++dx) {
                    const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                    const double w = (((dx == 0) ? (1.0 - fx) : fx) * wy * wz) * w_scale;
                    const int c = mf_detail::cell_flat(ix, iy, iz, gcfg);
                    t00[c] += w * hp.E;
                    t0x[c] += w * hp.px;     t0y[c] += w * hp.py;     t0z[c] += w * hp.pz;
                    txx[c] += w * hp.px * vx; txy[c] += w * hp.px * vy; txz[c] += w * hp.px * vz;
                    tyy[c] += w * hp.py * vy; tyz[c] += w * hp.py * vz; tzz[c] += w * hp.pz * vz;
                    rp[c]  += w * wp;         rn[c]  += w * wn;
                    jb0[c] += w * B;          jbx[c] += w * B * vx;   jby[c] += w * B * vy; jbz[c] += w * B * vz;
                    jq0[c] += w * Q;          jqx[c] += w * Q * vx;   jqy[c] += w * Q * vy; jqz[c] += w * Q * vz;
                }
            }
        }
    }

    for (auto* vec : {&t00, &t0x, &t0y, &t0z, &txx, &txy, &txz, &tyy, &tyz, &tzz,
                      &rp, &rn, &jb0, &jbx, &jby, &jbz, &jq0, &jqx, &jqy, &jqz}) {
        hydro_detail::smooth_host_field_3d(*vec, tmp, gcfg);
    }

    // Second CIC pass for exact Lorentzian momentum-dependent potential psi_md(c) when C_md != 0
    std::vector<double> psi_md(nc, 0.0);
    if (mf_cfg.enabled && std::fabs(mf_cfg.C_md_gev) > 1.0e-12) {
        const double inv_n_dV_rho0 = inv_n_dV / mf_cfg.rho0_fm3;
        const double inv_lam2 = 1.0 / (mf_cfg.Lambda_md_gev * mf_cfg.Lambda_md_gev);
        for (const auto& hp : host_p) {
            if (exclude_frozen && hp.freezeout_tag != 0) continue;
            const SpeciesInfo sp = get_species_info(hp.species);
            if (sp.baryon_num <= 0) continue;
            int i0, j0, k0;
            double fx, fy, fz;
            if (!mf_detail::cic_weights(hp.x, hp.y, hp.z, gcfg, i0, j0, k0, fx, fy, fz)) continue;

            double r_tot = 0.0, pfx = 0.0, pfy = 0.0, pfz = 0.0;
            for (int dz = 0; dz <= 1; ++dz) {
                const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
                const double wz = (dz == 0) ? (1.0 - fz) : fz;
                for (int dy = 0; dy <= 1; ++dy) {
                    const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
                    const double wy = (dy == 0) ? (1.0 - fy) : fy;
                    for (int dx = 0; dx <= 1; ++dx) {
                        const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                        const double w = ((dx == 0) ? (1.0 - fx) : fx) * wy * wz;
                        const int c = mf_detail::cell_flat(ix, iy, iz, gcfg);
                        r_tot += w * (rp[c] + rn[c]);
                        pfx += w * t0x[c]; pfy += w * t0y[c]; pfz += w * t0z[c];
                    }
                }
            }
            if (r_tot > 1.0e-10) { pfx /= r_tot; pfy /= r_tot; pfz /= r_tot; }
            const double inv_fo = (hp.freezeout_tag > 1) ? (1.0 / static_cast<double>(hp.freezeout_tag)) : 1.0;
            const double inv_A = 1.0 / static_cast<double>(sp.baryon_num);
            const double qx = hp.px * inv_A - pfx, qy = hp.py * inv_A - pfy, qz = hp.pz * inv_A - pfz;
            const double contrib = (sp.baryon_num * inv_n_dV_rho0 * inv_fo) /
                                   (1.0 + (qx * qx + qy * qy + qz * qz) * inv_lam2);
            for (int dz = 0; dz <= 1; ++dz) {
                const int iz = mf_detail::wrap_or_clamp_idx(k0 + dz, gcfg.nz, gcfg.boundary);
                const double wz = (dz == 0) ? (1.0 - fz) : fz;
                for (int dy = 0; dy <= 1; ++dy) {
                    const int iy = mf_detail::wrap_or_clamp_idx(j0 + dy, gcfg.ny, gcfg.boundary);
                    const double wy = (dy == 0) ? (1.0 - fy) : fy;
                    for (int dx = 0; dx <= 1; ++dx) {
                        const int ix = mf_detail::wrap_or_clamp_idx(i0 + dx, gcfg.nx, gcfg.boundary);
                        const double w = ((dx == 0) ? (1.0 - fx) : fx) * wy * wz;
                        psi_md[mf_detail::cell_flat(ix, iy, iz, gcfg)] += w * contrib;
                    }
                }
            }
        }
        hydro_detail::smooth_host_field_3d(psi_md, tmp, gcfg);
    }

    for (int c = 0; c < nc; ++c) {
        double T00 = t00[c], T0x = t0x[c], T0y = t0y[c], T0z = t0z[c];
        double Txx = txx[c], Txy = txy[c], Txz = txz[c], Tyy = tyy[c], Tyz = tyz[c], Tzz = tzz[c];

        // Include Skyrme + Momentum-Dependent potential energy density strictly in T^00
        // (preserving exact 3-momentum conservation int T^0i d^3x == P_B^i and int T^00 d^3x == H_B!)
        const double rb_eff = rp[c] + rn[c];
        if (mf_cfg.enabled && rb_eff > 1.0e-12) {
            const double delta = (rn[c] - rp[c]) / rb_eff;
            const double v_pot = hcgf_detail::eval_skyrme_md_t00_pot_density(rb_eff, delta, psi_md[c], mf_cfg);
            T00 = std::max(1.0e-12, T00 + v_pot);
        }

        double nb_e = 0.0, eps_e = 0.0, P_e = 0.0, vx_e = 0.0, vy_e = 0.0, vz_e = 0.0;
        hcgf_detail::eval_eckart_frame(jb0[c], jbx[c], jby[c], jbz[c],
                                       T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz,
                                       nb_e, eps_e, P_e, vx_e, vy_e, vz_e);

        double pkin_star[6] = {};
        if (nb_e > 1.0e-12) {
            hcgf_detail::project_stress_to_rest_frame(
                vx_e, vy_e, vz_e,
                t00[c], t0x[c], t0y[c], t0z[c],
                txx[c], txy[c], txz[c], tyy[c], tyz[c], tzz[c], pkin_star);
        }
        for (int s = 0; s < 6; ++s) snap.pkin_eckart[s][c] = static_cast<float>(pkin_star[s]);

        const double vals[24] = {
            T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz,
            jb0[c], jbx[c], jby[c], jbz[c],
            jq0[c], jqx[c], jqy[c], jqz[c],
            eps_e, P_e, nb_e, vx_e, vy_e, vz_e
        };
        for (int f = 0; f < 24; ++f) snap.fields[f][c] = static_cast<float>(vals[f]);
    }
    return snap;
}

inline void write_hcgf_tmunu_binary(const std::string& tmunu_dir, int idx, int step, double t_fm,
                                    const GridConfig& gcfg, const HcgfGridSnapshot& snap) {
    std::filesystem::create_directories(tmunu_dir);
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/tmunu_grid_%04d.bin", tmunu_dir.c_str(), idx);
    const int nc = gcfg.num_cells();
    HcgfTmunuHeader hdr{};
    hdr.step = step; hdr.nx = gcfg.nx; hdr.ny = gcfg.ny; hdr.nz = gcfg.nz; hdr.t_fm = t_fm;
    hdr.xmin = gcfg.xmin; hdr.xmax = gcfg.xmax;
    hdr.ymin = gcfg.ymin; hdr.ymax = gcfg.ymax;
    hdr.zmin = gcfg.zmin; hdr.zmax = gcfg.zmax;
    std::ofstream out(fname, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(HcgfTmunuHeader));
    for (int f = 0; f < 24; ++f) {
        out.write(reinterpret_cast<const char*>(snap.fields[f].data()),
                  static_cast<std::streamsize>(nc * sizeof(float)));
    }
}

// Reusable zero-allocation GPU HCGF T^munu + Eckart frame workspace (eliminates per-step host CIC & D2H!)
class HcgfGpuWorkspace {
public:
    const HcgfGridSnapshot& compute(ParticleSystem& particles,
                                    int n_test,
                                    const GridConfig& gcfg,
                                    const MeanFieldConfig& mf_cfg,
                                    bool exclude_frozen,
                                    bool full_24_fields,
                                    float n_melt_check = -1.0f) {
#if TRANSPORT_HAS_CUDA
        const int nc = gcfg.num_cells();
        if (nc <= 0) return snap_;
        if (d_raw_.size() != static_cast<std::size_t>(20 * nc)) {
            d_raw_.resize(static_cast<std::size_t>(20 * nc));
            d_tmp_.resize(static_cast<std::size_t>(20 * nc));
            d_psi_.resize(static_cast<std::size_t>(nc));
            d_out_.resize(static_cast<std::size_t>(30 * nc));
            d_has_frozen_.resize(2);
        }
        for (int f = 0; f < 24; ++f) {
            if (full_24_fields || f == 10 || f == 14 || f == 18 || f >= 20) {
                snap_.fields[f].resize(static_cast<std::size_t>(nc));
            }
        }
        for (int s = 0; s < 6; ++s) snap_.pkin_eckart[s].resize(static_cast<std::size_t>(nc));

        d_raw_.zero();
        d_has_frozen_.zero();
        snap_.has_frozen_particles = false;
        snap_.has_backflow_candidates = false;
        const int np = particles.slot_size();
        constexpr int TPB = 256;
        const float inv_n_dV = static_cast<float>(1.0 / (static_cast<double>(n_test) * gcfg.cell_volume_fm3()));

        if (np > 0) {
            const ParticleSoAView pv = particles.view();
            const int p_blocks = (np + TPB - 1) / TPB;
            hcgf_detail::deposit_hcgf_cic_kernel<<<p_blocks, TPB>>>(
                pv, gcfg, nc, inv_n_dV, exclude_frozen, d_raw_.data(), d_has_frozen_.data());
            CUDA_CHECK(cudaGetLastError());

            const int b20 = (20 * nc + TPB - 1) / TPB;
            hcgf_detail::smooth_hcgf_batch_kernel<<<b20, TPB>>>(d_raw_.data(), d_tmp_.data(), nc, 20, 0, gcfg);
            hcgf_detail::smooth_hcgf_batch_kernel<<<b20, TPB>>>(d_tmp_.data(), d_raw_.data(), nc, 20, 1, gcfg);
            hcgf_detail::smooth_hcgf_batch_kernel<<<b20, TPB>>>(d_raw_.data(), d_tmp_.data(), nc, 20, 2, gcfg);
            CUDA_CHECK(cudaGetLastError());
            // Smoothed 20 fields now reside in d_tmp_!

            const bool use_md = mf_cfg.enabled && (std::fabs(mf_cfg.C_md_gev) > 1.0e-12);
            if (use_md) {
                d_psi_.zero();
                const float inv_n_dV_rho0 = inv_n_dV / static_cast<float>(mf_cfg.rho0_fm3);
                const float inv_lam2 = static_cast<float>(1.0 / (mf_cfg.Lambda_md_gev * mf_cfg.Lambda_md_gev));
                hcgf_detail::deposit_hcgf_psi_kernel<<<p_blocks, TPB>>>(
                    pv, gcfg, nc, inv_n_dV_rho0, inv_lam2, exclude_frozen, d_tmp_.data(), d_psi_.data());
                const int b1 = (nc + TPB - 1) / TPB;
                // Reuse d_raw_[0..nc-1] as scratch for 3-axis smoothing of d_psi_
                hcgf_detail::smooth_hcgf_batch_kernel<<<b1, TPB>>>(d_psi_.data(), d_raw_.data(), nc, 1, 0, gcfg);
                hcgf_detail::smooth_hcgf_batch_kernel<<<b1, TPB>>>(d_raw_.data(), d_psi_.data(), nc, 1, 1, gcfg);
                hcgf_detail::smooth_hcgf_batch_kernel<<<b1, TPB>>>(d_psi_.data(), d_raw_.data(), nc, 1, 2, gcfg);
                CUDA_CHECK(cudaGetLastError());
            }

            const int c_blocks = (nc + TPB - 1) / TPB;
            hcgf_detail::finalize_hcgf_eckart_kernel<<<c_blocks, TPB>>>(
                d_tmp_.data(), use_md ? d_raw_.data() : nullptr, d_out_.data(), nc, mf_cfg);
            CUDA_CHECK(cudaGetLastError());

            if (n_melt_check >= 0.0f) {
                hcgf_detail::check_frozen_backflow_kernel<<<p_blocks, TPB>>>(
                    pv, gcfg, nc, d_out_.data() + 20 * nc, n_melt_check, d_has_frozen_.data() + 1);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            d_out_.zero();
        }

        const std::size_t bytes = static_cast<std::size_t>(nc) * sizeof(float);
        if (full_24_fields) {
            for (int f = 0; f < 24; ++f) {
                CUDA_CHECK(cudaMemcpy(snap_.fields[f].data(), d_out_.data() + f * nc, bytes, cudaMemcpyDeviceToHost));
            }
        } else {
            constexpr int NEEDED[7] = {10, 14, 18, 20, 21, 22, 23};
            for (int f : NEEDED) {
                CUDA_CHECK(cudaMemcpy(snap_.fields[f].data(), d_out_.data() + f * nc, bytes, cudaMemcpyDeviceToHost));
            }
        }
        for (int s = 0; s < 6; ++s) {
            CUDA_CHECK(cudaMemcpy(snap_.pkin_eckart[s].data(), d_out_.data() + (24 + s) * nc, bytes, cudaMemcpyDeviceToHost));
        }
        int h_fr[2] = {0, 0};
        d_has_frozen_.copy_to_host(h_fr, 2);
        snap_.has_frozen_particles = (h_fr[0] != 0);
        snap_.has_backflow_candidates = (n_melt_check >= 0.0f) ? (h_fr[1] != 0) : (h_fr[0] != 0);
        return snap_;
#else
        (void)full_24_fields;
        (void)n_melt_check;
        const auto host_p = particles.download();
        snap_ = compute_hcgf_tmunu_snapshot(n_test, gcfg, mf_cfg, host_p, exclude_frozen);
        snap_.has_backflow_candidates = snap_.has_frozen_particles;
        return snap_;
#endif
    }

private:
    HcgfGridSnapshot snap_;
#if TRANSPORT_HAS_CUDA
    DeviceBuffer<float> d_raw_, d_tmp_, d_psi_, d_out_;
    DeviceBuffer<int> d_has_frozen_;
#endif
};

}  // namespace transport
