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

// Self-describing binary 3D Stress-Energy Tensor & Conserved Currents header for HCGF ("HCGF", 88 bytes)
// Followed immediately by 24 contiguous float32 3D arrays of length nx * ny * nz:
//   [0..9]   T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz  [GeV / fm^3] (kinetic + mean-field)
//   [10..13] JB0, JBx, JBy, JBz                                [fm^-3]      (baryon 4-current)
//   [14..17] JQ0, JQx, JQy, JQz                                [fm^-3]      (electric charge 4-current)
//   [18..23] eps_LRF [GeV/fm^3], P_LRF [GeV/fm^3], nB_LRF [fm^-3], vx_L [c], vy_L [c], vz_L [c]
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
};

namespace hcgf_detail {

// Solve exact 4D Landau rest-frame eigenvalue problem T^mu_nu u^nu = eps_LRF u^mu:
//   eps = T00 - S . v,   (eps * I + T_spatial) v = S,   with |v| < 0.9995
inline void solve_landau_frame(double T00, double Sx, double Sy, double Sz,
                               double Txx, double Txy, double Txz,
                               double Tyy, double Tyz, double Tzz,
                               double& eps_lrf, double& P_lrf,
                               double& vx, double& vy, double& vz) {
    if (T00 <= 1.0e-12) {
        eps_lrf = 0.0; P_lrf = 0.0;
        vx = 0.0; vy = 0.0; vz = 0.0;
        return;
    }
    vx = Sx / T00; vy = Sy / T00; vz = Sz / T00;
    auto clamp_v = [&]() {
        const double v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > 0.9990) {
            const double sc = 0.9995 / std::sqrt(v2);
            vx *= sc; vy *= sc; vz *= sc;
        }
    };
    clamp_v();

    for (int it = 0; it < 8; ++it) {
        double eps = T00 - (Sx * vx + Sy * vy + Sz * vz);
        if (eps < 1.0e-12) eps = 1.0e-12;
        const double a00 = eps + Txx, a01 = Txy,       a02 = Txz;
        const double a11 = eps + Tyy, a12 = Tyz,       a22 = eps + Tzz;
        const double c00 = a11 * a22 - a12 * a12;
        const double c01 = -(a01 * a22 - a12 * a02);
        const double c02 = a01 * a12 - a11 * a02;
        const double c11 = a00 * a22 - a02 * a02;
        const double c12 = -(a00 * a12 - a01 * a02);
        const double c22 = a00 * a11 - a01 * a01;
        const double det = a00 * c00 + a01 * c01 + a02 * c02;
        if (std::fabs(det) <= 1.0e-30) break;
        const double inv_det = 1.0 / det;
        vx = (c00 * Sx + c01 * Sy + c02 * Sz) * inv_det;
        vy = (c01 * Sx + c11 * Sy + c12 * Sz) * inv_det;
        vz = (c02 * Sx + c12 * Sy + c22 * Sz) * inv_det;
        clamp_v();
    }
    eps_lrf = std::max(0.0, T00 - (Sx * vx + Sy * vy + Sz * vz));
    // Exact Lorentz-scalar trace identity: g_munu T^munu = T00 - Tr(T_ij) == eps_LRF - 3 P_LRF
    const double trace_gT = T00 - (Txx + Tyy + Tzz);
    P_lrf = (eps_lrf - trace_gT) / 3.0;
}

}  // namespace hcgf_detail

inline HcgfGridSnapshot compute_hcgf_tmunu_snapshot(int n_test,
                                                    const GridConfig& gcfg,
                                                    const MeanFieldConfig& mf_cfg,
                                                    const std::vector<HostParticle>& host_p) {
    HcgfGridSnapshot snap{};
    const int nc = gcfg.num_cells();
    if (nc <= 0) return snap;
    for (int f = 0; f < 24; ++f) snap.fields[f].assign(nc, 0.0f);

    const double inv_n_dV = 1.0 / (static_cast<double>(n_test) * gcfg.cell_volume_fm3());
    std::vector<double> t00(nc, 0.0), t0x(nc, 0.0), t0y(nc, 0.0), t0z(nc, 0.0);
    std::vector<double> txx(nc, 0.0), txy(nc, 0.0), txz(nc, 0.0);
    std::vector<double> tyy(nc, 0.0), tyz(nc, 0.0), tzz(nc, 0.0);
    std::vector<double> rp(nc, 0.0), rn(nc, 0.0), jb0(nc, 0.0), jbx(nc, 0.0), jby(nc, 0.0), jbz(nc, 0.0);
    std::vector<double> jq0(nc, 0.0), jqx(nc, 0.0), jqy(nc, 0.0), jqz(nc, 0.0);
    std::vector<double> tmp(nc, 0.0);

    for (const auto& hp : host_p) {
        int i0, j0, k0;
        double fx, fy, fz;
        if (!mf_detail::cic_weights(hp.x, hp.y, hp.z, gcfg, i0, j0, k0, fx, fy, fz)) continue;

        const SpeciesInfo sp = get_species_info(hp.species);
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
                    const double w = (((dx == 0) ? (1.0 - fx) : fx) * wy * wz) * inv_n_dV;
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

    for (int c = 0; c < nc; ++c) {
        double T00 = t00[c], T0x = t0x[c], T0y = t0y[c], T0z = t0z[c];
        double Txx = txx[c], Txy = txy[c], Txz = txz[c], Tyy = tyy[c], Tyz = tyz[c], Tzz = tzz[c];

        // Add covariant Skyrme + MD interaction stress-energy tensor T_pot^munu = (V + P_pot) u_B^mu u_B^nu - P_pot g^munu
        const double rb_lab = jb0[c];
        const double jb2 = jbx[c] * jbx[c] + jby[c] * jby[c] + jbz[c] * jbz[c];
        const double nb_baryon = (rb_lab * rb_lab > jb2) ? std::sqrt(rb_lab * rb_lab - jb2) : 0.0;
        if (mf_cfg.enabled && nb_baryon > 1.0e-10 && rb_lab > 1.0e-10) {
            const double u = nb_baryon / mf_cfg.rho0_fm3;
            const double rb_eff = std::max(rp[c] + rn[c], 1.0e-12);
            const double delta = (rn[c] - rp[c]) / rb_eff;
            const double v_pot =
                0.5 * mf_cfg.A_sk_gev * mf_cfg.rho0_fm3 * u * u +
                (mf_cfg.B_sk_gev / (mf_cfg.sigma_exp + 1.0)) * mf_cfg.rho0_fm3 * std::pow(u, mf_cfg.sigma_exp + 1.0) +
                mf_cfg.S_pot_gev * mf_cfg.rho0_fm3 * std::pow(u, mf_cfg.gamma_sym + 1.0) * delta * delta +
                0.5 * mf_cfg.C_md_gev * mf_cfg.rho0_fm3 * u * u * 0.80;
            const double p_pot =
                0.5 * mf_cfg.A_sk_gev * mf_cfg.rho0_fm3 * u * u +
                (mf_cfg.sigma_exp * mf_cfg.B_sk_gev / (mf_cfg.sigma_exp + 1.0)) * mf_cfg.rho0_fm3 * std::pow(u, mf_cfg.sigma_exp + 1.0) +
                mf_cfg.gamma_sym * mf_cfg.S_pot_gev * mf_cfg.rho0_fm3 * std::pow(u, mf_cfg.gamma_sym + 1.0) * delta * delta +
                0.5 * mf_cfg.C_md_gev * mf_cfg.rho0_fm3 * u * u * 0.80;
            const double u0 = rb_lab / nb_baryon;
            const double ux = jbx[c] / nb_baryon, uy = jby[c] / nb_baryon, uz = jbz[c] / nb_baryon;
            const double w_pot = v_pot + p_pot;
            T00 = std::max(1.0e-12, T00 + w_pot * u0 * u0 - p_pot);
            T0x += w_pot * u0 * ux; T0y += w_pot * u0 * uy; T0z += w_pot * u0 * uz;
            Txx += w_pot * ux * ux + p_pot; Txy += w_pot * ux * uy;         Txz += w_pot * ux * uz;
            Tyy += w_pot * uy * uy + p_pot; Tyz += w_pot * uy * uz;         Tzz += w_pot * uz * uz + p_pot;
        }

        double eps_lrf = 0.0, P_lrf = 0.0, vx_L = 0.0, vy_L = 0.0, vz_L = 0.0;
        hcgf_detail::solve_landau_frame(T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz,
                                        eps_lrf, P_lrf, vx_L, vy_L, vz_L);
        const double gam_L = 1.0 / std::sqrt(std::max(1.0e-8, 1.0 - (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L)));
        const double nb_lrf = std::max(0.0, gam_L * (jb0[c] - (jbx[c] * vx_L + jby[c] * vy_L + jbz[c] * vz_L)));

        const double vals[24] = {
            T00, T0x, T0y, T0z, Txx, Txy, Txz, Tyy, Tyz, Tzz,
            jb0[c], jbx[c], jby[c], jbz[c],
            jq0[c], jqx[c], jqy[c], jqz[c],
            eps_lrf, P_lrf, nb_lrf, vx_L, vy_L, vz_L
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

}  // namespace transport
