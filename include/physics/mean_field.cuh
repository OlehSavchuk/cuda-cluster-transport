#pragma once

#include <cmath>
#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"
#include "physics/coulomb_poisson.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/particles.cuh"

namespace transport {

enum class EOSPreset : int {
    None = 0,          // Free cascade (U = 0)
    SkyrmeSoft = 1,    // Local Skyrme only: K_0 = 200 MeV, C_md = 0 (m*/m = 1.0)
    MDSoft = 2,        // GBD / Danielewicz MD-Skyrme Soft:  K_0 = 215 MeV, m*(p_F)/m = 0.70
    MDStiff = 3        // GBD / Danielewicz MD-Skyrme Stiff: K_0 = 380 MeV, m*(p_F)/m = 0.70
};

// Parameters of the Skyrme + Symmetry + Surface + Lorentzian Momentum-Dependent + Coulomb EDF:
//   U_tau(r, p_N) = A_sk * u + B_sk * u^sigma + 2 * S_pot * u^gamma * delta * tau_z
//                   + 0.5 * C_md * [ Psi_md(r) + u / (1 + |p_N - p_f(r)|^2 / Lambda_md^2) ]
//                   + Q_i * Phi_Coul(r),   where -laplacian(Phi_Coul) = 4 pi e^2 rho_Q(r)
// All energy couplings stored in GeV so single-particle energies & forces match (E, p) in GeV!
struct MeanFieldConfig {
    bool enabled = false;
    bool coulomb_enabled = false;  // 3D Poisson electrostatic potential & Vlasov Lorentz/Coulomb force
    EOSPreset preset = EOSPreset::None;
    HD_INLINE bool is_active() const { return enabled || coulomb_enabled; }
    double rho0_fm3 = 0.16;        // Nuclear saturation density [fm^-3]
    double A_sk_gev = -0.2096;     // Attractive two-body Skyrme coefficient [GeV]
    double B_sk_gev = 0.1564;      // Repulsive density-dependent Skyrme coefficient [GeV]
    double sigma_exp = 1.26;       // Power-law exponent sigma (> 1)
    double S_pot_gev = 0.0185;     // Potential symmetry energy coefficient [GeV] (S_tot(rho0) ~ 31.5 MeV)
    double gamma_sym = 0.75;       // Symmetry energy density slope exponent
    double C_md_gev = -0.0750;     // Lorentzian momentum-dependent depth at p=p_f [GeV]
    double Lambda_md_gev = 0.420;  // Lorentzian momentum range scale [GeV/c]
    double K0_mev = 215.0;         // Incompressibility at saturation [MeV]
    int gaussian_smooth_passes = 1;// 3x3x3 Gaussian stencil passes (~1.2 fm finite nuclear range)
};

// Cold symmetric Fermi-gas helper <1 / (1 + p^2/Lambda^2)>_{p <= p_F} and its u-derivatives
inline void cold_fermi_md_factors(double u, double rho0_fm3, double Lambda_gev,
                                  double& alpha, double& u_dalpha, double& u2_d2alpha) {
    const double pF = phys::HBARC * std::cbrt(1.5 * phys::PI * phys::PI * rho0_fm3 * u);
    const double x = pF / Lambda_gev;
    if (x < 1.0e-5) {
        alpha = 1.0 - 0.6 * x * x;
        u_dalpha = -0.4 * x * x;
        u2_d2alpha = -0.4 * x * x * (-1.0 / 3.0);
        return;
    }
    const double x2 = x * x;
    const double x3 = x2 * x;
    alpha = 3.0 * (x - std::atan(x)) / x3;
    // Since x ~ u^(1/3), u * d/du = (1/3) * x * d/dx
    const double x_da_dx = 3.0 / (x2 * (1.0 + x2)) - 3.0 * alpha;
    const double x2_d2a_dx2 =
        -6.0 * (1.0 + 2.0 * x2) / (x2 * (1.0 + x2) * (1.0 + x2)) - 4.0 * x_da_dx + 6.0 * alpha;
    u_dalpha = x_da_dx / 3.0;
    u2_d2alpha = (x2_d2a_dx2 - 2.0 * x_da_dx) / 9.0;
}

// Exact relativistic kinetic + momentum-dependent energy per nucleon [GeV] of cold symmetric matter
inline double cold_kin_plus_md_per_nucleon_gev(double u, double rho0_fm3,
                                               double C_md_gev, double Lambda_md_gev) {
    if (u <= 1.0e-12) return 0.0;
    const double mN = phys::MASS_NUCLEON_AVG;
    const double pF = phys::HBARC * std::cbrt(1.5 * phys::PI * phys::PI * rho0_fm3 * u);

    // 32-point midpoint quadrature for exact relativistic Fermi-sea kinetic energy
    constexpr int N_Q = 32;
    const double dp = pF / N_Q;
    double kin_int = 0.0;
    for (int k = 0; k < N_Q; ++k) {
        const double p = (k + 0.5) * dp;
        kin_int += p * p * (std::sqrt(mN * mN + p * p) - mN) * dp;
    }
    const double e_kin = 3.0 * kin_int / (pF * pF * pF);

    double alpha = 1.0, da = 0.0, d2a = 0.0;
    cold_fermi_md_factors(u, rho0_fm3, Lambda_md_gev, alpha, da, d2a);
    const double e_md = 0.5 * C_md_gev * u * alpha;
    return e_kin + e_md;
}

// Construct self-consistent MeanFieldConfig matching saturation rho_0 = 0.16 fm^-3,
// E_B(rho_0) = -16.0 MeV, P(rho_0) = 0, and target incompressibility K_0!
inline MeanFieldConfig make_mean_field_config(EOSPreset preset) {
    MeanFieldConfig cfg{};
    cfg.preset = preset;
    if (preset == EOSPreset::None) {
        cfg.enabled = false;
        return cfg;
    }
    cfg.enabled = true;
    cfg.rho0_fm3 = 0.16;
    cfg.S_pot_gev = 0.0185;
    cfg.gamma_sym = 0.75;
    cfg.Lambda_md_gev = 0.550;  // 2.09 * p_F (GBD / Danielewicz momentum scale)

    if (preset == EOSPreset::SkyrmeSoft) {
        cfg.C_md_gev = 0.0;
        cfg.K0_mev = 200.0;
    } else if (preset == EOSPreset::MDSoft) {
        // Matches Hama optical potential: U(p=0)=-72 MeV, U(p~0.78 GeV)=0, U(inf)=+36 MeV, m*(p_F)/m=0.69
        cfg.C_md_gev = -0.216;
        cfg.K0_mev = 215.0;
    } else {
        cfg.C_md_gev = -0.216;
        cfg.K0_mev = 380.0;
    }

    constexpr double E_BIND_GEV = -0.0160;
    constexpr double h = 1.0e-4;
    const double e0 = cold_kin_plus_md_per_nucleon_gev(1.0, cfg.rho0_fm3, cfg.C_md_gev, cfg.Lambda_md_gev);
    const double ep = cold_kin_plus_md_per_nucleon_gev(1.0 + h, cfg.rho0_fm3, cfg.C_md_gev, cfg.Lambda_md_gev);
    const double em = cold_kin_plus_md_per_nucleon_gev(1.0 - h, cfg.rho0_fm3, cfg.C_md_gev, cfg.Lambda_md_gev);
    const double de = (ep - em) / (2.0 * h);
    const double d2e = (ep - 2.0 * e0 + em) / (h * h);

    const double K0_gev = cfg.K0_mev * 1.0e-3;
    const double denom = e0 - E_BIND_GEV - de;
    cfg.sigma_exp = (K0_gev / 9.0 - d2e) / denom;
    cfg.B_sk_gev = ((cfg.sigma_exp + 1.0) / (cfg.sigma_exp - 1.0)) * denom;
    cfg.A_sk_gev = -2.0 * (de + (cfg.sigma_exp / (cfg.sigma_exp + 1.0)) * cfg.B_sk_gev);
    return cfg;
}

// Cold symmetric nuclear matter total energy per nucleon E/A(u) - m_N [GeV]
inline double cold_eos_energy_per_nucleon_gev(double u, const MeanFieldConfig& cfg) {
    if (u <= 1.0e-12) return 0.0;
    const double e_km = cold_kin_plus_md_per_nucleon_gev(u, cfg.rho0_fm3, cfg.C_md_gev, cfg.Lambda_md_gev);
    const double e_sk = 0.5 * cfg.A_sk_gev * u +
                        (cfg.B_sk_gev / (cfg.sigma_exp + 1.0)) * std::pow(u, cfg.sigma_exp);
    return e_km + e_sk;
}

struct MeanFieldView {
    MeanFieldConfig cfg;
    GridConfig gcfg;
    Real* rho_p = nullptr;
    Real* rho_n = nullptr;
    Real* px_sum = nullptr;
    Real* py_sum = nullptr;
    Real* pz_sum = nullptr;
    Real* psi_md = nullptr;
    Real* u_loc_p = nullptr;
    Real* u_loc_n = nullptr;
    Real* u_md_dual = nullptr;
    Real* rho_q = nullptr;
    Real* phi_coul = nullptr;
    Real* tmp_buf = nullptr;
};

namespace mf_detail {

HD_INLINE int wrap_or_clamp_idx(int idx, int n, BoundaryMode bmode) {
    if (bmode == BoundaryMode::PeriodicBox) {
        int r = idx % n;
        return (r < 0) ? (r + n) : r;
    }
    if (idx < 0) return 0;
    if (idx >= n) return n - 1;
    return idx;
}

HD_INLINE int cell_flat(int ix, int iy, int iz, const GridConfig& g) {
    return ix + g.nx * (iy + g.ny * iz);
}

// Compute base cell index (i0, j0, k0) and trilinear weights (fx, fy, fz) in [0, 1)
// relative to cell centers x_c = xmin + (i + 0.5) * dx.
HD_INLINE bool cic_weights(double x, double y, double z, const GridConfig& g,
                           int& i0, int& j0, int& k0,
                           double& fx, double& fy, double& fz) {
    if (g.boundary == BoundaryMode::OpenExpanding) {
        if (x < g.xmin || x >= g.xmax || y < g.ymin || y >= g.ymax || z < g.zmin || z >= g.zmax) {
            return false;
        }
    }
    const double gx = (x - g.xmin) / g.dx() - 0.5;
    const double gy = (y - g.ymin) / g.dy() - 0.5;
    const double gz = (z - g.zmin) / g.dz() - 0.5;

    const double flx = ::floor(gx);
    const double fly = ::floor(gy);
    const double flz = ::floor(gz);

    i0 = static_cast<int>(flx);
    j0 = static_cast<int>(fly);
    k0 = static_cast<int>(flz);
    fx = gx - flx;
    fy = gy - fly;
    fz = gz - flz;
    return true;
}

HD_INLINE double sample_cic_scalar(const Real* field, double x, double y, double z,
                                   const GridConfig& g) {
    int i0, j0, k0;
    double fx, fy, fz;
    if (!cic_weights(x, y, z, g, i0, j0, k0, fx, fy, fz)) return 0.0;

    double val = 0.0;
    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = wrap_or_clamp_idx(k0 + dz, g.nz, g.boundary);
        const double wz = (dz == 0) ? (1.0 - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = wrap_or_clamp_idx(j0 + dy, g.ny, g.boundary);
            const double wy = (dy == 0) ? (1.0 - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = wrap_or_clamp_idx(i0 + dx, g.nx, g.boundary);
                const double wx = (dx == 0) ? (1.0 - fx) : fx;
                val += (wx * wy * wz) * static_cast<double>(field[cell_flat(ix, iy, iz, g)]);
            }
        }
    }
    return val;
}

// Compute centered gradient at grid cell (ix, iy, iz)
template <typename FieldT>
HD_INLINE void cell_centered_grad(const FieldT* field, int ix, int iy, int iz,
                                  const GridConfig& g,
                                  double& gx, double& gy, double& gz) {
    const int ixp = wrap_or_clamp_idx(ix + 1, g.nx, g.boundary);
    const int ixm = wrap_or_clamp_idx(ix - 1, g.nx, g.boundary);
    const int iyp = wrap_or_clamp_idx(iy + 1, g.ny, g.boundary);
    const int iym = wrap_or_clamp_idx(iy - 1, g.ny, g.boundary);
    const int izp = wrap_or_clamp_idx(iz + 1, g.nz, g.boundary);
    const int izm = wrap_or_clamp_idx(iz - 1, g.nz, g.boundary);

    gx = (field[cell_flat(ixp, iy, iz, g)] - field[cell_flat(ixm, iy, iz, g)]) / (2.0 * g.dx());
    gy = (field[cell_flat(ix, iyp, iz, g)] - field[cell_flat(ix, iym, iz, g)]) / (2.0 * g.dy());
    gz = (field[cell_flat(ix, iy, izp, g)] - field[cell_flat(ix, iy, izm, g)]) / (2.0 * g.dz());
}

// Exact variational gradient of CIC trilinear interpolation: d/dr (sum_c field[c] * W_CIC(r_c - r))
HD_INLINE void sample_cic_grad(const Real* field, double x, double y, double z,
                               const GridConfig& g,
                               double& out_gx, double& out_gy, double& out_gz) {
    out_gx = 0.0; out_gy = 0.0; out_gz = 0.0;
    int i0, j0, k0;
    double fx, fy, fz;
    if (!cic_weights(x, y, z, g, i0, j0, k0, fx, fy, fz)) return;

    const int ix0 = wrap_or_clamp_idx(i0, g.nx, g.boundary);
    const int ix1 = wrap_or_clamp_idx(i0 + 1, g.nx, g.boundary);
    const int iy0 = wrap_or_clamp_idx(j0, g.ny, g.boundary);
    const int iy1 = wrap_or_clamp_idx(j0 + 1, g.ny, g.boundary);
    const int iz0 = wrap_or_clamp_idx(k0, g.nz, g.boundary);
    const int iz1 = wrap_or_clamp_idx(k0 + 1, g.nz, g.boundary);

    const double c000 = field[cell_flat(ix0, iy0, iz0, g)];
    const double c100 = field[cell_flat(ix1, iy0, iz0, g)];
    const double c010 = field[cell_flat(ix0, iy1, iz0, g)];
    const double c110 = field[cell_flat(ix1, iy1, iz0, g)];
    const double c001 = field[cell_flat(ix0, iy0, iz1, g)];
    const double c101 = field[cell_flat(ix1, iy0, iz1, g)];
    const double c011 = field[cell_flat(ix0, iy1, iz1, g)];
    const double c111 = field[cell_flat(ix1, iy1, iz1, g)];

    out_gx = ((1.0 - fy) * (1.0 - fz) * (c100 - c000) +
              fy         * (1.0 - fz) * (c110 - c010) +
              (1.0 - fy) * fz         * (c101 - c001) +
              fy         * fz         * (c111 - c011)) / g.dx();

    out_gy = ((1.0 - fx) * (1.0 - fz) * (c010 - c000) +
              fx         * (1.0 - fz) * (c110 - c100) +
              (1.0 - fx) * fz         * (c011 - c001) +
              fx         * fz         * (c111 - c101)) / g.dy();

    out_gz = ((1.0 - fx) * (1.0 - fy) * (c001 - c000) +
              fx         * (1.0 - fy) * (c101 - c100) +
              (1.0 - fx) * fy         * (c011 - c010) +
              fx         * fy         * (c111 - c110)) / g.dz();
}

HD_INLINE void deposit_particle_baryon_cic(const ParticleSoAView& p, int i,
                                           double norm_inv_n_dV, MeanFieldView& mf) {
    if (!p.active[i]) return;
    const SpeciesInfo sp = get_species_info(p.species[i]);
    const bool do_nuc = mf.cfg.enabled && (p.freezeout_tag[i] == 0) && (sp.baryon_num > 0);
    const bool do_coul = mf.cfg.coulomb_enabled && (sp.charge != 0);
    if (!do_nuc && !do_coul) return;

    int i0, j0, k0;
    double fx, fy, fz;
    if (!cic_weights(p.x[i], p.y[i], p.z[i], mf.gcfg, i0, j0, k0, fx, fy, fz)) return;

    double zp = 0.0, zn = 0.0, pxw = 0.0, pyw = 0.0, pzw = 0.0;
    if (do_nuc) {
        double wp = 0.0, wn = 0.0;
        get_effective_pn_weights(p.species[i], wp, wn);
        zp = wp * norm_inv_n_dV;
        zn = wn * norm_inv_n_dV;
        pxw = p.px[i] * norm_inv_n_dV;
        pyw = p.py[i] * norm_inv_n_dV;
        pzw = p.pz[i] * norm_inv_n_dV;
    }
    const double zq = do_coul ? (static_cast<double>(sp.charge) * norm_inv_n_dV) : 0.0;

    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = wrap_or_clamp_idx(k0 + dz, mf.gcfg.nz, mf.gcfg.boundary);
        const double wz = (dz == 0) ? (1.0 - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = wrap_or_clamp_idx(j0 + dy, mf.gcfg.ny, mf.gcfg.boundary);
            const double wy = (dy == 0) ? (1.0 - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = wrap_or_clamp_idx(i0 + dx, mf.gcfg.nx, mf.gcfg.boundary);
                const double w = ((dx == 0) ? (1.0 - fx) : fx) * wy * wz;
                const int c = cell_flat(ix, iy, iz, mf.gcfg);
                if (do_nuc) {
                    if (zp > 0.0) portable_atomic_add_real(&mf.rho_p[c], static_cast<Real>(w * zp));
                    if (zn > 0.0) portable_atomic_add_real(&mf.rho_n[c], static_cast<Real>(zn * w));
                    portable_atomic_add_real(&mf.px_sum[c], static_cast<Real>(w * pxw));
                    portable_atomic_add_real(&mf.py_sum[c], static_cast<Real>(w * pyw));
                    portable_atomic_add_real(&mf.pz_sum[c], static_cast<Real>(w * pzw));
                }
                if (do_coul) {
                    portable_atomic_add_real(&mf.rho_q[c], static_cast<Real>(w * zq));
                }
            }
        }
    }
}

HD_INLINE void sample_cic_flow_and_u(const MeanFieldView& mf,
                                     double x, double y, double z,
                                     double& u_out,
                                     double& pfx, double& pfy, double& pfz) {
    u_out = 0.0; pfx = 0.0; pfy = 0.0; pfz = 0.0;
    int i0, j0, k0;
    double fx, fy, fz;
    if (!cic_weights(x, y, z, mf.gcfg, i0, j0, k0, fx, fy, fz)) return;

    double rho_tot = 0.0, pi_x = 0.0, pi_y = 0.0, pi_z = 0.0;
    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = wrap_or_clamp_idx(k0 + dz, mf.gcfg.nz, mf.gcfg.boundary);
        const double wz = (dz == 0) ? (1.0 - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = wrap_or_clamp_idx(j0 + dy, mf.gcfg.ny, mf.gcfg.boundary);
            const double wy = (dy == 0) ? (1.0 - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = wrap_or_clamp_idx(i0 + dx, mf.gcfg.nx, mf.gcfg.boundary);
                const double w = ((dx == 0) ? (1.0 - fx) : fx) * wy * wz;
                const int c = cell_flat(ix, iy, iz, mf.gcfg);
                rho_tot += w * (mf.rho_p[c] + mf.rho_n[c]);
                pi_x += w * mf.px_sum[c];
                pi_y += w * mf.py_sum[c];
                pi_z += w * mf.pz_sum[c];
            }
        }
    }
    u_out = rho_tot / mf.cfg.rho0_fm3;
    if (rho_tot > 1.0e-10) {
        const double inv_r = 1.0 / rho_tot;
        pfx = pi_x * inv_r;
        pfy = pi_y * inv_r;
        pfz = pi_z * inv_r;
    }
}

HD_INLINE void deposit_particle_md_psi_cic(const ParticleSoAView& p, int i,
                                           double norm_inv_n_dV_rho0, MeanFieldView& mf) {
    if (!p.active[i] || p.freezeout_tag[i] != 0) return;
    int i0, j0, k0;
    double fx, fy, fz;
    if (!cic_weights(p.x[i], p.y[i], p.z[i], mf.gcfg, i0, j0, k0, fx, fy, fz)) return;

    double u_loc, pfx, pfy, pfz;
    sample_cic_flow_and_u(mf, p.x[i], p.y[i], p.z[i], u_loc, pfx, pfy, pfz);

    const SpeciesInfo sp = get_species_info(p.species[i]);
    if (sp.baryon_num == 0) return;
    const double inv_A = 1.0 / static_cast<double>(sp.baryon_num);
    const double qx = p.px[i] * inv_A - pfx;
    const double qy = p.py[i] * inv_A - pfy;
    const double qz = p.pz[i] * inv_A - pfz;
    const double q2 = qx * qx + qy * qy + qz * qz;
    const double inv_lam2 = 1.0 / (mf.cfg.Lambda_md_gev * mf.cfg.Lambda_md_gev);
    const double contrib = static_cast<double>(sp.baryon_num) * norm_inv_n_dV_rho0 / (1.0 + q2 * inv_lam2);

    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = wrap_or_clamp_idx(k0 + dz, mf.gcfg.nz, mf.gcfg.boundary);
        const double wz = (dz == 0) ? (1.0 - fz) : fz;
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = wrap_or_clamp_idx(j0 + dy, mf.gcfg.ny, mf.gcfg.boundary);
            const double wy = (dy == 0) ? (1.0 - fy) : fy;
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = wrap_or_clamp_idx(i0 + dx, mf.gcfg.nx, mf.gcfg.boundary);
                const double w = ((dx == 0) ? (1.0 - fx) : fx) * wy * wz;
                portable_atomic_add_real(&mf.psi_md[cell_flat(ix, iy, iz, mf.gcfg)], static_cast<Real>(w * contrib));
            }
        }
    }
}

// Separable 1D [1/4, 1/2, 1/4] Gaussian-like smoothing pass along axis in {0, 1, 2}
template <typename FieldT>
HD_INLINE void smooth_axis_cell(const FieldT* src, FieldT* dst, int cid,
                                int axis, const GridConfig& g) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const int ix = cid % nx;
    const int iy = (cid / nx) % ny;
    const int iz = cid / (nx * ny);

    int ip_x = ix, ip_y = iy, ip_z = iz;
    int im_x = ix, im_y = iy, im_z = iz;
    if (axis == 0) {
        ip_x = wrap_or_clamp_idx(ix + 1, nx, g.boundary);
        im_x = wrap_or_clamp_idx(ix - 1, nx, g.boundary);
    } else if (axis == 1) {
        ip_y = wrap_or_clamp_idx(iy + 1, ny, g.boundary);
        im_y = wrap_or_clamp_idx(iy - 1, ny, g.boundary);
    } else {
        ip_z = wrap_or_clamp_idx(iz + 1, nz, g.boundary);
        im_z = wrap_or_clamp_idx(iz - 1, nz, g.boundary);
    }

    dst[cid] = FieldT(0.50) * src[cid] +
               FieldT(0.25) * src[cell_flat(ip_x, ip_y, ip_z, g)] +
               FieldT(0.25) * src[cell_flat(im_x, im_y, im_z, g)];
}

HD_INLINE void eval_node_potentials(MeanFieldView& mf, int c) {
    const double rp = mf.rho_p[c];
    const double rn = mf.rho_n[c];
    const double rb = rp + rn;
    if (rb <= 1.0e-12) {
        mf.u_loc_p[c] = Real(0);
        mf.u_loc_n[c] = Real(0);
        mf.u_md_dual[c] = Real(0);
        return;
    }
    const double u = rb / mf.cfg.rho0_fm3;
    const double delta = (rn - rp) / rb;
    const double u_sk = mf.cfg.A_sk_gev * u + mf.cfg.B_sk_gev * ::pow(u, mf.cfg.sigma_exp);
    const double u_sym = 2.0 * mf.cfg.S_pot_gev * ::pow(u, mf.cfg.gamma_sym) * delta;
    const double u_md_bg = 0.5 * mf.cfg.C_md_gev * static_cast<double>(mf.psi_md[c]);

    mf.u_loc_p[c] = static_cast<Real>(u_sk - u_sym + u_md_bg);
    mf.u_loc_n[c] = static_cast<Real>(u_sk + u_sym + u_md_bg);
    mf.u_md_dual[c] = static_cast<Real>(0.5 * mf.cfg.C_md_gev * u);
}

// Symplectic momentum kick dp/dt = -grad_r U_cluster(A, Z, r, p) - Q * grad_r Phi_Coul(r) over dt
HD_INLINE void kick_particle_vlasov(ParticleSoAView& p, int i, double dt,
                                    const MeanFieldView& mf, double* cm_accum) {
    if (!p.active[i]) return;
    const SpeciesInfo sp = get_species_info(p.species[i]);
    const bool do_nuc = mf.cfg.enabled && (p.freezeout_tag[i] == 0) && (sp.baryon_num > 0);
    const bool do_coul = mf.cfg.coulomb_enabled && (sp.charge != 0);
    if (!do_nuc && !do_coul) return;

    const double x = p.x[i], y = p.y[i], z = p.z[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    if (do_coul) {
        double gcx = 0.0, gcy = 0.0, gcz = 0.0;
        sample_cic_grad(mf.phi_coul, x, y, z, mf.gcfg, gcx, gcy, gcz);
        const double Q = static_cast<double>(sp.charge);
        fx -= Q * gcx;
        fy -= Q * gcy;
        fz -= Q * gcz;
    }

    if (do_nuc) {
        double Z = 0.0, N = 0.0;
        get_effective_pn_weights(p.species[i], Z, N);
        const double A = static_cast<double>(sp.baryon_num);

        double gpx = 0.0, gpy = 0.0, gpz = 0.0;
        double gnx = 0.0, gny = 0.0, gnz = 0.0;
        sample_cic_grad(mf.u_loc_p, x, y, z, mf.gcfg, gpx, gpy, gpz);
        sample_cic_grad(mf.u_loc_n, x, y, z, mf.gcfg, gnx, gny, gnz);

        fx += -Z * gpx - N * gnx;
        fy += -Z * gpy - N * gny;
        fz += -Z * gpz - N * gnz;

        if (::fabs(mf.cfg.C_md_gev) > 1.0e-12) {
            double u_loc, pfx, pfy, pfz;
            sample_cic_flow_and_u(mf, x, y, z, u_loc, pfx, pfy, pfz);
            double gmdx, gmdy, gmdz;
            sample_cic_grad(mf.u_md_dual, x, y, z, mf.gcfg, gmdx, gmdy, gmdz);

            const double qx = p.px[i] / A - pfx;
            const double qy = p.py[i] / A - pfy;
            const double qz = p.pz[i] / A - pfz;
            const double inv_lam2 = 1.0 / (mf.cfg.Lambda_md_gev * mf.cfg.Lambda_md_gev);
            const double denom = 1.0 + (qx * qx + qy * qy + qz * qz) * inv_lam2;
            const double md_coef = -A / denom;
            fx += md_coef * gmdx;
            fy += md_coef * gmdy;
            fz += md_coef * gmdz;

            // Product-rule force from spatial variation of local collective frame p_f(r) = Pi(r)/rho_B(r)
            const double rho_loc = u_loc * mf.cfg.rho0_fm3;
            if (rho_loc > 1.0e-4) {
                const double u_dual = sample_cic_scalar(mf.u_md_dual, x, y, z, mf.gcfg);
                double grpx, grpy, grpz, grnx, grny, grnz;
                sample_cic_grad(mf.rho_p, x, y, z, mf.gcfg, grpx, grpy, grpz);
                sample_cic_grad(mf.rho_n, x, y, z, mf.gcfg, grnx, grny, grnz);
                const double grx = grpx + grnx, gry = grpy + grny, grz = grpz + grnz;

                double gpix_x, gpix_y, gpix_z, gpiy_x, gpiy_y, gpiy_z, gpiz_x, gpiz_y, gpiz_z;
                sample_cic_grad(mf.px_sum, x, y, z, mf.gcfg, gpix_x, gpix_y, gpix_z);
                sample_cic_grad(mf.py_sum, x, y, z, mf.gcfg, gpiy_x, gpiy_y, gpiy_z);
                sample_cic_grad(mf.pz_sum, x, y, z, mf.gcfg, gpiz_x, gpiz_y, gpiz_z);

                const double inv_rho = 1.0 / rho_loc;
                const double q_dpf_dx = (qx * (gpix_x - pfx * grx) + qy * (gpiy_x - pfy * grx) + qz * (gpiz_x - pfz * grx)) * inv_rho;
                const double q_dpf_dy = (qx * (gpix_y - pfx * gry) + qy * (gpiy_y - pfy * gry) + qz * (gpiz_y - pfz * gry)) * inv_rho;
                const double q_dpf_dz = (qx * (gpix_z - pfx * grz) + qy * (gpiy_z - pfy * grz) + qz * (gpiz_z - pfz * grz)) * inv_rho;
                const double pf_coef = -(2.0 * A * u_dual * inv_lam2) / (denom * denom);
                fx += pf_coef * q_dpf_dx;
                fy += pf_coef * q_dpf_dy;
                fz += pf_coef * q_dpf_dz;
            }
        }
    }

    const double dpx = fx * dt;
    const double dpy = fy * dt;
    const double dpz = fz * dt;
    const double w_cm = static_cast<double>(p.mass[i]);
    p.px[i] += dpx;
    p.py[i] += dpy;
    p.pz[i] += dpz;
    portable_atomic_add_double(&cm_accum[0], dpx);
    portable_atomic_add_double(&cm_accum[1], dpy);
    portable_atomic_add_double(&cm_accum[2], dpz);
    portable_atomic_add_double(&cm_accum[3], w_cm);
}

// Subtract spurious discrete-lattice net impulse proportional to mass so total 3-momentum is exact to ~1e-14
HD_INLINE void remove_lattice_cm_impulse(ParticleSoAView& p, int i, const MeanFieldConfig& cfg,
                                         double corr_x, double corr_y, double corr_z) {
    if (!p.active[i]) return;
    const SpeciesInfo sp = get_species_info(p.species[i]);
    const bool do_nuc = cfg.enabled && (p.freezeout_tag[i] == 0) && (sp.baryon_num > 0);
    const bool do_coul = cfg.coulomb_enabled && (sp.charge != 0);
    if (!do_nuc && !do_coul) return;
    const double w_cm = static_cast<double>(p.mass[i]);
    p.px[i] -= w_cm * corr_x;
    p.py[i] -= w_cm * corr_y;
    p.pz[i] -= w_cm * corr_z;
    const double m = p.mass[i];
    p.E[i] = ::sqrt(m * m + p.px[i] * p.px[i] + p.py[i] * p.py[i] + p.pz[i] * p.pz[i]);
}

// Symplectic coordinate drift dx/dt = p/E + dU_MD/dp (includes m* < m group-velocity boost!)
HD_INLINE void drift_particle_vlasov(ParticleSoAView& p, int i, double dt, const MeanFieldView& mf) {
    if (!p.active[i]) return;
    const double inv_E = 1.0 / p.E[i];
    double vx = p.px[i] * inv_E;
    double vy = p.py[i] * inv_E;
    double vz = p.pz[i] * inv_E;

    const int bnum = get_species_info(p.species[i]).baryon_num;
    if (bnum > 0 && p.freezeout_tag[i] == 0 && mf.cfg.enabled && ::fabs(mf.cfg.C_md_gev) > 1.0e-12) {
        double u_loc, pfx, pfy, pfz;
        sample_cic_flow_and_u(mf, p.x[i], p.y[i], p.z[i], u_loc, pfx, pfy, pfz);
        const double A = static_cast<double>(bnum);
        const double qx = p.px[i] / A - pfx;
        const double qy = p.py[i] / A - pfy;
        const double qz = p.pz[i] / A - pfz;
        const double inv_lam2 = 1.0 / (mf.cfg.Lambda_md_gev * mf.cfg.Lambda_md_gev);
        const double denom = 1.0 + (qx * qx + qy * qy + qz * qz) * inv_lam2;
        const double dv_coef = -(mf.cfg.C_md_gev * u_loc * inv_lam2) / (denom * denom);

        vx += dv_coef * qx;
        vy += dv_coef * qy;
        vz += dv_coef * qz;

        // Causality limiter |v_eff| <= 0.9995 c
        const double v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > 0.9990) {
            const double scale = 0.9995 / ::sqrt(v2);
            vx *= scale; vy *= scale; vz *= scale;
        }
    }

    double nx = p.x[i] + vx * dt;
    double ny = p.y[i] + vy * dt;
    double nz = p.z[i] + vz * dt;
    if (mf.gcfg.boundary == BoundaryMode::PeriodicBox) {
        nx = detail::wrap_periodic(nx, mf.gcfg.xmin, mf.gcfg.lx());
        ny = detail::wrap_periodic(ny, mf.gcfg.ymin, mf.gcfg.ly());
        nz = detail::wrap_periodic(nz, mf.gcfg.zmin, mf.gcfg.lz());
    }
    p.x[i] = nx;
    p.y[i] = ny;
    p.z[i] = nz;
    p.t[i] += dt;
}

// Single-particle potential U_i(r, p) = dE_pot / df_i [GeV] matching compute_potential_energy_test_gev
HD_INLINE double eval_particle_potential_gev(const MeanFieldView& mf,
                                             int species_id,
                                             double x, double y, double z,
                                             double px, double py, double pz) {
    const SpeciesInfo sp = get_species_info(species_id);
    double u_tot = 0.0;
    if (mf.cfg.coulomb_enabled && sp.charge != 0 && mf.phi_coul != nullptr) {
        u_tot += static_cast<double>(sp.charge) * sample_cic_scalar(mf.phi_coul, x, y, z, mf.gcfg);
    }
    if (!mf.cfg.enabled || sp.baryon_num == 0) return u_tot;
    double Z = 0.0, N = 0.0;
    get_effective_pn_weights(species_id, Z, N);
    const double A = static_cast<double>(sp.baryon_num);

    u_tot += Z * sample_cic_scalar(mf.u_loc_p, x, y, z, mf.gcfg) +
             N * sample_cic_scalar(mf.u_loc_n, x, y, z, mf.gcfg);
    if (::fabs(mf.cfg.C_md_gev) > 1.0e-12) {
        double u_loc, pfx, pfy, pfz;
        sample_cic_flow_and_u(mf, x, y, z, u_loc, pfx, pfy, pfz);
        const double u_dual = sample_cic_scalar(mf.u_md_dual, x, y, z, mf.gcfg);
        const double qx = px / A - pfx;
        const double qy = py / A - pfy;
        const double qz = pz / A - pfz;
        const double denom = 1.0 + (qx * qx + qy * qy + qz * qz) /
                                   (mf.cfg.Lambda_md_gev * mf.cfg.Lambda_md_gev);
        u_tot += (A * u_dual) / denom;
    }
    return u_tot;
}

// Solve omega_3(r3, p3(q)) + omega_4(r4, p4(q)) == omega_target with exact p3(q) + p4(q) == P
HD_INLINE bool solve_in_medium_2body_momenta(const MeanFieldView& mf,
                                             int id3, double m3, double x3, double y3, double z3,
                                             int id4, double m4, double x4, double y4, double z4,
                                             double Px, double Py, double Pz,
                                             double nx, double ny, double nz,
                                             double q_vac_guess, double omega_target,
                                             Real& E3, Real& p3x, Real& p3y, Real& p3z,
                                             Real& E4, Real& p4x, Real& p4y, Real& p4z) {
    const double P2 = Px * Px + Py * Py + Pz * Pz;
    auto eval_q = [&](double q) {
        const double e3s = ::sqrt(m3 * m3 + q * q);
        const double e4s = ::sqrt(m4 * m4 + q * q);
        const double W = e3s + e4s;
        const double inv_W = 1.0 / W;
        const double ux = Px * inv_W, uy = Py * inv_W, uz = Pz * inv_W;
        const double gam = ::sqrt(1.0 + P2 * inv_W * inv_W);
        const double qnx = q * nx, qny = q * ny, qnz = q * nz;
        const double alpha = (ux * qnx + uy * qny + uz * qnz) / (1.0 + gam);
        const double d3x = qnx + (alpha + e3s) * ux, d3y = qny + (alpha + e3s) * uy, d3z = qnz + (alpha + e3s) * uz;
        const double d4x = -qnx + (-alpha + e4s) * ux, d4y = -qny + (-alpha + e4s) * uy, d4z = -qnz + (-alpha + e4s) * uz;
        const double dE3 = ::sqrt(m3 * m3 + d3x * d3x + d3y * d3y + d3z * d3z);
        const double dE4 = ::sqrt(m4 * m4 + d4x * d4x + d4y * d4y + d4z * d4z);
        E3 = static_cast<Real>(dE3); p3x = static_cast<Real>(d3x); p3y = static_cast<Real>(d3y); p3z = static_cast<Real>(d3z);
        E4 = static_cast<Real>(dE4); p4x = static_cast<Real>(d4x); p4y = static_cast<Real>(d4y); p4z = static_cast<Real>(d4z);
        return (dE3 + eval_particle_potential_gev(mf, id3, x3, y3, z3, d3x, d3y, d3z)) +
               (dE4 + eval_particle_potential_gev(mf, id4, x4, y4, z4, d4x, d4y, d4z)) - omega_target;
    };

    double q_lo = 0.0, f_lo = eval_q(0.0);
    if (f_lo > 0.0) return false;
    double q_hi = (q_vac_guess > 0.02 ? q_vac_guess : 0.02) + 0.25;
    double f_hi = eval_q(q_hi);
    for (int b = 0; b < 6 && f_hi < 0.0; ++b) { q_hi *= 2.0; f_hi = eval_q(q_hi); }
    if (f_hi < 0.0) return false;

    double q_cur = (q_vac_guess > 0.0 && q_vac_guess < q_hi) ? q_vac_guess : 0.5 * q_hi;
    constexpr double f_tol = (sizeof(Real) == 4) ? 2.0e-6 : 1.0e-13;
    constexpr double q_tol = (sizeof(Real) == 4) ? 1.0e-6 : 1.0e-13;
    for (int it = 0; it < 14; ++it) {
        const double f_cur = eval_q(q_cur);
        if (::fabs(f_cur) < f_tol || (q_hi - q_lo) < q_tol) return true;
        if (f_cur < 0.0) { q_lo = q_cur; f_lo = f_cur; } else { q_hi = q_cur; f_hi = f_cur; }
        const double df = f_hi - f_lo;
        const double q_sec = (::fabs(df) > 1.0e-15) ? (q_lo - f_lo * (q_hi - q_lo) / df) : 0.5 * (q_lo + q_hi);
        const double pad = 0.10 * (q_hi - q_lo);
        q_cur = (q_sec > q_lo + pad && q_sec < q_hi - pad) ? q_sec : 0.5 * (q_lo + q_hi);
    }
    eval_q(0.5 * (q_lo + q_hi));
    return true;
}

HD_INLINE void accum_kin_cm_particle(const ParticleSoAView& p, int i, double* acc5) {
    if (!p.active[i]) return;
    const int bnum = get_species_info(p.species[i]).baryon_num;
    const double W = (bnum > 0) ? static_cast<double>(bnum) : 1.0;
    portable_atomic_add_double(&acc5[0], p.E[i]);
    portable_atomic_add_double(&acc5[1], p.px[i]);
    portable_atomic_add_double(&acc5[2], p.py[i]);
    portable_atomic_add_double(&acc5[3], p.pz[i]);
    portable_atomic_add_double(&acc5[4], W);
}

HD_INLINE void accum_peculiar_virial_particle(const ParticleSoAView& p, int i,
                                              double cx, double cy, double cz, double* virial_acc) {
    if (!p.active[i]) return;
    const int bnum = get_species_info(p.species[i]).baryon_num;
    const double W = (bnum > 0) ? static_cast<double>(bnum) : 1.0;
    const double dpx = p.px[i] - W * cx, dpy = p.py[i] - W * cy, dpz = p.pz[i] - W * cz;
    const double v = (p.px[i] * dpx + p.py[i] * dpy + p.pz[i] * dpz) / p.E[i];
    portable_atomic_add_double(virial_acc, v);
}

HD_INLINE void scale_peculiar_momentum_particle(ParticleSoAView& p, int i,
                                                double cx, double cy, double cz, double lam) {
    if (!p.active[i]) return;
    const int bnum = get_species_info(p.species[i]).baryon_num;
    const double W = (bnum > 0) ? static_cast<double>(bnum) : 1.0;
    const double mx = W * cx, my = W * cy, mz = W * cz;
    p.px[i] = mx + lam * (p.px[i] - mx);
    p.py[i] = my + lam * (p.py[i] - my);
    p.pz[i] = mz + lam * (p.pz[i] - mz);
    const double m = p.mass[i];
    p.E[i] = ::sqrt(m * m + p.px[i] * p.px[i] + p.py[i] * p.py[i] + p.pz[i] * p.pz[i]);
}

#if TRANSPORT_HAS_CUDA
__global__ void deposit_baryon_cic_kernel(ParticleSoAView p, double norm_inv, MeanFieldView mf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) deposit_particle_baryon_cic(p, i, norm_inv, mf);
}

__global__ void deposit_md_psi_cic_kernel(ParticleSoAView p, double norm_inv, MeanFieldView mf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) deposit_particle_md_psi_cic(p, i, norm_inv, mf);
}

__global__ void smooth_axis_kernel(const Real* src, Real* dst, int n_cells, int axis, GridConfig g) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_cells) smooth_axis_cell(src, dst, c, axis, g);
}

__global__ void eval_potentials_kernel(MeanFieldView mf, int n_cells) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_cells) eval_node_potentials(mf, c);
}

__global__ void kick_vlasov_kernel(ParticleSoAView p, double dt, MeanFieldView mf, double* cm_accum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) kick_particle_vlasov(p, i, dt, mf, cm_accum);
}

__global__ void remove_cm_impulse_kernel(ParticleSoAView p, MeanFieldConfig cfg, double cx, double cy, double cz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) remove_lattice_cm_impulse(p, i, cfg, cx, cy, cz);
}

__global__ void drift_vlasov_kernel(ParticleSoAView p, double dt, MeanFieldView mf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) drift_particle_vlasov(p, i, dt, mf);
}

__global__ void accum_kin_cm_kernel(ParticleSoAView p, double* acc5) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) accum_kin_cm_particle(p, i, acc5);
}

__global__ void accum_virial_kernel(ParticleSoAView p, double cx, double cy, double cz, double* vir) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) accum_peculiar_virial_particle(p, i, cx, cy, cz, vir);
}

__global__ void scale_peculiar_kernel(ParticleSoAView p, double cx, double cy, double cz, double lam) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) scale_peculiar_momentum_particle(p, i, cx, cy, cz, lam);
}
#endif

}  // namespace mf_detail

class MeanFieldGrid {
public:
    MeanFieldGrid() = default;
    MeanFieldGrid(const MeanFieldConfig& cfg, const GridConfig& gcfg) { configure(cfg, gcfg); }

    void configure(const MeanFieldConfig& cfg, const GridConfig& gcfg) {
        cfg_ = cfg;
        gcfg_ = gcfg;
        const int nc = gcfg_.num_cells();
        rho_p_.resize(nc);
        rho_n_.resize(nc);
        px_sum_.resize(nc);
        py_sum_.resize(nc);
        pz_sum_.resize(nc);
        psi_md_.resize(nc);
        u_loc_p_.resize(nc); u_loc_p_.zero();
        u_loc_n_.resize(nc); u_loc_n_.zero();
        u_md_dual_.resize(nc); u_md_dual_.zero();
        rho_q_.resize(nc); rho_q_.zero();
        phi_coul_raw_.resize(nc); phi_coul_raw_.zero();
        phi_coul_.resize(nc); phi_coul_.zero();
        tmp_buf_.resize(nc); tmp_buf_.zero();
        cm_accum_.resize(6); cm_accum_.zero();
        if (cfg_.coulomb_enabled && nc > 0) {
            poisson_solver_.configure(gcfg_);
        }
    }

    void rebuild_potentials(ParticleSystem& particles, int n_test) {
        if (!cfg_.is_active()) return;
        const int np = particles.slot_size();
        const int nc = gcfg_.num_cells();
        if (nc == 0) return;

        if (cfg_.enabled) {
            rho_p_.zero(); rho_n_.zero();
            px_sum_.zero(); py_sum_.zero(); pz_sum_.zero();
            psi_md_.zero();
        }
        if (cfg_.coulomb_enabled) {
            rho_q_.zero();
        }

        if (np == 0) {
            u_loc_p_.zero(); u_loc_n_.zero(); u_md_dual_.zero();
            phi_coul_raw_.zero(); phi_coul_.zero();
            return;
        }

        ParticleSoAView pv = particles.view();
        MeanFieldView mfv = view();
        const double inv_n_dV = 1.0 / (static_cast<double>(n_test) * gcfg_.cell_volume_fm3());

#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        const int p_blocks = (np + TPB - 1) / TPB;
        const int c_blocks = (nc + TPB - 1) / TPB;
        mf_detail::deposit_baryon_cic_kernel<<<p_blocks, TPB>>>(pv, inv_n_dV, mfv);
        CUDA_CHECK(cudaGetLastError());
#else
        for (int i = 0; i < np; ++i) mf_detail::deposit_particle_baryon_cic(pv, i, inv_n_dV, mfv);
#endif

        if (cfg_.enabled) {
            for (int pass = 0; pass < cfg_.gaussian_smooth_passes; ++pass) {
                smooth_field_3d(rho_p_);
                smooth_field_3d(rho_n_);
                smooth_field_3d(px_sum_);
                smooth_field_3d(py_sum_);
                smooth_field_3d(pz_sum_);
            }

            if (std::fabs(cfg_.C_md_gev) > 1.0e-12) {
                const double inv_md = inv_n_dV / cfg_.rho0_fm3;
#if TRANSPORT_HAS_CUDA
                mf_detail::deposit_md_psi_cic_kernel<<<p_blocks, TPB>>>(pv, inv_md, mfv);
                CUDA_CHECK(cudaGetLastError());
#else
                for (int i = 0; i < np; ++i) mf_detail::deposit_particle_md_psi_cic(pv, i, inv_md, mfv);
#endif
                for (int pass = 0; pass < cfg_.gaussian_smooth_passes; ++pass) {
                    smooth_field_3d(psi_md_);
                }
            }

#if TRANSPORT_HAS_CUDA
            mf_detail::eval_potentials_kernel<<<c_blocks, TPB>>>(mfv, nc);
            CUDA_CHECK(cudaGetLastError());
#else
            for (int c = 0; c < nc; ++c) mf_detail::eval_node_potentials(mfv, c);
#endif
            // Apply self-adjoint Gaussian filter G^T = G to node potentials for exact variational forces!
            for (int pass = 0; pass < cfg_.gaussian_smooth_passes; ++pass) {
                smooth_field_3d(u_loc_p_);
                smooth_field_3d(u_loc_n_);
                if (std::fabs(cfg_.C_md_gev) > 1.0e-12) smooth_field_3d(u_md_dual_);
            }
        }

        if (cfg_.coulomb_enabled) {
            for (int pass = 0; pass < cfg_.gaussian_smooth_passes; ++pass) {
                smooth_field_3d(rho_q_);
            }
            poisson_solver_.solve(rho_q_, phi_coul_raw_);
#if TRANSPORT_HAS_CUDA
            CUDA_CHECK(cudaMemcpy(phi_coul_.data(), phi_coul_raw_.data(), nc * sizeof(Real), cudaMemcpyDeviceToDevice));
#else
            std::memcpy(phi_coul_.data(), phi_coul_raw_.data(), nc * sizeof(Real));
#endif
            for (int pass = 0; pass < cfg_.gaussian_smooth_passes; ++pass) {
                smooth_field_3d(phi_coul_);
            }
        }
    }

    void kick(ParticleSystem& particles, double dt) {
        if (!cfg_.is_active()) return;
        const int np = particles.slot_size();
        if (np == 0) return;
        ParticleSoAView pv = particles.view();
        MeanFieldView mfv = view();
        cm_accum_.zero();
#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        const int p_blocks = (np + TPB - 1) / TPB;
        mf_detail::kick_vlasov_kernel<<<p_blocks, TPB>>>(pv, dt, mfv, cm_accum_.data());
        CUDA_CHECK(cudaGetLastError());
        double h_cm[4] = {};
        cm_accum_.copy_to_host(h_cm, 4);
        const double inv_W = (h_cm[3] > 0.0) ? (1.0 / h_cm[3]) : 0.0;
        mf_detail::remove_cm_impulse_kernel<<<p_blocks, TPB>>>(
            pv, cfg_, h_cm[0] * inv_W, h_cm[1] * inv_W, h_cm[2] * inv_W);
        CUDA_CHECK(cudaGetLastError());
#else
        double* cm = cm_accum_.data();
        for (int i = 0; i < np; ++i) mf_detail::kick_particle_vlasov(pv, i, dt, mfv, cm);
        const double inv_W = (cm[3] > 0.0) ? (1.0 / cm[3]) : 0.0;
        for (int i = 0; i < np; ++i) {
            mf_detail::remove_lattice_cm_impulse(pv, i, cfg_, cm[0] * inv_W, cm[1] * inv_W, cm[2] * inv_W);
        }
#endif
    }

    void drift(ParticleSystem& particles, double dt) {
        const int np = particles.slot_size();
        if (np == 0) return;
        ParticleSoAView pv = particles.view();
        MeanFieldView mfv = view();
#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        mf_detail::drift_vlasov_kernel<<<(np + TPB - 1) / TPB, TPB>>>(pv, dt, mfv);
        CUDA_CHECK(cudaGetLastError());
#else
        for (int i = 0; i < np; ++i) mf_detail::drift_particle_vlasov(pv, i, dt, mfv);
#endif
    }

    // Electrostatic field energy (1/2) int rho_Q(r) * Phi_Coul(r) d^3r in test-particle GeV units
    double compute_coulomb_energy_test_gev(int n_test) const {
        if (!cfg_.coulomb_enabled) return 0.0;
        const auto hrq = rho_q_.to_vector();
        const auto hphi = phi_coul_raw_.to_vector();
        const double dV = gcfg_.cell_volume_fm3();
        double e_coul = 0.0;
        for (std::size_t c = 0; c < hrq.size(); ++c) {
            e_coul += 0.5 * static_cast<double>(hrq[c]) * static_cast<double>(hphi[c]) * dV;
        }
        return static_cast<double>(n_test) * e_coul;
    }

    // Total potential energy in test-particle GeV units (matches GlobalObservables::E_tot scale!)
    double compute_potential_energy_test_gev(int n_test) const {
        double e_tot = compute_coulomb_energy_test_gev(n_test);
        if (!cfg_.enabled) return e_tot;
        const auto hrp = rho_p_.to_vector();
        const auto hrn = rho_n_.to_vector();
        const auto hpsi = psi_md_.to_vector();
        const double dV = gcfg_.cell_volume_fm3();
        double e_phys = 0.0;
        for (std::size_t c = 0; c < hrp.size(); ++c) {
            const double rb = hrp[c] + hrn[c];
            if (rb <= 1.0e-12) continue;
            const double u = rb / cfg_.rho0_fm3;
            const double delta = (hrn[c] - hrp[c]) / rb;
            const double edens =
                0.5 * cfg_.A_sk_gev * cfg_.rho0_fm3 * u * u +
                (cfg_.B_sk_gev / (cfg_.sigma_exp + 1.0)) * cfg_.rho0_fm3 * std::pow(u, cfg_.sigma_exp + 1.0) +
                cfg_.S_pot_gev * cfg_.rho0_fm3 * std::pow(u, cfg_.gamma_sym + 1.0) * delta * delta +
                0.5 * cfg_.C_md_gev * rb * hpsi[c];
            e_phys += edens * dV;
        }
        return e_tot + static_cast<double>(n_test) * e_phys;
    }

    // Symplectic manifold projection: rescales peculiar momenta p_i - A_i*p_cm so H(t) == H_target
    // to machine precision (<1e-12) while keeping total 3-momentum P_tot, B, and Q strictly invariant.
    void project_exact_hamiltonian(ParticleSystem& particles, int n_test, double H_target_test_gev) {
        if (!cfg_.is_active()) return;
        const int np = particles.slot_size();
        if (np == 0) return;
        ParticleSoAView pv = particles.view();

        double prev_dH = 0.0, prev_step = 0.0;
        for (int it = 0; it < 6; ++it) {
            rebuild_potentials(particles, n_test);
            cm_accum_.zero();
#if TRANSPORT_HAS_CUDA
            constexpr int TPB = 256;
            const int blocks = (np + TPB - 1) / TPB;
            mf_detail::accum_kin_cm_kernel<<<blocks, TPB>>>(pv, cm_accum_.data());
            CUDA_CHECK(cudaGetLastError());
            double h5[6] = {};
            cm_accum_.copy_to_host(h5, 5);
#else
            double* h5 = cm_accum_.data();
            for (int i = 0; i < np; ++i) mf_detail::accum_kin_cm_particle(pv, i, h5);
#endif
            if (h5[4] <= 0.0) return;
            const double E_pot = compute_potential_energy_test_gev(n_test);
            const double dH = (h5[0] + E_pot) - H_target_test_gev;
            constexpr double rel_tol = (sizeof(Real) == 4) ? 5.0e-7 : 1.0e-13;
            if (std::fabs(dH) <= rel_tol * std::fabs(H_target_test_gev)) return;

            const double cx = h5[1] / h5[4], cy = h5[2] / h5[4], cz = h5[3] / h5[4];
            double slope = 0.0;
            if (it > 0 && std::fabs(dH - prev_dH) > 1.0e-15 && std::fabs(prev_step) > 1.0e-15) {
                slope = (dH - prev_dH) / prev_step;
            } else {
#if TRANSPORT_HAS_CUDA
                mf_detail::accum_virial_kernel<<<blocks, TPB>>>(pv, cx, cy, cz, cm_accum_.data() + 5);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(&slope, cm_accum_.data() + 5, sizeof(double), cudaMemcpyDeviceToHost));
#else
                for (int i = 0; i < np; ++i) mf_detail::accum_peculiar_virial_particle(pv, i, cx, cy, cz, &slope);
#endif
                slope *= 1.35;  // Account for m*/m ~ 0.70 on first step before exact secant kicks in
            }
            if (std::fabs(slope) <= 1.0e-9) return;
            double dlam = -dH / slope;
            if (dlam < -0.02) dlam = -0.02;
            if (dlam > 0.02) dlam = 0.02;
            prev_dH = dH;
            prev_step = dlam;
            const double lam = 1.0 + dlam;
#if TRANSPORT_HAS_CUDA
            mf_detail::scale_peculiar_kernel<<<blocks, TPB>>>(pv, cx, cy, cz, lam);
            CUDA_CHECK(cudaGetLastError());
#else
            for (int i = 0; i < np; ++i) mf_detail::scale_peculiar_momentum_particle(pv, i, cx, cy, cz, lam);
#endif
        }
    }

    // Relative spatial baryon density fluctuation sigma(rho_B) / <rho_B> (tracks spinodal clumping!)
    double density_contrast_rms() const {
        if (!cfg_.enabled) return 0.0;
        const auto hrp = rho_p_.to_vector();
        const auto hrn = rho_n_.to_vector();
        if (hrp.empty()) return 0.0;
        double mean = 0.0, mean_sq = 0.0;
        for (std::size_t c = 0; c < hrp.size(); ++c) {
            const double rb = hrp[c] + hrn[c];
            mean += rb;
            mean_sq += rb * rb;
        }
        mean /= static_cast<double>(hrp.size());
        mean_sq /= static_cast<double>(hrp.size());
        if (mean <= 1.0e-12) return 0.0;
        const double var = (mean_sq > mean * mean) ? (mean_sq - mean * mean) : 0.0;
        return std::sqrt(var) / mean;
    }

    MeanFieldView view() {
        return MeanFieldView{
            cfg_, gcfg_,
            rho_p_.data(), rho_n_.data(),
            px_sum_.data(), py_sum_.data(), pz_sum_.data(),
            psi_md_.data(), u_loc_p_.data(), u_loc_n_.data(),
            u_md_dual_.data(), rho_q_.data(), phi_coul_.data(), tmp_buf_.data()
        };
    }

    std::vector<Real> phi_coul_raw_host() const { return phi_coul_raw_.to_vector(); }
    std::vector<Real> phi_coul_host() const { return phi_coul_.to_vector(); }
    const MeanFieldConfig& config() const { return cfg_; }

private:
    void smooth_field_3d(DeviceBuffer<Real>& buf) {
        const int nc = gcfg_.num_cells();
        for (int axis = 0; axis < 3; ++axis) {
#if TRANSPORT_HAS_CUDA
            constexpr int TPB = 256;
            mf_detail::smooth_axis_kernel<<<(nc + TPB - 1) / TPB, TPB>>>(
                buf.data(), tmp_buf_.data(), nc, axis, gcfg_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(buf.data(), tmp_buf_.data(), nc * sizeof(Real), cudaMemcpyDeviceToDevice));
#else
            for (int c = 0; c < nc; ++c) {
                mf_detail::smooth_axis_cell(buf.data(), tmp_buf_.data(), c, axis, gcfg_);
            }
            std::memcpy(buf.data(), tmp_buf_.data(), nc * sizeof(Real));
#endif
        }
    }

    MeanFieldConfig cfg_;
    GridConfig gcfg_;
    DeviceBuffer<Real> rho_p_, rho_n_;
    DeviceBuffer<Real> px_sum_, py_sum_, pz_sum_;
    DeviceBuffer<Real> psi_md_, u_loc_p_, u_loc_n_, u_md_dual_;
    DeviceBuffer<Real> rho_q_, phi_coul_raw_, phi_coul_, tmp_buf_;
    DeviceBuffer<double> cm_accum_;
    CoulombPoissonSolver poisson_solver_;
};

}  // namespace transport
