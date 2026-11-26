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
#include "sim/particles.cuh"

namespace transport {

// Self-describing binary 3D hydro/density grid header ("CHYD", 88 bytes)
// Followed immediately by 8 contiguous float32 3D arrays of length nx * ny * nz
// (indexed as c = ix + nx * (iy + ny * iz)):
//   [0] rho_B_lab    : Computational-frame baryon density J^0 [fm^-3]
//   [1] rho_B_lrf    : Comoving proper baryon density sqrt((J^0)^2 - |J|^2) [fm^-3]
//   [2] rho_Q_lab    : Computational-frame electric charge density [fm^-3]
//   [3] e_tot_lab    : Total energy density T^00 = T^00_kin + e_pot [GeV / fm^3]
//   [4] eps_lrf      : Local rest-frame energy density [GeV / fm^3]
//   [5] vx           : Local collective velocity v_x = T^0x / T^00_kin [c]
//   [6] vy           : Local collective velocity v_y = T^0y / T^00_kin [c]
//   [7] vz           : Local collective velocity v_z = T^0z / T^00_kin [c]
#pragma pack(push, 1)
struct HydroGridHeader {
    std::uint32_t magic = 0x44594843U;  // "CHYD"
    std::int32_t version = 1;
    std::int32_t step = 0;
    std::int32_t nx = 0;
    std::int32_t ny = 0;
    std::int32_t nz = 0;
    std::int32_t n_fields = 8;
    std::int32_t reserved = 0;
    double t_fm = 0.0;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
};
#pragma pack(pop)

namespace hydro_detail {

inline void smooth_host_field_3d(std::vector<double>& buf, std::vector<double>& tmp,
                                 const GridConfig& g) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const int nc = g.num_cells();
    for (int axis = 0; axis < 3; ++axis) {
        for (int c = 0; c < nc; ++c) {
            mf_detail::smooth_axis_cell(buf.data(), tmp.data(), c, axis, g);
        }
        buf.swap(tmp);
    }
    (void)nx; (void)ny; (void)nz;
}

}  // namespace hydro_detail

inline void write_hydro_grid_binary(const std::string& hydro_dir,
                                    int hydro_idx,
                                    int step,
                                    double t_fm,
                                    int n_test,
                                    const GridConfig& gcfg,
                                    const MeanFieldConfig& mf_cfg,
                                    const std::vector<HostParticle>& host_p) {
    std::filesystem::create_directories(hydro_dir);
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/hydro_grid_%04d.bin", hydro_dir.c_str(), hydro_idx);

    const int nc = gcfg.num_cells();
    if (nc <= 0) return;
    const double inv_n_dV = 1.0 / (static_cast<double>(n_test) * gcfg.cell_volume_fm3());

    std::vector<double> rp(nc, 0.0), rn(nc, 0.0), rq(nc, 0.0);
    std::vector<double> jbx(nc, 0.0), jby(nc, 0.0), jbz(nc, 0.0);
    std::vector<double> t00(nc, 0.0), t0x(nc, 0.0), t0y(nc, 0.0), t0z(nc, 0.0);
    std::vector<double> tmp(nc, 0.0);

    for (const auto& hp : host_p) {
        int i0, j0, k0;
        double fx, fy, fz;
        if (!mf_detail::cic_weights(hp.x, hp.y, hp.z, gcfg, i0, j0, k0, fx, fy, fz)) continue;

        const SpeciesInfo sp = get_species_info(hp.species);
        const double A = static_cast<double>(sp.baryon_num);
        const double Q = static_cast<double>(sp.charge);
        double wp = 0.0, wn = 0.0;
        get_effective_pn_weights(hp.species, wp, wn);
        const double inv_E = 1.0 / hp.E;

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
                    rp[c] += w * wp;
                    rn[c] += w * wn;
                    rq[c] += w * Q;
                    jbx[c] += w * A * (hp.px * inv_E);
                    jby[c] += w * A * (hp.py * inv_E);
                    jbz[c] += w * A * (hp.pz * inv_E);
                    t00[c] += w * hp.E;
                    t0x[c] += w * hp.px;
                    t0y[c] += w * hp.py;
                    t0z[c] += w * hp.pz;
                }
            }
        }
    }

    for (auto* vec : {&rp, &rn, &rq, &jbx, &jby, &jbz, &t00, &t0x, &t0y, &t0z}) {
        hydro_detail::smooth_host_field_3d(*vec, tmp, gcfg);
    }

    std::vector<float> f_rb_lab(nc), f_rb_lrf(nc), f_rq_lab(nc);
    std::vector<float> f_etot(nc), f_elrf(nc), f_vx(nc), f_vy(nc), f_vz(nc);

    for (int c = 0; c < nc; ++c) {
        const double rb = rp[c] + rn[c];
        const double jb2 = jbx[c] * jbx[c] + jby[c] * jby[c] + jbz[c] * jbz[c];
        const double rb_lrf = (rb * rb > jb2) ? std::sqrt(rb * rb - jb2) : 0.0;

        double epot = 0.0;
        if (mf_cfg.enabled && rb > 1.0e-12) {
            const double u = rb / mf_cfg.rho0_fm3;
            const double delta = (rn[c] - rp[c]) / rb;
            epot = 0.5 * mf_cfg.A_sk_gev * mf_cfg.rho0_fm3 * u * u +
                   (mf_cfg.B_sk_gev / (mf_cfg.sigma_exp + 1.0)) * mf_cfg.rho0_fm3 *
                       std::pow(u, mf_cfg.sigma_exp + 1.0) +
                   mf_cfg.S_pot_gev * mf_cfg.rho0_fm3 * std::pow(u, mf_cfg.gamma_sym + 1.0) * delta * delta +
                   0.5 * mf_cfg.C_md_gev * mf_cfg.rho0_fm3 * u * u * 0.80;
        }

        const double e_kin = t00[c];
        const double e_tot = e_kin + epot;
        const double p_fl2 = t0x[c] * t0x[c] + t0y[c] * t0y[c] + t0z[c] * t0z[c];
        const double eps_kin_lrf = (e_kin * e_kin > p_fl2) ? std::sqrt(e_kin * e_kin - p_fl2) : 0.0;
        const double eps_lrf = eps_kin_lrf + epot;

        f_rb_lab[c] = static_cast<float>(rb);
        f_rb_lrf[c] = static_cast<float>(rb_lrf);
        f_rq_lab[c] = static_cast<float>(rq[c]);
        f_etot[c]   = static_cast<float>(e_tot);
        f_elrf[c]   = static_cast<float>(eps_lrf);
        if (e_kin > 1.0e-12) {
            f_vx[c] = static_cast<float>(t0x[c] / e_kin);
            f_vy[c] = static_cast<float>(t0y[c] / e_kin);
            f_vz[c] = static_cast<float>(t0z[c] / e_kin);
        }
    }

    HydroGridHeader hdr{};
    hdr.step = step;
    hdr.nx = gcfg.nx; hdr.ny = gcfg.ny; hdr.nz = gcfg.nz;
    hdr.t_fm = t_fm;
    hdr.xmin = gcfg.xmin; hdr.xmax = gcfg.xmax;
    hdr.ymin = gcfg.ymin; hdr.ymax = gcfg.ymax;
    hdr.zmin = gcfg.zmin; hdr.zmax = gcfg.zmax;

    std::ofstream out(fname, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(HydroGridHeader));
    for (const auto* arr : {&f_rb_lab, &f_rb_lrf, &f_rq_lab, &f_etot, &f_elrf, &f_vx, &f_vy, &f_vz}) {
        out.write(reinterpret_cast<const char*>(arr->data()),
                  static_cast<std::streamsize>(nc * sizeof(float)));
    }
}

}  // namespace transport
