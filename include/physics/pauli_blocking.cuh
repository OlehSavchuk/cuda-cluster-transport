#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"
#include "physics/species.cuh"
#include "sim/particles.cuh"

namespace transport {

// Fast O(N_part) GPU/Host 6D Phase-Space Occupancy Grid f_tau(r, p) for Pauli Blocking (1 - f_3)(1 - f_4).
// Covers [-L_r, +L_r]^3 fm^3 x [-P_max, +P_max]^3 (GeV/c)^3 on a 16^3 x 10^3 lattice (4.096M floats = 16.4 MB
// per nucleon species, fitting inside CPU L3 & GPU L2 cache) with Husimi/Gaussian 3-point smoothing.
struct PauliGridConfig {
    bool enabled = false;
    int nr = 20;                // Spatial bins per axis: dr = 1.8 fm over [-18, +18] fm (UrQMD packet scale!)
    int np = 12;                // Momentum bins per axis: dp = 0.12 GeV/c over [-0.72, +0.72] GeV/c
    double r_half_fm = 18.0;
    double p_half_gev = 0.72;

    HD_INLINE int num_r_cells() const { return nr * nr * nr; }
    HD_INLINE int num_p_cells() const { return np * np * np; }
    HD_INLINE int total_bins() const { return num_r_cells() * num_p_cells(); }
    HD_INLINE double dr() const { return (2.0 * r_half_fm) / nr; }
    HD_INLINE double dp() const { return (2.0 * p_half_gev) / np; }
};

struct PauliPhaseSpaceView {
    PauliGridConfig cfg;
    const float* f_p = nullptr;
    const float* f_n = nullptr;
};

HD_INLINE Real eval_pauli_f(const PauliPhaseSpaceView& pv,
                            int sp_id,
                            Real x, Real y, Real z,
                            Real px, Real py, Real pz) {
    if (!pv.cfg.enabled) return Real(0);
    const float* tbl = (sp_id == PROTON) ? pv.f_p : ((sp_id == NEUTRON) ? pv.f_n : nullptr);
    if (tbl == nullptr) return Real(0);

    const int nr = pv.cfg.nr, np = pv.cfg.np;
    const Real r_half = static_cast<Real>(pv.cfg.r_half_fm);
    const Real p_half = static_cast<Real>(pv.cfg.p_half_gev);
    const Real inv_dr = static_cast<Real>(nr) / (Real(2) * r_half);
    const Real inv_dp = static_cast<Real>(np) / (Real(2) * p_half);
    const Real ix_f = (x + r_half) * inv_dr;
    const Real iy_f = (y + r_half) * inv_dr;
    const Real iz_f = (z + r_half) * inv_dr;
    const Real ipx_f = (px + p_half) * inv_dp;
    const Real ipy_f = (py + p_half) * inv_dp;
    const Real ipz_f = (pz + p_half) * inv_dp;

    if (ix_f < Real(0) || ix_f >= Real(nr) || iy_f < Real(0) || iy_f >= Real(nr) || iz_f < Real(0) || iz_f >= Real(nr) ||
        ipx_f < Real(0) || ipx_f >= Real(np) || ipy_f < Real(0) || ipy_f >= Real(np) || ipz_f < Real(0) || ipz_f >= Real(np)) {
        return Real(0);
    }
    const int ir = (static_cast<int>(iz_f) * nr + static_cast<int>(iy_f)) * nr + static_cast<int>(ix_f);
    const int ip = (static_cast<int>(ipz_f) * np + static_cast<int>(ipy_f)) * np + static_cast<int>(ipx_f);
    const Real f = static_cast<Real>(tbl[ir * (np * np * np) + ip]);
    return (f < Real(0)) ? Real(0) : ((f > Real(0.98)) ? Real(0.98) : f);
}

namespace pauli_detail {

HD_INLINE void deposit_nucleon_6d_cic(float* tbl, const PauliGridConfig& cfg,
                                               float x, float y, float z,
                                               float px, float py, float pz, float weight_norm) {
    const int nr = cfg.nr, np = cfg.np;
    const float inv_dr = static_cast<float>(cfg.nr) / static_cast<float>(2.0 * cfg.r_half_fm);
    const float inv_dp = static_cast<float>(cfg.np) / static_cast<float>(2.0 * cfg.p_half_gev);
    const float r_half = static_cast<float>(cfg.r_half_fm);
    const float p_half = static_cast<float>(cfg.p_half_gev);
    const float rx = (x + r_half) * inv_dr - 0.5f;
    const float ry = (y + r_half) * inv_dr - 0.5f;
    const float rz = (z + r_half) * inv_dr - 0.5f;
    const float qx = (px + p_half) * inv_dp - 0.5f;
    const float qy = (py + p_half) * inv_dp - 0.5f;
    const float qz = (pz + p_half) * inv_dp - 0.5f;

    const int ir0 = static_cast<int>(std::floor(rx)), jr0 = static_cast<int>(std::floor(ry)), kr0 = static_cast<int>(std::floor(rz));
    const int ip0 = static_cast<int>(std::floor(qx)), jp0 = static_cast<int>(std::floor(qy)), kp0 = static_cast<int>(std::floor(qz));
    if (ir0 < -1 || ir0 >= nr || jr0 < -1 || jr0 >= nr || kr0 < -1 || kr0 >= nr ||
        ip0 < -1 || ip0 >= np || jp0 < -1 || jp0 >= np || kp0 < -1 || kp0 >= np) {
        return;
    }
    const float frx[2] = {1.0f - (rx - ir0), rx - ir0};
    const float fry[2] = {1.0f - (ry - jr0), ry - jr0};
    const float frz[2] = {1.0f - (rz - kr0), rz - kr0};
    const float fpx[2] = {1.0f - (qx - ip0), qx - ip0};
    const float fpy[2] = {1.0f - (qy - jp0), qy - jp0};
    const float fpz[2] = {1.0f - (qz - kp0), qz - kp0};
    const int np3 = np * np * np;

    for (int dz = 0; dz < 2; ++dz) {
        const int kz = kr0 + dz;
        if (kz < 0 || kz >= nr) continue;
        for (int dy = 0; dy < 2; ++dy) {
            const int jy = jr0 + dy;
            if (jy < 0 || jy >= nr) continue;
            for (int dx = 0; dx < 2; ++dx) {
                const int ix = ir0 + dx;
                if (ix < 0 || ix >= nr) continue;
                const float wr = weight_norm * frx[dx] * fry[dy] * frz[dz];
                float* p_blk = tbl + ((kz * nr + jy) * nr + ix) * np3;
                for (int dpz = 0; dpz < 2; ++dpz) {
                    const int kp = kp0 + dpz;
                    if (kp < 0 || kp >= np) continue;
                    for (int dpy = 0; dpy < 2; ++dpy) {
                        const int jp = jp0 + dpy;
                        if (jp < 0 || jp >= np) continue;
                        const float wr_yz = wr * fpz[dpz] * fpy[dpy];
                        const int base_p = (kp * np + jp) * np;
                        for (int dpx = 0; dpx < 2; ++dpx) {
                            const int ip = ip0 + dpx;
                            if (ip >= 0 && ip < np) portable_atomic_add_float(&p_blk[base_p + ip], wr_yz * fpx[dpx]);
                        }
                    }
                }
            }
        }
    }
}

#if TRANSPORT_HAS_CUDA
template <int = 0>
__global__ void deposit_pauli_6d_kernel(ParticleSoAView p, int n_slots, PauliGridConfig cfg,
                                        float norm, float* d_fp, float* d_fn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slots || !p.active[i] || p.freezeout_tag[i] != 0) return;
    const int id = p.species[i];
    if (id == PROTON) {
        deposit_nucleon_6d_cic(d_fp, cfg, static_cast<float>(p.x[i]), static_cast<float>(p.y[i]),
                               static_cast<float>(p.z[i]), static_cast<float>(p.px[i]),
                               static_cast<float>(p.py[i]), static_cast<float>(p.pz[i]), norm);
    } else if (id == NEUTRON) {
        deposit_nucleon_6d_cic(d_fn, cfg, static_cast<float>(p.x[i]), static_cast<float>(p.y[i]),
                               static_cast<float>(p.z[i]), static_cast<float>(p.px[i]),
                               static_cast<float>(p.py[i]), static_cast<float>(p.pz[i]), norm);
    }
}
#endif

}  // namespace pauli_detail

class PauliPhaseSpaceGrid {
public:
    explicit PauliPhaseSpaceGrid(const PauliGridConfig& cfg = PauliGridConfig{}) : cfg_(cfg) {
        if (cfg_.enabled) {
            d_fp_.resize(cfg_.total_bins());
            d_fn_.resize(cfg_.total_bins());
            d_fp_.zero();
            d_fn_.zero();
        }
    }

    void configure(const PauliGridConfig& cfg) {
        cfg_ = cfg;
        if (cfg_.enabled && static_cast<int>(d_fp_.size()) != cfg_.total_bins()) {
            d_fp_.resize(cfg_.total_bins());
            d_fn_.resize(cfg_.total_bins());
            d_fp_.zero();
            d_fn_.zero();
        }
    }

    void rebuild(ParticleSystem& particles, int n_test) {
        if (!cfg_.enabled) return;
        const int n_slots = particles.slot_size();
        const double two_pi_hc = phys::TWO_PI * phys::HBARC;
        const double h3 = two_pi_hc * two_pi_hc * two_pi_hc;
        const double d3r = cfg_.dr() * cfg_.dr() * cfg_.dr();
        const double d3p = cfg_.dp() * cfg_.dp() * cfg_.dp();
        const float norm = static_cast<float>(h3 / (2.0 * std::max(1, n_test) * d3r * d3p));

#if TRANSPORT_HAS_CUDA
        d_fp_.zero();
        d_fn_.zero();
        if (n_slots <= 0) return;
        constexpr int TPB = 256;
        pauli_detail::deposit_pauli_6d_kernel<<<(n_slots + TPB - 1) / TPB, TPB>>>(
            particles.view(), n_slots, cfg_, norm, d_fp_.data(), d_fn_.data());
        CUDA_CHECK(cudaGetLastError());
#else
        const int np3 = cfg_.num_p_cells();
        float* fp = d_fp_.data();
        float* fn = d_fn_.data();
        for (int blk : touched_blocks_) {
            std::fill_n(fp + blk * np3, np3, 0.0f);
            std::fill_n(fn + blk * np3, np3, 0.0f);
        }
        touched_blocks_.clear();
        if (n_slots <= 0) return;
        if (static_cast<int>(blk_flag_.size()) != cfg_.num_r_cells()) blk_flag_.assign(cfg_.num_r_cells(), 0);

        const ParticleSoAView p = particles.view();
        const int nr = cfg_.nr;
        for (int i = 0; i < n_slots; ++i) {
            if (!p.active[i] || p.freezeout_tag[i] != 0) continue;
            const int id = p.species[i];
            if (id != PROTON && id != NEUTRON) continue;
            const int ir0 = static_cast<int>(std::floor((p.x[i] + cfg_.r_half_fm) / cfg_.dr() - 0.5));
            const int jr0 = static_cast<int>(std::floor((p.y[i] + cfg_.r_half_fm) / cfg_.dr() - 0.5));
            const int kr0 = static_cast<int>(std::floor((p.z[i] + cfg_.r_half_fm) / cfg_.dr() - 0.5));
            if (ir0 < -1 || ir0 >= nr || jr0 < -1 || jr0 >= nr || kr0 < -1 || kr0 >= nr) continue;
            for (int dz = 0; dz < 2; ++dz) {
                const int kz = kr0 + dz; if (kz < 0 || kz >= nr) continue;
                for (int dy = 0; dy < 2; ++dy) {
                    const int jy = jr0 + dy; if (jy < 0 || jy >= nr) continue;
                    for (int dx = 0; dx < 2; ++dx) {
                        const int ix = ir0 + dx; if (ix < 0 || ix >= nr) continue;
                        const int b = (kz * nr + jy) * nr + ix;
                        if (!blk_flag_[b]) { blk_flag_[b] = 1; touched_blocks_.push_back(b); }
                    }
                }
            }
            pauli_detail::deposit_nucleon_6d_cic(
                (id == PROTON) ? fp : fn, cfg_,
                static_cast<float>(p.x[i]), static_cast<float>(p.y[i]), static_cast<float>(p.z[i]),
                static_cast<float>(p.px[i]), static_cast<float>(p.py[i]), static_cast<float>(p.pz[i]), norm);
        }
        for (int b : touched_blocks_) blk_flag_[b] = 0;
#endif
    }

    PauliPhaseSpaceView view() const { return PauliPhaseSpaceView{cfg_, d_fp_.data(), d_fn_.data()}; }
    std::vector<float> download_fp() const { return d_fp_.to_vector(); }
    std::vector<float> download_fn() const { return d_fn_.to_vector(); }
    const PauliGridConfig& config() const { return cfg_; }

private:
    PauliGridConfig cfg_;
    DeviceBuffer<float> d_fp_, d_fn_;
    std::vector<int> touched_blocks_;
    std::vector<std::uint8_t> blk_flag_;
};

}  // namespace transport
