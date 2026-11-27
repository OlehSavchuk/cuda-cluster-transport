#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/mean_field.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/hcgf_io.cuh"
#include "sim/init_conditions.cuh"
#include "sim/particles.cuh"

namespace transport {

enum class ClusterMode : int {
    Microscopic = 0,       // Full dynamical 2<->1 & 2<->2 cluster reaction network
    None = 1,              // Pure N + Delta(1232) + pi transport (exports T^munu / hypersurface for HCGF)
    FreezeoutThermal = 2   // Pure N + Delta + pi in dense phase + thermal clusters in cells reaching n_B < n_fr
};

enum class BackflowMode : int {
    RemeltInDense = 0,     // Re-absorb inward nucleons & Mott-dissociate clusters re-entering n_B > n_melt
    OutwardOnly = 1,       // Cooper-Frye outward cut (dn_B/dt + v.grad(n_B) <= 0) + dense re-melting
    GhostFreeStream = 2    // Diagnostic: no backflow filtering or re-melting
};

enum class FreezeoutSampling : int {
    MicrocanonicalParcel = 0,  // Constrained parcel sampling conserving exact integer (B, Q) & 4-momentum P^mu
    GrandCanonical = 1         // Unconstrained Cooper-Frye Poisson(V_c * n_eq,i) per cell (standard HCGF!)
};

struct FreezeoutConfig {
    ClusterMode mode = ClusterMode::Microscopic;
    BackflowMode backflow = BackflowMode::RemeltInDense;
    FreezeoutSampling sampling = FreezeoutSampling::MicrocanonicalParcel;
    int oversample_factor = 1;        // Freeze-out Poisson oversampling factor M >= 1 (backflow resamples to N_test!)
    double n_fr_fm3 = 0.040;          // Freeze-out Eckart baryon density threshold (~rho_0 / 4) [fm^-3]
    double melt_factor = 1.15;        // Hysteresis ratio n_melt / n_fr for re-entering dense matter
    double min_time_fm = 0.0;         // Minimum time before expansion freeze-out check activates [fm/c]
    double peak_track_time_fm = 0.0;  // Time from which compressed peak_nb >= n_fr is tracked [fm/c]
    bool require_peak_crossing = true;// Require cell to have compressed >= n_fr and cross downward (dn_B/dt <= 0)
    bool quantum_statistics = true;   // Include Fermi-Dirac (fermions) & Bose-Einstein (bosons) in INCG EoS
    int max_cluster_A = 6;            // Max nuclear mass number A included in EoS & sampling (up to 6!)
    bool include_excited_clusters = true;  // Include d* and A=4,5,6 statistical resonances (feed-down!)
    bool local_anisotropy = true;     // Pratt-Torrieri p*_i = Lambda_ij q_j (det Lambda = 1) matching P*_ij!
    std::string cells_csv_path;       // Optional CSV log of every freeze-out cell (t, x, y, z, T, muB, muQ, u^mu)
};

struct FreezeoutCellRecord {
    int step = 0;
    int cell_id = 0;
    double t_fm = 0.0;
    double x_fm = 0.0, y_fm = 0.0, z_fm = 0.0;
    double nB_eckart_fm3 = 0.0;
    double e_rest_mev_per_b = 0.0;
    double T_mev = 0.0;
    double muB_mev = 0.0;
    double muQ_mev = 0.0;
    double ux = 0.0, uy = 0.0, uz = 0.0;
    double aniso_stretch = 1.0;       // alpha_max / alpha_min of Pratt-Torrieri Lambda
    double shear_xy = 0.0;            // (P*_xx - P*_yy) / (P*_xx + P*_yy) in Eckart rest frame
    int baryons_converted = 0;
    int clusters_formed = 0;
    int back_emitted = 0;
};

namespace fo_detail {

// Exact 28-point Fermi-Dirac moments for nucleons (p, n) valid from T = 1.5 MeV degenerate sea to Boltzmann
inline void nucleon_fermi_dirac_moments(double m, int g, double T_gev, double u_chem,
                                        double& n_fm3, double& dn_du, double& e_kin_gev_fm3) {
    using namespace phys;
    constexpr int N_Q = 28;
    const double s_max = std::sqrt(std::max(18.0, u_chem + 16.0));
    const double ds = s_max / N_Q;
    const double pref = (static_cast<double>(g) * T_gev) / (2.0 * PI * PI * HBARC * HBARC * HBARC);
    n_fm3 = 0.0; dn_du = 0.0; e_kin_gev_fm3 = 0.0;
    for (int j = 0; j < N_Q; ++j) {
        const double s = (j + 0.5) * ds;
        const double u = s * s;
        const double K = T_gev * u;
        const double E = m + K;
        const double p = std::sqrt(K * (K + 2.0 * m));
        const double arg = std::clamp(u - u_chem, -80.0, 80.0);
        const double f = 1.0 / (std::exp(arg) + 1.0);
        const double w = pref * (2.0 * s * ds) * p * E;
        n_fm3 += w * f;
        dn_du += w * f * (1.0 - f);
        e_kin_gev_fm3 += w * f * K;
    }
}

// Pure thermal Maxwell-Juttner base density phi_0(T) [fm^-3] (WITHOUT exp(-bind_shift/T)) and <K>(T) [GeV]
inline void species_thermal_base(int sp_id, double T_gev, double& phi0_fm3, double& k_mean_gev, double& bind_shift_gev) {
    using namespace phys;
    const SpeciesInfo sp = get_species_info(sp_id);
    const double m = sp.pole_mass_gev;
    bind_shift_gev = m - (sp.charge * MASS_PROTON + (sp.baryon_num - sp.charge) * MASS_NEUTRON);
    constexpr int N_Q = 40;
    constexpr double S_MAX = 6.2;
    const double ds = S_MAX / N_Q;
    double i_n = 0.0, i_k = 0.0;
    for (int j = 0; j < N_Q; ++j) {
        const double s = (j + 0.5) * ds;
        const double u = s * s;
        const double K = T_gev * u;
        const double E = m + K;
        const double p = std::sqrt(K * (K + 2.0 * m));
        const double w = (2.0 * s) * p * E * std::exp(-u) * ds;
        i_n += w;
        i_k += w * K;
    }
    const double pref = static_cast<double>(sp.degeneracy) / (2.0 * PI * PI * HBARC * HBARC * HBARC);
    phi0_fm3 = pref * T_gev * i_n;
    k_mean_gev = (i_n > 0.0) ? (i_k / i_n) : (1.5 * T_gev);
}

// Precomputed 1024-point thermal base LUT over T in [1.4, 166.0] MeV (dT = 0.16 MeV -> <1e-6 interp accuracy!)
struct ThermalBaseLut {
    static constexpr int N_T = 1024;
    static constexpr double T_MIN = 0.0014;
    static constexpr double T_MAX = 0.1660;
    double phi0[NUM_SPECIES][N_T] = {};
    double kmean[NUM_SPECIES][N_T] = {};
    double bshift[NUM_SPECIES] = {};

    ThermalBaseLut() {
        const double dT = (T_MAX - T_MIN) / (N_T - 1);
        for (int id = 0; id < NUM_SPECIES; ++id) {
            const SpeciesInfo sp = get_species_info(id);
            if (is_pion(id) || is_delta(id) || sp.baryon_num <= 0) continue;
            for (int k = 0; k < N_T; ++k) {
                species_thermal_base(id, T_MIN + k * dT, phi0[id][k], kmean[id][k], bshift[id]);
            }
        }
    }

    inline void eval(int id, double T_gev, double& out_phi0, double& out_kmean, double& out_bshift) const {
        out_bshift = bshift[id];
        if (T_gev <= T_MIN || T_gev >= T_MAX) {
            species_thermal_base(id, T_gev, out_phi0, out_kmean, out_bshift);
            return;
        }
        const double x = (T_gev - T_MIN) * ((N_T - 1) / (T_MAX - T_MIN));
        const int idx = std::min(N_T - 2, std::max(0, static_cast<int>(x)));
        const double f = x - idx;
        out_phi0  = phi0[id][idx]  + f * (phi0[id][idx + 1]  - phi0[id][idx]);
        out_kmean = kmean[id][idx] + f * (kmean[id][idx + 1] - kmean[id][idx]);
    }
};

inline const ThermalBaseLut& get_thermal_base_lut() {
    static const ThermalBaseLut lut{};
    return lut;
}

// Given (T_gev, nB_fm3, Yp), solve 2x2 SPD Newton system for (u_p, u_n) = ((mu_p-m_p)/T, (mu_n-m_n)/T)
// with Fermi-Dirac (fermions) & Bose-Einstein (bosons) quantum statistics in the INCG EoS!
inline double solve_hrg_fugacities(double T_gev, double nB_fm3, double Yp, bool inc_exc, int max_A,
                                   double n_out[NUM_SPECIES], double& out_up, double& out_un,
                                   bool quantum_stats = true, bool warm_start = false) {
    const double np_tgt = std::max(1.0e-9, Yp * nB_fm3);
    const double nn_tgt = std::max(1.0e-9, (1.0 - Yp) * nB_fm3);
    const ThermalBaseLut& lut = get_thermal_base_lut();

    double phi0[NUM_SPECIES] = {}, kmean[NUM_SPECIES] = {}, bshift[NUM_SPECIES] = {};
    int active_ids[NUM_SPECIES];
    int n_act_sp = 0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (is_pion(id) || is_delta(id) || sp.baryon_num <= 0 || sp.baryon_num > max_A) continue;
        if (sp.is_excited && !inc_exc) continue;
        lut.eval(id, T_gev, phi0[id], kmean[id], bshift[id]);
        if (phi0[id] > 0.0) active_ids[n_act_sp++] = id;
    }

    double up = warm_start ? std::clamp(out_up, -150.0, 50.0) : std::log(np_tgt / std::max(phi0[PROTON], 1.0e-30));
    double un = warm_start ? std::clamp(out_un, -150.0, 50.0) : std::log(nn_tgt / std::max(phi0[NEUTRON], 1.0e-30));
    const int n_qterms = quantum_stats ? 6 : 1;
    static constexpr double INV_Q15[7] = {0.0, 1.0, 0.35355339059327379, 0.19245008972987526,
                                          0.125, 0.089442719099991588, 0.068041381743977170};
    static constexpr double INV_Q05[7] = {0.0, 1.0, 0.70710678118654752, 0.57735026918962576,
                                          0.5, 0.44721359549995794, 0.40824829046386302};
    static constexpr double INV_Q25[7] = {0.0, 1.0, 0.17677669529663689, 0.064150029909958421,
                                          0.03125, 0.017888543819998318, 0.011340230290662862};
    const double inv_T = 1.0 / T_gev;

    auto eval_species = [&](int id, double u_p, double u_n, double& na, double& dna_deta, double& e_tot_a) {
        const SpeciesInfo sp = get_species_info(id);
        if (quantum_stats && (id == PROTON || id == NEUTRON)) {
            double ek = 0.0;
            nucleon_fermi_dirac_moments(sp.pole_mass_gev, sp.degeneracy, T_gev,
                                        (id == PROTON) ? u_p : u_n, na, dna_deta, ek);
            e_tot_a = ek;
            return;
        }
        const double Z = sp.charge, N = sp.baryon_num - sp.charge;
        const bool is_boson = (sp.baryon_num % 2 == 0);
        double eta = Z * u_p + N * u_n - bshift[id] * inv_T;
        eta = std::clamp(eta, -200.0, is_boson ? -0.012 : 60.0);
        const double z1 = std::exp(eta);
        if (n_qterms == 1 || z1 < 1.0e-5) {
            na = phi0[id] * z1;
            dna_deta = na;
            e_tot_a = na * (bshift[id] + kmean[id]);
            return;
        }
        double s_n = 0.0, s_dn = 0.0, s_k = 0.0, zq = z1;
        for (int q = 1; q <= n_qterms; ++q) {
            const double ez = ((!is_boson && (q % 2 == 0)) ? -1.0 : 1.0) * zq;
            s_n  += ez * INV_Q15[q];
            s_dn += ez * INV_Q05[q];
            s_k  += ez * INV_Q25[q];
            zq *= z1;
        }
        na = std::max(0.0, phi0[id] * s_n);
        dna_deta = std::max(0.0, phi0[id] * s_dn);
        e_tot_a = na * bshift[id] + phi0[id] * kmean[id] * s_k;
    };

    for (int it = 0; it < 20; ++it) {
        double fp = -np_tgt, fn = -nn_tgt, jpp = 1.0e-30, jpn = 0.0, jnn = 1.0e-30;
        for (int k = 0; k < n_act_sp; ++k) {
            const int id = active_ids[k];
            const SpeciesInfo sp = get_species_info(id);
            const double Z = sp.charge, N = sp.baryon_num - sp.charge;
            double na = 0.0, dnd = 0.0, ea = 0.0;
            eval_species(id, up, un, na, dnd, ea);
            fp += Z * na; fn += N * na;
            jpp += Z * Z * dnd; jpn += Z * N * dnd; jnn += N * N * dnd;
        }
        if (std::fabs(fp) + std::fabs(fn) < 1.0e-11 * nB_fm3) break;
        const double det = jpp * jnn - jpn * jpn;
        if (det <= 1.0e-60) break;
        up -= std::clamp(( jnn * fp - jpn * fn) / det, -1.8, 1.8);
        un -= std::clamp((-jpn * fp + jpp * fn) / det, -1.8, 1.8);
    }
    out_up = up; out_un = un;
    for (int id = 0; id < NUM_SPECIES; ++id) n_out[id] = 0.0;
    double e_sum = 0.0;
    for (int k = 0; k < n_act_sp; ++k) {
        const int id = active_ids[k];
        double dnd = 0.0, ea = 0.0;
        eval_species(id, up, un, n_out[id], dnd, ea);
        e_sum += ea;
    }
    return e_sum / std::max(nB_fm3, 1.0e-12);
}

}  // namespace fo_detail

namespace fo_detail {

inline void invert_local_cluster_eos(double nB_fm3, double Yp, double e_th_gev, bool inc_exc, int max_A,
                                     double& T_gev, double& muB_gev, double& muQ_gev,
                                     double n_eq[NUM_SPECIES], bool quantum_stats = true) {
    double T_lo = 0.0015, T_hi = 0.165;
    double up = 0.0, un = 0.0;
    double f_lo = solve_hrg_fugacities(T_lo, nB_fm3, Yp, inc_exc, max_A, n_eq, up, un, quantum_stats, false) - e_th_gev;
    if (f_lo >= 0.0) {
        T_gev = T_lo;
    } else {
        double up_hi = up, un_hi = un;
        double f_hi = solve_hrg_fugacities(T_hi, nB_fm3, Yp, inc_exc, max_A, n_eq, up_hi, un_hi, quantum_stats, false) - e_th_gev;
        if (f_hi <= 0.0) {
            T_gev = T_hi;
            up = up_hi; un = un_hi;
        } else {
            // Illinois-damped Secant + periodic midpoint bisection on strictly monotonic e_hrg(T) (~6-8 evals!)
            double T_cur = std::clamp(0.67 * std::max(0.003, e_th_gev + 0.012), T_lo + 1.0e-4, T_hi - 1.0e-4);
            int prev_side = 0;
            for (int b = 0; b < 24; ++b) {
                if (T_hi - T_lo < 5.0e-7) break;
                const double f_cur = solve_hrg_fugacities(
                    T_cur, nB_fm3, Yp, inc_exc, max_A, n_eq, up, un, quantum_stats, false) - e_th_gev;
                if (std::fabs(f_cur) < 5.0e-8) {
                    T_lo = T_hi = T_cur;
                    break;
                }
                if (f_cur < 0.0) {
                    T_lo = T_cur; f_lo = f_cur;
                    if (prev_side == -1) f_hi *= 0.5;
                    prev_side = -1;
                } else {
                    T_hi = T_cur; f_hi = f_cur;
                    if (prev_side == +1) f_lo *= 0.5;
                    prev_side = +1;
                }
                if (b % 4 == 3) {
                    T_cur = 0.5 * (T_lo + T_hi);
                } else {
                    const double pad = 0.05 * (T_hi - T_lo);
                    const double denom = f_hi - f_lo;
                    const double T_sec = (std::fabs(denom) > 1.0e-18)
                                             ? (T_lo - f_lo * (T_hi - T_lo) / denom)
                                             : (0.5 * (T_lo + T_hi));
                    T_cur = (T_sec > T_lo + pad && T_sec < T_hi - pad) ? T_sec : (0.5 * (T_lo + T_hi));
                }
            }
            T_gev = 0.5 * (T_lo + T_hi);
            solve_hrg_fugacities(T_gev, nB_fm3, Yp, inc_exc, max_A, n_eq, up, un, quantum_stats, false);
        }
    }
    muB_gev = phys::MASS_NEUTRON + T_gev * un;
    muQ_gev = (phys::MASS_PROTON + T_gev * up) - muB_gev;
}

inline int sample_poisson(double lam, PhiloxRng& rng) {
    if (lam <= 0.0) return 0;
    if (lam > 25.0) {
        const int k = static_cast<int>(std::llround(lam + std::sqrt(lam) * rng.normal()));
        return (k > 0) ? k : 0;
    }
    const double L = std::exp(-lam);
    int k = 0;
    double p = 1.0;
    do {
        ++k;
        p *= std::max(1.0e-15, rng.uniform());
    } while (p > L);
    return k - 1;
}

// Conservative K-body Pratt-Torrieri (p*_i = Lambda_ij q_j, det Lambda = 1) Maxwell-Juttner sampler
// matching both the local 3x3 Eckart stress anisotropy and exact cell 4-momentum (E_tot, P_tot)
inline bool sample_conservative_kbody(const std::vector<int>& sp_ids,
                                      const std::vector<double>& masses,
                                      double T_gev,
                                      double E_tot, double Px, double Py, double Pz,
                                      bool check_outward,
                                      double dt_nb, double gnx, double gny, double gnz,
                                      PhiloxRng& rng,
                                      std::vector<FourVector>& out_p4,
                                      int& out_inward_cnt,
                                      const double* Lambda_opt = nullptr) {
    static constexpr double I3[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double* L = (Lambda_opt != nullptr) ? Lambda_opt : I3;
    const int K = static_cast<int>(sp_ids.size());
    const double P2 = Px * Px + Py * Py + Pz * Pz;
    if (K < 2 || E_tot * E_tot <= P2) return false;
    const double Mc = std::sqrt(E_tot * E_tot - P2);
    double m_sum = 0.0;
    for (double m : masses) m_sum += m;
    if (m_sum >= Mc - 1.0e-5) return false;

    const double bx = Px / E_tot, by = Py / E_tot, bz = Pz / E_tot;
    const double gam = E_tot / Mc;
    std::vector<double> qx(K), qy(K), qz(K);
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int k = 0; k < K; ++k) {
        double dx = 0.0, dy = 0.0, dz = 0.0;
        for (int tr = 0; tr < (check_outward ? 16 : 1); ++tr) {
            const FourVector q4 = sample_maxwell_juttner(masses[k], T_gev, rng);
            dx = L[0] * q4.x + L[1] * q4.y + L[2] * q4.z;
            dy = L[3] * q4.x + L[4] * q4.y + L[5] * q4.z;
            dz = L[6] * q4.x + L[7] * q4.y + L[8] * q4.z;
            if (!check_outward) break;
            const double ed = std::sqrt(masses[k] * masses[k] + dx * dx + dy * dy + dz * dz);
            const FourVector pl = lorentz_boost_from_cm(masses[k], ed, dx, dy, dz, bx, by, bz, gam);
            const double dnb_dt = dt_nb + (pl.x * gnx + pl.y * gny + pl.z * gnz) / pl.t;
            if (dnb_dt <= 0.0) break;
        }
        qx[k] = dx; qy[k] = dy; qz[k] = dz;
        sx += qx[k]; sy += qy[k]; sz += qz[k];
    }
    double ke_unit = 0.0;
    for (int k = 0; k < K; ++k) {
        const double wf = masses[k] / m_sum;
        qx[k] -= wf * sx; qy[k] -= wf * sy; qz[k] -= wf * sz;
        ke_unit += (qx[k] * qx[k] + qy[k] * qy[k] + qz[k] * qz[k]) / (2.0 * masses[k]);
    }
    double eta = (ke_unit > 1.0e-18) ? std::sqrt(std::max(1.0e-8, (Mc - m_sum) / ke_unit)) : 1.0;
    for (int it = 0; it < 18; ++it) {
        double f = -Mc, df = 0.0;
        for (int k = 0; k < K; ++k) {
            const double q2 = qx[k] * qx[k] + qy[k] * qy[k] + qz[k] * qz[k];
            const double ek = std::sqrt(masses[k] * masses[k] + eta * eta * q2);
            f += ek; df += (eta * q2) / ek;
        }
        if (std::fabs(f) < 1.0e-14 || df <= 1.0e-20) break;
        eta = std::max(1.0e-5, eta - f / df);
    }
    out_p4.resize(K);
    out_inward_cnt = 0;
    for (int k = 0; k < K; ++k) {
        const double px_c = eta * qx[k], py_c = eta * qy[k], pz_c = eta * qz[k];
        const double e_c = std::sqrt(masses[k] * masses[k] + px_c * px_c + py_c * py_c + pz_c * pz_c);
        out_p4[k] = lorentz_boost_from_cm(masses[k], e_c, px_c, py_c, pz_c, bx, by, bz, gam);
        if (dt_nb + (out_p4[k].x * gnx + out_p4[k].y * gny + out_p4[k].z * gnz) / out_p4[k].t > 0.0) {
            ++out_inward_cnt;
        }
    }
    return true;
}

}  // namespace fo_detail

class FreezeoutCellClusterizer {
public:
    FreezeoutCellClusterizer() = default;
    explicit FreezeoutCellClusterizer(const FreezeoutConfig& cfg, const GridConfig& gcfg) {
        configure(cfg, gcfg);
    }

    void configure(const FreezeoutConfig& cfg, const GridConfig& gcfg) {
        cfg_ = cfg;
        gcfg_ = gcfg;
        const int nc = std::max(0, gcfg.num_cells());
        peak_nb_.assign(nc, 0.0);
        prev_nb_.assign(nc, 0.0);
        dnb_dt_vec_.assign(nc, 0.0);
        dense_nucleons_.assign(nc, {});
        back_clusters_.assign(nc, {});
        cell_baryons_.assign(nc, {});
        touched_dn_.clear();
        touched_bc_.clear();
        touched_cb_.clear();
        prev_t_fm_ = -1.0;
        records_.clear();
        total_baryons_converted_ = 0;
        total_clusters_formed_ = 0;
        reabsorbed_nucleons_ = 0;
        remelted_clusters_ = 0;
        back_emitted_total_ = 0;
        thinned_oversamples_ = 0;
    }

    // Track local Eckart baryon density n_B(c) across grid cells; handle back-emitted particles
    // re-entering n_B > n_melt, and convert nucleons in cells reaching n_B < n_fr using E_rest = u^0 E - P.u!
    HcgfGridSnapshot process_step(ParticleSystem& particles,
                                  int step,
                                  double t_fm,
                                  int n_test,
                                  const MeanFieldConfig& mf_cfg,
                                  std::uint64_t seed,
                                  bool force_finalize = false,
                                  bool full_tmunu = false) {
        const bool excl_frozen = (cfg_.mode == ClusterMode::FreezeoutThermal);
        const int nc = gcfg_.num_cells();
        if (nc <= 0) return HcgfGridSnapshot{};
        if (cfg_.mode == ClusterMode::Microscopic) {
            return gpu_ws_.compute(particles, n_test, gcfg_, mf_cfg, excl_frozen, /*full_24_fields=*/true);
        }

        const HcgfGridSnapshot& snap = gpu_ws_.compute(particles, n_test, gcfg_, mf_cfg, excl_frozen, full_tmunu);

        const double dt_snap = (prev_t_fm_ >= 0.0 && t_fm > prev_t_fm_) ? (t_fm - prev_t_fm_) : 1.0;
        const double n_melt = cfg_.melt_factor * cfg_.n_fr_fm3;
        bool any_candidate_cell = force_finalize;
        for (int c = 0; c < nc; ++c) {
            const double nb = snap.fields[20][c];
            if (t_fm >= cfg_.peak_track_time_fm && nb > peak_nb_[c]) peak_nb_[c] = nb;
            const double dnb = (prev_t_fm_ >= 0.0) ? ((nb - prev_nb_[c]) / dt_snap) : 0.0;
            dnb_dt_vec_[c] = dnb;
            prev_nb_[c] = nb;
            if (!any_candidate_cell && t_fm >= cfg_.min_time_fm) {
                const bool post_sep = (t_fm >= 1.35 * cfg_.min_time_fm) && (nb < cfg_.n_fr_fm3) &&
                                      (dnb <= 0.0) && (snap.fields[10][c] > 0.0f);
                const bool crossed = ((peak_nb_[c] >= cfg_.n_fr_fm3) && (nb < cfg_.n_fr_fm3) && (dnb <= 0.0)) || post_sep;
                const bool fo_trig = cfg_.require_peak_crossing ? crossed : (nb < cfg_.n_fr_fm3 && snap.fields[10][c] > 0.0f);
                const bool melt_trig = snap.has_frozen_particles &&
                                       (cfg_.backflow != BackflowMode::GhostFreeStream) && (nb > n_melt);
                if (fo_trig || melt_trig) any_candidate_cell = true;
            }
        }
        prev_t_fm_ = t_fm;
        if (!force_finalize && (t_fm < cfg_.min_time_fm || !any_candidate_cell)) return snap;

        auto host_p = particles.download();
        bool mutated = false;
        std::vector<std::uint8_t> remove_mask(host_p.size(), 0);
        std::vector<HostParticle> spawned;

        for (int c : touched_dn_) dense_nucleons_[c].clear();
        for (int c : touched_bc_) back_clusters_[c].clear();
        for (int c : touched_cb_) cell_baryons_[c].clear();
        touched_dn_.clear();
        touched_bc_.clear();
        touched_cb_.clear();

        // 1. Handle back-emitted freeze-out particles that flew back into dense matter (n_B > n_melt)!
        if (!force_finalize && snap.has_frozen_particles && cfg_.backflow != BackflowMode::GhostFreeStream) {
            PhiloxRng thin_rng(seed ^ 0x7A11C0DEULL, static_cast<std::uint64_t>(step), 0ULL);
            double species_acc[NUM_SPECIES] = {};
            for (int id = 0; id < NUM_SPECIES; ++id) species_acc[id] = thin_rng.uniform();

            for (std::size_t i = 0; i < host_p.size(); ++i) {
                const SpeciesInfo sp = get_species_info(host_p[i].species);
                if (sp.baryon_num <= 0) continue;
                const int ix = static_cast<int>(std::floor((host_p[i].x - gcfg_.xmin) / gcfg_.dx()));
                const int iy = static_cast<int>(std::floor((host_p[i].y - gcfg_.ymin) / gcfg_.dy()));
                const int iz = static_cast<int>(std::floor((host_p[i].z - gcfg_.zmin) / gcfg_.dz()));
                if (ix < 0 || ix >= gcfg_.nx || iy < 0 || iy >= gcfg_.ny || iz < 0 || iz >= gcfg_.nz) continue;
                const int c = mf_detail::cell_flat(ix, iy, iz, gcfg_);
                if (snap.fields[20][c] <= n_melt) continue;

                // Systematic resampling of oversampled backflow (freezeout_tag = M > 1) back down to
                // the simulation's N_test-per-nucleon scale BEFORE reabsorption / Mott remelting!
                if (host_p[i].freezeout_tag > 1) {
                    species_acc[host_p[i].species] += 1.0 / static_cast<double>(host_p[i].freezeout_tag);
                    if (species_acc[host_p[i].species] >= 1.0) {
                        species_acc[host_p[i].species] -= 1.0;
                        host_p[i].freezeout_tag = 1;  // Promoted to 1 / N_test transport scale!
                    } else {
                        remove_mask[i] = 1;
                        ++thinned_oversamples_;
                        mutated = true;
                        continue;
                    }
                }

                if (host_p[i].freezeout_tag != 0 && sp.baryon_num == 1) {
                    host_p[i].freezeout_tag = 0;
                    ++reabsorbed_nucleons_;
                    mutated = true;
                }
                if (host_p[i].freezeout_tag == 0 && sp.baryon_num == 1 && !sp.is_excited) {
                    if (dense_nucleons_[c].empty()) touched_dn_.push_back(c);
                    dense_nucleons_[c].push_back(i);
                } else if (host_p[i].freezeout_tag != 0 && sp.baryon_num >= 2) {
                    if (back_clusters_[c].empty()) touched_bc_.push_back(c);
                    back_clusters_[c].push_back(i);
                }
            }

            std::sort(touched_bc_.begin(), touched_bc_.end());
            // Mott-dissociate back-emitted clusters (cluster + bath_N -> Z p + N n + bath_N) with exact E, P, B, Q!
            for (int c : touched_bc_) {
                PhiloxRng mrng(seed ^ 0x4E1700ULL, static_cast<std::uint64_t>(step), static_cast<std::uint64_t>(c));
                std::size_t partner_cursor = 0;
                for (std::size_t cl_idx : back_clusters_[c]) {
                    if (remove_mask[cl_idx]) continue;
                    const HostParticle& hcl = host_p[cl_idx];
                    const SpeciesInfo spc = get_species_info(hcl.species);
                    const int Z = spc.charge, N = spc.baryon_num - spc.charge;
                    if (Z < 0 || N < 0) continue;

                    std::vector<int> melt_sp;
                    std::vector<double> melt_m;
                    for (int z = 0; z < Z; ++z) { melt_sp.push_back(PROTON);  melt_m.push_back(phys::MASS_PROTON); }
                    for (int n = 0; n < N; ++n) { melt_sp.push_back(NEUTRON); melt_m.push_back(phys::MASS_NEUTRON); }

                    double Et = hcl.E, Pxt = hcl.px, Pyt = hcl.py, Pzt = hcl.pz;
                    std::size_t used_partner = static_cast<std::size_t>(-1);
                    const std::size_t n_dn = dense_nucleons_[c].size();
                    for (std::size_t att = 0; att < n_dn; ++att) {
                        const std::size_t cand = dense_nucleons_[c][(partner_cursor++) % n_dn];
                        if (remove_mask[cand]) continue;
                        const HostParticle& hb = host_p[cand];
                        const double E2 = Et + hb.E, Px2 = Pxt + hb.px, Py2 = Pyt + hb.py, Pz2 = Pzt + hb.pz;
                        const double M2_sq = E2 * E2 - (Px2 * Px2 + Py2 * Py2 + Pz2 * Pz2);
                        const double m_need = Z * phys::MASS_PROTON + N * phys::MASS_NEUTRON + hb.mass + 1.0e-3;
                        if (M2_sq > m_need * m_need) {
                            Et = E2; Pxt = Px2; Pyt = Py2; Pzt = Pz2;
                            melt_sp.push_back(hb.species); melt_m.push_back(hb.mass);
                            used_partner = cand;
                            break;
                        }
                    }
                    std::vector<FourVector> p_out;
                    int dummy_in = 0;
                    if (fo_detail::sample_conservative_kbody(
                            melt_sp, melt_m, 0.015, Et, Pxt, Pyt, Pzt, false, 0.0, 0.0, 0.0, 0.0, mrng, p_out, dummy_in)) {
                        remove_mask[cl_idx] = 1;
                        const std::size_t n_spawn = (used_partner != static_cast<std::size_t>(-1))
                                                        ? (melt_sp.size() - 1)
                                                        : melt_sp.size();
                        if (used_partner != static_cast<std::size_t>(-1)) {
                            HostParticle& hbp = host_p[used_partner];
                            const FourVector& pb = p_out.back();
                            hbp.E = pb.t; hbp.px = pb.x; hbp.py = pb.y; hbp.pz = pb.z;
                            hbp.collided = 2; hbp.freezeout_tag = 0;
                        }
                        for (std::size_t k = 0; k < n_spawn; ++k) {
                            HostParticle hn = hcl;
                            hn.species = melt_sp[k]; hn.mass = melt_m[k];
                            hn.E = p_out[k].t; hn.px = p_out[k].x; hn.py = p_out[k].y; hn.pz = p_out[k].z;
                            hn.collided = 2; hn.freezeout_tag = 0;
                            spawned.push_back(hn);
                        }
                        ++remelted_clusters_;
                        mutated = true;
                    }
                }
            }
        }

        // 2. Bucket remaining un-frozen baryons (p, n, Delta) by spatial grid cell
        for (std::size_t i = 0; i < host_p.size(); ++i) {
            if (remove_mask[i] || host_p[i].freezeout_tag != 0) continue;
            const SpeciesInfo sp = get_species_info(host_p[i].species);
            if (sp.baryon_num != 1) continue;
            const int ix = static_cast<int>(std::floor((host_p[i].x - gcfg_.xmin) / gcfg_.dx()));
            const int iy = static_cast<int>(std::floor((host_p[i].y - gcfg_.ymin) / gcfg_.dy()));
            const int iz = static_cast<int>(std::floor((host_p[i].z - gcfg_.zmin) / gcfg_.dz()));
            if (ix < 0 || ix >= gcfg_.nx || iy < 0 || iy >= gcfg_.ny || iz < 0 || iz >= gcfg_.nz) continue;
            const int c = mf_detail::cell_flat(ix, iy, iz, gcfg_);
            if (cell_baryons_[c].empty()) touched_cb_.push_back(c);
            cell_baryons_[c].push_back(i);
        }

        struct ReadyMicroCell {
            int cell_id = 0, ix = 0, iy = 0, iz = 0, morton_blk = 0;
            int Np = 0, Nn = 0, Bc = 0;
            double Ec = 0.0, Ec_t00 = 0.0, Pxc = 0.0, Pyc = 0.0, Pzc = 0.0;
            double T_gev = 0.0, dnb_dt = 0.0, gnx = 0.0, gny = 0.0, gnz = 0.0;
            double Pstar[6] = {};
            double n_expect[NUM_SPECIES] = {};
            std::size_t rec_idx = 0;
        };
        std::vector<ReadyMicroCell> ready_cells;

        std::sort(touched_cb_.begin(), touched_cb_.end());
        struct TriggeredCell {
            int c = 0;
            bool crossed_downward = false;
            PhiloxRng rng{0ULL, 0ULL, 0ULL};
        };
        std::vector<TriggeredCell> trig_cells;
        trig_cells.reserve(touched_cb_.size());

        for (int c : touched_cb_) {
            auto& idxs = cell_baryons_[c];
            const double nb_cell = snap.fields[20][c];
            const bool post_sep_expanding = (t_fm >= 1.35 * cfg_.min_time_fm) &&
                                            (nb_cell < cfg_.n_fr_fm3) &&
                                            (dnb_dt_vec_[c] <= 0.0);
            const bool crossed_downward = ((peak_nb_[c] >= cfg_.n_fr_fm3) &&
                                           (nb_cell < cfg_.n_fr_fm3) &&
                                           (dnb_dt_vec_[c] <= 0.0)) || post_sep_expanding;
            if (!force_finalize && !(cfg_.require_peak_crossing ? crossed_downward : (nb_cell < cfg_.n_fr_fm3))) {
                continue;
            }

            PhiloxRng rng(seed ^ 0xC1057E2ULL, static_cast<std::uint64_t>(step), static_cast<std::uint64_t>(c));

            // Decay any excited Delta(1232) inside this freeze-out cell first -> N + pi (exact P^mu, B, Q)
            for (std::size_t k = 0; k < idxs.size(); ++k) {
                HostParticle& hp = host_p[idxs[k]];
                if (!is_delta(hp.species)) continue;
                const ResonanceDecaySpec dspec = get_decay_spec(hp.species);
                double br_w[3] = {}, br_sum = 0.0;
                for (int b = 0; b < dspec.n_branches; ++b) {
                    br_w[b] = branch_partial_width_gev(hp.species, b, hp.mass);
                    br_sum += br_w[b];
                }
                int ch = 0;
                const double tgt = rng.uniform() * br_sum;
                double acc = 0.0;
                for (int b = 0; b < dspec.n_branches; ++b) {
                    acc += br_w[b];
                    if (tgt <= acc) { ch = b; break; }
                }
                const int d1 = dspec.branches[ch].d1, d2 = dspec.branches[ch].d2;
                const double ma = get_species_info(d1).pole_mass_gev, mb = get_species_info(d2).pole_mass_gev;
                const double m_eff = std::max(hp.mass, ma + mb + 1.0e-4);
                const double E_eff = std::sqrt(m_eff * m_eff + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
                FourVector pa{}, pb{};
                decay_isotropic_1to2(FourVector{E_eff, hp.px, hp.py, hp.pz}, m_eff, ma, mb,
                                     1.0 - 2.0 * rng.uniform(), phys::TWO_PI * rng.uniform(), pa, pb);
                HostParticle hpi = hp;
                hp.species = d1; hp.mass = ma; hp.E = pa.t; hp.px = pa.x; hp.py = pa.y; hp.pz = pa.z;
                hpi.species = d2; hpi.mass = mb; hpi.E = pb.t; hpi.px = pb.x; hpi.py = pb.y; hpi.pz = pb.z;
                hpi.freezeout_tag = 1;
                spawned.push_back(hpi);
                mutated = true;
            }
            trig_cells.push_back(TriggeredCell{c, crossed_downward, rng});
        }

        // Parallelize local EoS inversion + Pratt-Torrieri tensor diagonalization across CPU threads!
        struct CellThermo {
            int ix = 0, iy = 0, iz = 0, Np = 0, Nn = 0, Bc = 0;
            double Ec = 0.0, Ec_t00 = 0.0, Pxc = 0.0, Pyc = 0.0, Pzc = 0.0;
            double vx_e = 0.0, vy_e = 0.0, vz_e = 0.0, u0 = 1.0;
            double T_gev = 0.0, vol_scale = 0.0;
            double Pstar[6] = {};
            double Lambda[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            double n_eq[NUM_SPECIES] = {};
            FreezeoutCellRecord rec{};
        };
        const std::size_t n_trig = trig_cells.size();
        std::vector<CellThermo> thermos(n_trig);

        auto compute_thermo_range = [&](std::size_t m0, std::size_t m1) {
            for (std::size_t m = m0; m < m1; ++m) {
                const int c = trig_cells[m].c;
                const bool crossed_downward = trig_cells[m].crossed_downward;
                const auto& idxs = cell_baryons_[c];
                const double nb_cell = snap.fields[20][c];
                CellThermo& th = thermos[m];

                int Np = 0, Nn = 0;
                double Ec = 0.0, Pxc = 0.0, Pyc = 0.0, Pzc = 0.0;
                double jbx_c = 0.0, jby_c = 0.0, jbz_c = 0.0;
                for (std::size_t idx : idxs) {
                    const HostParticle& hp = host_p[idx];
                    if (hp.species == PROTON) ++Np; else ++Nn;
                    Ec += hp.E; Pxc += hp.px; Pyc += hp.py; Pzc += hp.pz;
                    const double inv_e = 1.0 / hp.E;
                    jbx_c += hp.px * inv_e; jby_c += hp.py * inv_e; jbz_c += hp.pz * inv_e;
                }
                const int Bc = Np + Nn;
                const double P2 = Pxc * Pxc + Pyc * Pyc + Pzc * Pzc;
                const double m_rest_nuc = Np * phys::MASS_PROTON + Nn * phys::MASS_NEUTRON;

                double vx_e = jbx_c / static_cast<double>(Bc);
                double vy_e = jby_c / static_cast<double>(Bc);
                double vz_e = jbz_c / static_cast<double>(Bc);
                if (cfg_.sampling == FreezeoutSampling::GrandCanonical && Bc < 16 && snap.fields[10][c] > 1.0e-5f) {
                    vx_e = snap.fields[21][c];
                    vy_e = snap.fields[22][c];
                    vz_e = snap.fields[23][c];
                }
                const double ve2 = std::min(0.9990, vx_e * vx_e + vy_e * vy_e + vz_e * vz_e);
                const double u0 = 1.0 / std::sqrt(1.0 - ve2);
                const double ux = u0 * vx_e, uy = u0 * vy_e, uz = u0 * vz_e;
                const double nb_raw = static_cast<double>(Bc) / (std::max(1, n_test) * gcfg_.cell_volume_fm3() * u0);
                const double nb_loc = std::clamp(std::max(nb_cell, nb_raw), 0.005, cfg_.n_fr_fm3);
                const double nB_eff = (crossed_downward && !force_finalize) ? cfg_.n_fr_fm3 : nb_loc;

                const double delta_c = static_cast<double>(Nn - Np) / static_cast<double>(Bc);
                double e_pot_cell_gev = 0.0;
                if (mf_cfg.enabled) {
                    const double rho_b_lab = nB_eff * u0;
                    double md_avg = 0.0;
                    if (std::fabs(mf_cfg.C_md_gev) > 1.0e-12) {
                        const double pfx = Pxc / Bc, pfy = Pyc / Bc, pfz = Pzc / Bc;
                        const double inv_lam2 = 1.0 / (mf_cfg.Lambda_md_gev * mf_cfg.Lambda_md_gev);
                        for (std::size_t idx : idxs) {
                            const HostParticle& hp = host_p[idx];
                            const double qx = hp.px - pfx, qy = hp.py - pfy, qz = hp.pz - pfz;
                            md_avg += 1.0 / (1.0 + (qx * qx + qy * qy + qz * qz) * inv_lam2);
                        }
                        md_avg /= static_cast<double>(Bc);
                    }
                    const double psi_val = (rho_b_lab / mf_cfg.rho0_fm3) * md_avg;
                    const double v_pot_dens = hcgf_detail::eval_skyrme_md_t00_pot_density(rho_b_lab, delta_c, psi_val, mf_cfg);
                    e_pot_cell_gev = static_cast<double>(Bc) * (v_pot_dens / std::max(rho_b_lab, 1.0e-10));
                }

                const double E_floor = std::sqrt(P2 + (0.992 * m_rest_nuc) * (0.992 * m_rest_nuc));
                const double Ec_t00 = mf_cfg.enabled ? std::max(E_floor, Ec + e_pot_cell_gev) : Ec;
                double e_th_gev = (u0 * Ec_t00 - (Pxc * ux + Pyc * uy + Pzc * uz) - m_rest_nuc) / static_cast<double>(Bc);
                if (cfg_.sampling != FreezeoutSampling::GrandCanonical && Bc == 1 && nb_cell > 1.0e-5 && snap.fields[18][c] > 0.0f) {
                    e_th_gev = 0.5 * ((static_cast<double>(snap.fields[18][c]) / nb_cell) - m_rest_nuc);
                }
                e_th_gev = std::clamp(e_th_gev, -0.0065, 0.350);
                const bool use_smooth_yp = (cfg_.sampling == FreezeoutSampling::GrandCanonical) || (Bc < 16);
                const double Yp = (!use_smooth_yp || snap.fields[10][c] <= 1.0e-6f)
                                      ? (static_cast<double>(Np) / static_cast<double>(Bc))
                                      : std::clamp(static_cast<double>(snap.fields[14][c]) /
                                                   static_cast<double>(snap.fields[10][c]),
                                                   0.001, 0.999);

                double T_gev = 0.0, muB_gev = 0.0, muQ_gev = 0.0;
                fo_detail::invert_local_cluster_eos(nB_eff, Yp, e_th_gev, cfg_.include_excited_clusters,
                                                    cfg_.max_cluster_A, T_gev, muB_gev, muQ_gev, th.n_eq,
                                                    cfg_.quantum_statistics);

                for (int s = 0; s < 6; ++s) th.Pstar[s] = snap.pkin_eckart[s][c];
                double alphas[3] = {1.0, 1.0, 1.0};
                if (cfg_.local_anisotropy) {
                    hcgf_detail::compute_pratt_torrieri_lambda(th.Pstar, th.Lambda, alphas);
                }
                const double a_min = std::min({alphas[0], alphas[1], alphas[2]});
                const double a_max = std::max({alphas[0], alphas[1], alphas[2]});
                const double p_xy_sum = th.Pstar[0] + th.Pstar[3];

                const int ix = c % gcfg_.nx, iy = (c / gcfg_.nx) % gcfg_.ny, iz = c / (gcfg_.nx * gcfg_.ny);
                th.ix = ix; th.iy = iy; th.iz = iz;
                th.Np = Np; th.Nn = Nn; th.Bc = Bc;
                th.Ec = Ec; th.Ec_t00 = Ec_t00; th.Pxc = Pxc; th.Pyc = Pyc; th.Pzc = Pzc;
                th.vx_e = vx_e; th.vy_e = vy_e; th.vz_e = vz_e; th.u0 = u0;
                th.T_gev = T_gev;

                double nB_hrg = 0.0;
                for (int id = 0; id < NUM_SPECIES; ++id) nB_hrg += get_species_info(id).baryon_num * th.n_eq[id];
                th.vol_scale = (nB_hrg > 1.0e-12) ? (static_cast<double>(Bc) / nB_hrg) : 0.0;

                th.rec.step = step; th.rec.cell_id = c; th.rec.t_fm = t_fm;
                th.rec.x_fm = gcfg_.xmin + (ix + 0.5) * gcfg_.dx();
                th.rec.y_fm = gcfg_.ymin + (iy + 0.5) * gcfg_.dy();
                th.rec.z_fm = gcfg_.zmin + (iz + 0.5) * gcfg_.dz();
                th.rec.nB_eckart_fm3 = nb_cell;
                th.rec.e_rest_mev_per_b = e_th_gev * 1.0e3;
                th.rec.T_mev = T_gev * 1.0e3; th.rec.muB_mev = muB_gev * 1.0e3; th.rec.muQ_mev = muQ_gev * 1.0e3;
                th.rec.ux = ux; th.rec.uy = uy; th.rec.uz = uz;
                th.rec.aniso_stretch = (a_min > 0.0) ? (a_max / a_min) : 1.0;
                th.rec.shear_xy = (p_xy_sum > 1.0e-14) ? ((th.Pstar[0] - th.Pstar[3]) / p_xy_sum) : 0.0;
                th.rec.baryons_converted = Bc;
            }
        };

        if (n_trig > 0) {
            (void)fo_detail::get_thermal_base_lut();  // Ensure static LUT initialized before spawning threads
            const unsigned int hw = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
            if (n_trig >= 32 && hw > 1) {
                const std::size_t n_th = std::min<std::size_t>(hw, (n_trig + 15) / 16);
                std::vector<std::thread> pool;
                pool.reserve(n_th);
                for (std::size_t t = 0; t < n_th; ++t) {
                    const std::size_t m0 = (t * n_trig) / n_th;
                    const std::size_t m1 = ((t + 1) * n_trig) / n_th;
                    pool.emplace_back(compute_thermo_range, m0, m1);
                }
                for (auto& th_obj : pool) th_obj.join();
            } else {
                compute_thermo_range(0, n_trig);
            }
        }

        for (std::size_t m = 0; m < n_trig; ++m) {
            const int c = trig_cells[m].c;
            PhiloxRng& rng = trig_cells[m].rng;
            auto& idxs = cell_baryons_[c];
            const CellThermo& th = thermos[m];
            const int ix = th.ix, iy = th.iy, iz = th.iz;
            const int Np = th.Np, Nn = th.Nn, Bc = th.Bc;
            const double Ec = th.Ec, Ec_t00 = th.Ec_t00, Pxc = th.Pxc, Pyc = th.Pyc, Pzc = th.Pzc;
            const double vx_e = th.vx_e, vy_e = th.vy_e, vz_e = th.vz_e, u0 = th.u0;
            const double T_gev = th.T_gev, vol_scale = th.vol_scale;
            const double* Pstar_cell = th.Pstar;
            const double* Lambda_cell = th.Lambda;
            const double* n_eq = th.n_eq;
            FreezeoutCellRecord rec = th.rec;

            if (cfg_.mode == ClusterMode::None) {
                for (std::size_t idx : idxs) host_p[idx].freezeout_tag = 1;
                records_.push_back(rec);
                mutated = true;
                continue;
            }

            if (cfg_.sampling == FreezeoutSampling::GrandCanonical) {
                double gnx = 0.0, gny = 0.0, gnz = 0.0;
                mf_detail::cell_centered_grad(snap.fields[20].data(), ix, iy, iz, gcfg_, gnx, gny, gnz);
                const bool outward_cut = (!force_finalize && cfg_.backflow == BackflowMode::OutwardOnly);
                // Use smooth coarse-grained Eckart fluid velocity on sparse 1-fm^3 cells to avoid small-Bc boost noise!
                double vxg = vx_e, vyg = vy_e, vzg = vz_e, u0g = u0;
                if (Bc < 16 && snap.fields[10][c] > 1.0e-5f) {
                    vxg = snap.fields[21][c]; vyg = snap.fields[22][c]; vzg = snap.fields[23][c];
                    u0g = 1.0 / std::sqrt(1.0 - std::min(0.9990, vxg * vxg + vyg * vyg + vzg * vzg));
                }
                const int m_over = std::max(1, cfg_.oversample_factor);
                int n_cl_gc = 0, inward_gc = 0;
                std::size_t slot_cur = 0;

                for (int id = 0; id < NUM_SPECIES; ++id) {
                    const SpeciesInfo sp = get_species_info(id);
                    if (sp.baryon_num < 1 || sp.baryon_num > cfg_.max_cluster_A || is_delta(id) || n_eq[id] <= 0.0) {
                        continue;
                    }
                    if (sp.is_excited && !cfg_.include_excited_clusters) continue;
                    const int n_draw = fo_detail::sample_poisson(static_cast<double>(m_over) * vol_scale * n_eq[id], rng);
                    for (int d = 0; d < n_draw; ++d) {
                        const double mk = sp.is_excited ? sample_thermal_breit_wigner_mass(id, T_gev, rng)
                                                        : sp.pole_mass_gev;
                        const FourVector q = sample_maxwell_juttner(mk, T_gev, rng);
                        const double pxs = Lambda_cell[0] * q.x + Lambda_cell[1] * q.y + Lambda_cell[2] * q.z;
                        const double pys = Lambda_cell[3] * q.x + Lambda_cell[4] * q.y + Lambda_cell[5] * q.z;
                        const double pzs = Lambda_cell[6] * q.x + Lambda_cell[7] * q.y + Lambda_cell[8] * q.z;
                        const double es = std::sqrt(mk * mk + pxs * pxs + pys * pys + pzs * pzs);
                        const FourVector plab = lorentz_boost_from_cm(mk, es, pxs, pys, pzs, vxg, vyg, vzg, u0g);

                        const double adv = dnb_dt_vec_[c] + (plab.x * gnx + plab.y * gny + plab.z * gnz) / plab.t;
                        if (adv > 0.0) {
                            ++inward_gc;
                            if (outward_cut) continue;
                        }

                        HostParticle hp = host_p[idxs[slot_cur % idxs.size()]];
                        if (slot_cur >= idxs.size()) {
                            hp.x = gcfg_.xmin + (ix + rng.uniform()) * gcfg_.dx();
                            hp.y = gcfg_.ymin + (iy + rng.uniform()) * gcfg_.dy();
                            hp.z = gcfg_.zmin + (iz + rng.uniform()) * gcfg_.dz();
                        }
                        ++slot_cur;
                        hp.species = id; hp.mass = mk;
                        hp.E = plab.t; hp.px = plab.x; hp.py = plab.y; hp.pz = plab.z;
                        hp.collided = 5; hp.freezeout_tag = m_over;
                        spawned.push_back(hp);
                        if (sp.baryon_num >= 2) ++n_cl_gc;
                    }
                }
                for (std::size_t idx : idxs) remove_mask[idx] = 1;
                const int cl_scaled = static_cast<int>(std::llround(static_cast<double>(n_cl_gc) / m_over));
                const int in_scaled = static_cast<int>(std::llround(static_cast<double>(inward_gc) / m_over));
                rec.clusters_formed = cl_scaled;
                rec.back_emitted = in_scaled;
                records_.push_back(rec);
                total_baryons_converted_ += Bc;
                total_clusters_formed_ += cl_scaled;
                back_emitted_total_ += in_scaled;
                mutated = true;
                continue;
            }

            ReadyMicroCell mc{};
            mc.cell_id = c; mc.ix = ix; mc.iy = iy; mc.iz = iz;
            mc.morton_blk = ((iz / 3) * 1024) + ((iy / 3) * 32) + (ix / 3);
            mc.Np = Np; mc.Nn = Nn; mc.Bc = Bc;
            mc.Ec = Ec; mc.Ec_t00 = Ec_t00; mc.Pxc = Pxc; mc.Pyc = Pyc; mc.Pzc = Pzc;
            mc.T_gev = T_gev; mc.dnb_dt = dnb_dt_vec_[c];
            mf_detail::cell_centered_grad(snap.fields[20].data(), ix, iy, iz, gcfg_, mc.gnx, mc.gny, mc.gnz);
            for (int s = 0; s < 6; ++s) mc.Pstar[s] = Pstar_cell[s];
            for (int id = 0; id < NUM_SPECIES; ++id) mc.n_expect[id] = vol_scale * n_eq[id];
            mc.rec_idx = records_.size();
            records_.push_back(rec);
            ready_cells.push_back(mc);
        }

        // Sort sparse micro-cells by spatial 3x3x3 neighborhood so grouped parcels are strictly local (~2-3 fm)
        if (ready_cells.size() > 1) {
            const bool any_sparse = std::any_of(ready_cells.begin(), ready_cells.end(),
                                                [](const ReadyMicroCell& m) { return m.Bc < 28; });
            if (any_sparse) {
                std::stable_sort(ready_cells.begin(), ready_cells.end(),
                                 [](const ReadyMicroCell& a, const ReadyMicroCell& b) {
                                     return a.morton_blk < b.morton_blk;
                                 });
            }
        }

        constexpr int MIN_PARCEL_B = 28;
        for (std::size_t p0 = 0; p0 < ready_cells.size(); ) {
            std::size_t p1 = p0 + 1;
            int Bp = ready_cells[p0].Bc;
            while (p1 < ready_cells.size() && Bp < MIN_PARCEL_B) {
                Bp += ready_cells[p1].Bc;
                ++p1;
            }

            int Np = 0, Nn = 0;
            double Ep = 0.0, Ep_t00 = 0.0, Pxp = 0.0, Pyp = 0.0, Pzp = 0.0;
            double T_w = 0.0, dnb_w = 0.0, gnx_w = 0.0, gny_w = 0.0, gnz_w = 0.0;
            double Pstar_p[6] = {}, n_exp_p[NUM_SPECIES] = {};
            std::vector<std::size_t> p_idxs;
            p_idxs.reserve(static_cast<std::size_t>(Bp));

            for (std::size_t m = p0; m < p1; ++m) {
                const ReadyMicroCell& mc = ready_cells[m];
                Np += mc.Np; Nn += mc.Nn;
                Ep += mc.Ec; Ep_t00 += mc.Ec_t00;
                Pxp += mc.Pxc; Pyp += mc.Pyc; Pzp += mc.Pzc;
                T_w += mc.Bc * mc.T_gev;
                dnb_w += mc.Bc * mc.dnb_dt;
                gnx_w += mc.Bc * mc.gnx; gny_w += mc.Bc * mc.gny; gnz_w += mc.Bc * mc.gnz;
                for (int s = 0; s < 6; ++s) Pstar_p[s] += mc.Bc * mc.Pstar[s];
                for (int id = 0; id < NUM_SPECIES; ++id) n_exp_p[id] += mc.n_expect[id];
                for (std::size_t idx : cell_baryons_[mc.cell_id]) p_idxs.push_back(idx);
            }
            if (Bp < 2) {
                for (std::size_t idx : p_idxs) host_p[idx].freezeout_tag = 1;
                mutated = true;
                p0 = p1;
                continue;
            }
            const double inv_Bp = 1.0 / static_cast<double>(Bp);
            const double T_gev = T_w * inv_Bp;
            for (int s = 0; s < 6; ++s) Pstar_p[s] *= inv_Bp;
            double Lambda_p[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, alp_p[3] = {1.0, 1.0, 1.0};
            if (cfg_.local_anisotropy) hcgf_detail::compute_pratt_torrieri_lambda(Pstar_p, Lambda_p, alp_p);

            const double P2 = Pxp * Pxp + Pyp * Pyp + Pzp * Pzp;
            const double m_rest_nuc = Np * phys::MASS_PROTON + Nn * phys::MASS_NEUTRON;
            const double Mp_kin = (Ep * Ep > P2) ? std::sqrt(Ep * Ep - P2) : m_rest_nuc;
            const double Mp = mf_cfg.enabled
                                  ? std::max(m_rest_nuc - 0.0065 * Bp, std::sqrt(std::max(0.0, Ep_t00 * Ep_t00 - P2)))
                                  : Mp_kin;

            PhiloxRng rng(seed ^ 0xC1057E2ULL, static_cast<std::uint64_t>(step),
                          static_cast<std::uint64_t>(ready_cells[p0].cell_id) + 0x55AAULL);

            std::vector<int> out_species;
            int rem_p = Np, rem_n = Nn, n_cl_formed = 0;
            double m_out_pole = 0.0;
            for (int target_A = cfg_.max_cluster_A; target_A >= 2; --target_A) {
                for (int id = 0; id < NUM_SPECIES; ++id) {
                    const SpeciesInfo sp = get_species_info(id);
                    if (sp.baryon_num != target_A || n_exp_p[id] <= 0.0) continue;
                    const int Z = sp.charge, N = sp.baryon_num - sp.charge;
                    if (Z < 0 || N < 0) continue;
                    const int n_draw = fo_detail::sample_poisson(n_exp_p[id], rng);
                    for (int d = 0; d < n_draw; ++d) {
                        const int after_rem = (rem_p - Z) + (rem_n - N) + static_cast<int>(out_species.size()) + 1;
                        const double m_cand = m_out_pole + sp.pole_mass_gev +
                                              (rem_p - Z) * phys::MASS_PROTON + (rem_n - N) * phys::MASS_NEUTRON;
                        const bool mass_ok = mf_cfg.enabled || (m_cand < Mp_kin - 1.0e-3);
                        if (rem_p >= Z && rem_n >= N && after_rem >= 2 && mass_ok) {
                            out_species.push_back(id);
                            m_out_pole += sp.pole_mass_gev;
                            rem_p -= Z; rem_n -= N;
                            ++n_cl_formed;
                        }
                    }
                }
            }

            records_[ready_cells[p0].rec_idx].clusters_formed = n_cl_formed;
            total_baryons_converted_ += Bp;
            total_clusters_formed_ += n_cl_formed;

            if (n_cl_formed == 0) {
                for (std::size_t idx : p_idxs) host_p[idx].freezeout_tag = 1;
                mutated = true;
                p0 = p1;
                continue;
            }
            for (int k = 0; k < rem_p; ++k) out_species.push_back(PROTON);
            for (int k = 0; k < rem_n; ++k) out_species.push_back(NEUTRON);

            const int K = static_cast<int>(out_species.size());
            std::vector<double> mk(K);
            double m_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                const SpeciesInfo sp = get_species_info(out_species[k]);
                mk[k] = sp.is_excited ? sample_thermal_breit_wigner_mass(out_species[k], T_gev, rng) : sp.pole_mass_gev;
                m_sum += mk[k];
            }
            if (!mf_cfg.enabled && m_sum >= Mp - 1.0e-5) {
                m_sum = 0.0;
                for (int k = 0; k < K; ++k) {
                    mk[k] = get_species_info(out_species[k]).pole_mass_gev;
                    m_sum += mk[k];
                }
            }

            const bool outward_cut = (!force_finalize && cfg_.backflow == BackflowMode::OutwardOnly);
            const double Mp_use = mf_cfg.enabled ? std::max(Mp, m_sum + 0.0012 * K) : Mp;
            const double Ep_use = mf_cfg.enabled ? std::sqrt(Mp_use * Mp_use + P2) : Ep;
            // Ensure sampling temperature matches available parcel Q-value so k-body Newton solver never rejects cold parcels
            const double T_kin_sample = std::max(0.0015, std::min(T_gev, 0.67 * (Mp_use - m_sum) / std::max(1, K - 1)));
            std::vector<FourVector> p_out;
            int inward_cnt = 0;
            if (!fo_detail::sample_conservative_kbody(
                    out_species, mk, T_kin_sample, Ep_use, Pxp, Pyp, Pzp,
                    outward_cut, dnb_w * inv_Bp, gnx_w * inv_Bp, gny_w * inv_Bp, gnz_w * inv_Bp,
                    rng, p_out, inward_cnt, cfg_.local_anisotropy ? Lambda_p : nullptr)) {
                p0 = p1;
                continue;
            }
            records_[ready_cells[p0].rec_idx].back_emitted = inward_cnt;
            back_emitted_total_ += inward_cnt;

            for (int k = 0; k < K; ++k) {
                HostParticle hp = host_p[p_idxs[k % p_idxs.size()]];
                hp.species = out_species[k];
                hp.mass = mk[k];
                hp.E = p_out[k].t; hp.px = p_out[k].x; hp.py = p_out[k].y; hp.pz = p_out[k].z;
                hp.collided = 5;
                hp.freezeout_tag = 1;
                spawned.push_back(hp);
            }
            for (std::size_t idx : p_idxs) remove_mask[idx] = 1;
            mutated = true;
            p0 = p1;
        }

        if (mutated) {
            next_p_.clear();
            next_p_.reserve(host_p.size() + spawned.size());
            for (std::size_t i = 0; i < host_p.size(); ++i) {
                if (!remove_mask[i]) next_p_.push_back(host_p[i]);
            }
            for (const auto& sp : spawned) next_p_.push_back(sp);
            particles.upload(next_p_);
        }
        return snap;
    }

    void write_freezeout_cells_csv() const {
        if (cfg_.cells_csv_path.empty()) return;
        std::ofstream f(cfg_.cells_csv_path);
        f << "step,cell_id,t_fm,x_fm,y_fm,z_fm,nB_eckart_fm3,e_rest_mev_per_b,T_mev,muB_mev,muQ_mev,"
             "ux,uy,uz,aniso_stretch,shear_xy,baryons_converted,clusters_formed,back_emitted\n";
        for (const auto& r : records_) {
            f << r.step << "," << r.cell_id << "," << r.t_fm << ","
              << r.x_fm << "," << r.y_fm << "," << r.z_fm << ","
              << r.nB_eckart_fm3 << "," << r.e_rest_mev_per_b << ","
              << r.T_mev << "," << r.muB_mev << "," << r.muQ_mev << ","
              << r.ux << "," << r.uy << "," << r.uz << ","
              << r.aniso_stretch << "," << r.shear_xy << ","
              << r.baryons_converted << "," << r.clusters_formed << "," << r.back_emitted << "\n";
        }
    }

    int total_cells_frozen() const { return static_cast<int>(records_.size()); }
    long long total_baryons_converted() const { return total_baryons_converted_; }
    long long total_clusters_formed() const { return total_clusters_formed_; }
    long long reabsorbed_nucleons() const { return reabsorbed_nucleons_; }
    long long remelted_clusters() const { return remelted_clusters_; }
    long long back_emitted_total() const { return back_emitted_total_; }
    long long thinned_oversamples() const { return thinned_oversamples_; }

private:
    FreezeoutConfig cfg_;
    GridConfig gcfg_;
    HcgfGpuWorkspace gpu_ws_;
    std::vector<double> peak_nb_, prev_nb_, dnb_dt_vec_;
    std::vector<std::vector<std::size_t>> dense_nucleons_, back_clusters_, cell_baryons_;
    std::vector<int> touched_dn_, touched_bc_, touched_cb_;
    std::vector<HostParticle> next_p_;
    double prev_t_fm_ = -1.0;
    std::vector<FreezeoutCellRecord> records_;
    long long total_baryons_converted_ = 0;
    long long total_clusters_formed_ = 0;
    long long reabsorbed_nucleons_ = 0;
    long long remelted_clusters_ = 0;
    long long back_emitted_total_ = 0;
    long long thinned_oversamples_ = 0;
};

}  // namespace transport
