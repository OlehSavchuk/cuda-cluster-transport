#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
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

struct FreezeoutConfig {
    ClusterMode mode = ClusterMode::Microscopic;
    double n_fr_fm3 = 0.040;          // Freeze-out proper baryon density threshold (~rho_0 / 4) [fm^-3]
    double min_time_fm = 0.0;         // Minimum time before expansion freeze-out check activates [fm/c]
    bool include_excited_clusters = true;  // Include d* and 4He* statistical states (feed-down to p,n,d,t,3He)
    std::string cells_csv_path;       // Optional CSV log of every freeze-out cell (t, x, y, z, T, muB, muQ, u^mu)
};

struct FreezeoutCellRecord {
    int step = 0;
    int cell_id = 0;
    double t_fm = 0.0;
    double x_fm = 0.0, y_fm = 0.0, z_fm = 0.0;
    double nB_lrf_fm3 = 0.0;
    double eps_lrf_gev_fm3 = 0.0;
    double T_mev = 0.0;
    double muB_mev = 0.0;
    double muQ_mev = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    int baryons_converted = 0;
    int clusters_formed = 0;
};

namespace fo_detail {

// Relativistic Maxwell-Juttner phase-space prefactor phi_a(T) [fm^-3] and mean excitation
// energy <E_a(p) - Z m_p - N m_n>(T) [GeV] so n_a = phi_a(T) * exp(Z * u_p + N * u_n)
inline void species_thermal_moments(int sp_id, double T_gev, double& phi_fm3, double& e_rel_gev) {
    using namespace phys;
    const SpeciesInfo sp = get_species_info(sp_id);
    const double m = sp.pole_mass_gev;
    const double bind_shift = m - (sp.charge * MASS_PROTON + (sp.baryon_num - sp.charge) * MASS_NEUTRON);
    if ( -bind_shift / T_gev > 120.0 ) {
        phi_fm3 = 0.0; e_rel_gev = bind_shift; return;
    }
    constexpr int N_Q = 48;
    constexpr double S_MAX = 6.5;
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
    phi_fm3 = pref * T_gev * i_n * std::exp(-bind_shift / T_gev);
    e_rel_gev = bind_shift + ((i_n > 0.0) ? (i_k / i_n) : (1.5 * T_gev));
}

// Given (T_gev, nB_fm3, Yp), solve exact 2x2 SPD Newton system for (u_p, u_n) = ((mu_p-m_p)/T, (mu_n-m_n)/T)
// and return equilibrium densities n_sp[NUM_SPECIES] + energy per baryon e_per_baryon_gev.
inline double solve_hrg_fugacities(double T_gev, double nB_fm3, double Yp, bool inc_exc,
                                   double n_out[NUM_SPECIES], double& out_up, double& out_un) {
    const double np_tgt = std::max(1.0e-9, Yp * nB_fm3);
    const double nn_tgt = std::max(1.0e-9, (1.0 - Yp) * nB_fm3);

    double phi[NUM_SPECIES] = {}, erel[NUM_SPECIES] = {};
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (is_pion(id) || is_delta(id)) continue;
        if (sp.is_excited && !inc_exc) continue;
        species_thermal_moments(id, T_gev, phi[id], erel[id]);
    }

    double up = std::log(np_tgt / std::max(phi[PROTON], 1.0e-30));
    double un = std::log(nn_tgt / std::max(phi[NEUTRON], 1.0e-30));
    for (int it = 0; it < 16; ++it) {
        double fp = -np_tgt, fn = -nn_tgt, jpp = 0.0, jpn = 0.0, jnn = 0.0;
        for (int id = 0; id < NUM_SPECIES; ++id) {
            if (phi[id] <= 0.0) continue;
            const SpeciesInfo sp = get_species_info(id);
            const double Z = sp.charge, N = sp.baryon_num - sp.charge;
            const double arg = std::clamp(Z * up + N * un, -200.0, 80.0);
            const double na = phi[id] * std::exp(arg);
            fp += Z * na; fn += N * na;
            jpp += Z * Z * na; jpn += Z * N * na; jnn += N * N * na;
        }
        if (std::fabs(fp) + std::fabs(fn) < 1.0e-13 * nB_fm3) break;
        const double det = jpp * jnn - jpn * jpn;
        if (det <= 1.0e-60) break;
        up -= std::clamp(( jnn * fp - jpn * fn) / det, -2.0, 2.0);
        un -= std::clamp((-jpn * fp + jpp * fn) / det, -2.0, 2.0);
    }
    out_up = up; out_un = un;
    double e_sum = 0.0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        if (phi[id] <= 0.0) { n_out[id] = 0.0; continue; }
        const SpeciesInfo sp = get_species_info(id);
        n_out[id] = phi[id] * std::exp(std::clamp(sp.charge * up + (sp.baryon_num - sp.charge) * un, -200.0, 80.0));
        e_sum += n_out[id] * erel[id];
    }
    return e_sum / std::max(nB_fm3, 1.0e-12);
}

}  // namespace fo_detail

namespace fo_detail {

// Invert HCGF Nuclear-Cluster EoS (nB, Yp, e_th_per_baryon) -> (T_gev, muB_gev, muQ_gev, n_eq[NUM_SPECIES])
inline void invert_local_cluster_eos(double nB_fm3, double Yp, double e_th_gev, bool inc_exc,
                                     double& T_gev, double& muB_gev, double& muQ_gev,
                                     double n_eq[NUM_SPECIES]) {
    double T_lo = 0.0015, T_hi = 0.120;
    double up = 0.0, un = 0.0;
    const double e_lo = solve_hrg_fugacities(T_lo, nB_fm3, Yp, inc_exc, n_eq, up, un);
    if (e_th_gev <= e_lo) {
        T_gev = T_lo;
    } else {
        for (int b = 0; b < 22; ++b) {
            const double T_mid = 0.5 * (T_lo + T_hi);
            const double e_mid = solve_hrg_fugacities(T_mid, nB_fm3, Yp, inc_exc, n_eq, up, un);
            if (e_mid < e_th_gev) T_lo = T_mid;
            else T_hi = T_mid;
        }
        T_gev = 0.5 * (T_lo + T_hi);
        solve_hrg_fugacities(T_gev, nB_fm3, Yp, inc_exc, n_eq, up, un);
    }
    // mu_n = m_n + T * u_n == mu_B;   mu_p = m_p + T * u_p == mu_B + mu_Q
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
        peak_nb_.assign(gcfg.num_cells(), 0.0);
        records_.clear();
        total_baryons_converted_ = 0;
        total_clusters_formed_ = 0;
    }

    // Track local proper baryon density n_B,LRF(c) across grid cells; for cells reaching n_B < n_fr
    // after t >= min_time_fm (or all remaining un-frozen cells when force_finalize == true),
    // record HCGF hypersurface state and (in FreezeoutThermal mode) convert nucleons into thermal clusters!
    HcgfGridSnapshot process_step(ParticleSystem& particles,
                                  int step,
                                  double t_fm,
                                  int n_test,
                                  const MeanFieldConfig& mf_cfg,
                                  std::uint64_t seed,
                                  bool force_finalize = false) {
        auto host_p = particles.download();
        HcgfGridSnapshot snap = compute_hcgf_tmunu_snapshot(n_test, gcfg_, mf_cfg, host_p);
        const int nc = gcfg_.num_cells();
        if (nc <= 0 || cfg_.mode == ClusterMode::Microscopic) return snap;

        for (int c = 0; c < nc; ++c) {
            const double nb = snap.fields[20][c];
            if (nb > peak_nb_[c]) peak_nb_[c] = nb;
        }
        if (!force_finalize && t_fm < cfg_.min_time_fm) return snap;

        // Bucket un-frozen baryons (p, n, Delta) by spatial grid cell
        std::vector<std::vector<std::size_t>> cell_baryons(nc);
        for (std::size_t i = 0; i < host_p.size(); ++i) {
            if (host_p[i].freezeout_tag != 0) continue;
            const SpeciesInfo sp = get_species_info(host_p[i].species);
            if (sp.baryon_num != 1) continue;
            const int ix = static_cast<int>(std::floor((host_p[i].x - gcfg_.xmin) / gcfg_.dx()));
            const int iy = static_cast<int>(std::floor((host_p[i].y - gcfg_.ymin) / gcfg_.dy()));
            const int iz = static_cast<int>(std::floor((host_p[i].z - gcfg_.zmin) / gcfg_.dz()));
            if (ix < 0 || ix >= gcfg_.nx || iy < 0 || iy >= gcfg_.ny || iz < 0 || iz >= gcfg_.nz) continue;
            cell_baryons[mf_detail::cell_flat(ix, iy, iz, gcfg_)].push_back(i);
        }

        bool mutated = false;
        std::vector<std::uint8_t> remove_mask(host_p.size(), 0);
        std::vector<HostParticle> spawned;

        for (int c = 0; c < nc; ++c) {
            auto& idxs = cell_baryons[c];
            if (idxs.size() < 2) continue;
            const double nb_cell = snap.fields[20][c];
            if (!force_finalize && nb_cell >= cfg_.n_fr_fm3) continue;

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

            // Compute exact local cell nucleon parcel state (N_p, N_n, E_c, P_c, M_c, e_th)
            int Np = 0, Nn = 0;
            double Ec = 0.0, Pxc = 0.0, Pyc = 0.0, Pzc = 0.0;
            for (std::size_t idx : idxs) {
                const HostParticle& hp = host_p[idx];
                if (hp.species == PROTON) ++Np; else ++Nn;
                Ec += hp.E; Pxc += hp.px; Pyc += hp.py; Pzc += hp.pz;
            }
            const int Bc = Np + Nn;
            const double P2 = Pxc * Pxc + Pyc * Pyc + Pzc * Pzc;
            const double Mc = (Ec * Ec > P2) ? std::sqrt(Ec * Ec - P2) : (Np * phys::MASS_PROTON + Nn * phys::MASS_NEUTRON);
            const double m_rest_nuc = Np * phys::MASS_PROTON + Nn * phys::MASS_NEUTRON;
            const double e_th_gev = std::max(0.002, (Mc - m_rest_nuc) / static_cast<double>(Bc));
            const double nB_eff = std::clamp(nb_cell, 0.005, cfg_.n_fr_fm3);
            const double Yp = static_cast<double>(Np) / static_cast<double>(Bc);

            double T_gev = 0.0, muB_gev = 0.0, muQ_gev = 0.0, n_eq[NUM_SPECIES] = {};
            fo_detail::invert_local_cluster_eos(nB_eff, Yp, e_th_gev, cfg_.include_excited_clusters,
                                                T_gev, muB_gev, muQ_gev, n_eq);

            const int ix = c % gcfg_.nx, iy = (c / gcfg_.nx) % gcfg_.ny, iz = c / (gcfg_.nx * gcfg_.ny);
            FreezeoutCellRecord rec{};
            rec.step = step; rec.cell_id = c; rec.t_fm = t_fm;
            rec.x_fm = gcfg_.xmin + (ix + 0.5) * gcfg_.dx();
            rec.y_fm = gcfg_.ymin + (iy + 0.5) * gcfg_.dy();
            rec.z_fm = gcfg_.zmin + (iz + 0.5) * gcfg_.dz();
            rec.nB_lrf_fm3 = nb_cell; rec.eps_lrf_gev_fm3 = snap.fields[18][c];
            rec.T_mev = T_gev * 1.0e3; rec.muB_mev = muB_gev * 1.0e3; rec.muQ_mev = muQ_gev * 1.0e3;
            rec.vx = snap.fields[21][c]; rec.vy = snap.fields[22][c]; rec.vz = snap.fields[23][c];
            rec.baryons_converted = Bc;

            if (cfg_.mode == ClusterMode::None) {
                for (std::size_t idx : idxs) host_p[idx].freezeout_tag = 1;
                records_.push_back(rec);
                mutated = true;
                continue;
            }

            // Sample thermally distributed clusters (A = 4 -> A = 3 -> A = 2) from n_eq(T_c, muB_c, muQ_c)
            double nB_hrg = 0.0;
            for (int id = 0; id < NUM_SPECIES; ++id) nB_hrg += get_species_info(id).baryon_num * n_eq[id];
            const double vol_scale = (nB_hrg > 1.0e-12) ? (static_cast<double>(Bc) / nB_hrg) : 0.0;

            std::vector<int> out_species;
            int rem_p = Np, rem_n = Nn, n_cl_formed = 0;
            double m_out_pole = 0.0;
            // Sample heaviest clusters first so rare 4He / 3He / t aren't starved by d draws
            for (int target_A : {4, 3, 2}) {
                for (int id = 0; id < NUM_SPECIES; ++id) {
                    const SpeciesInfo sp = get_species_info(id);
                    if (sp.baryon_num != target_A || n_eq[id] <= 0.0) continue;
                    const int Z = sp.charge, N = sp.baryon_num - sp.charge;
                    if (Z < 0 || N < 0) continue;
                    const int n_draw = fo_detail::sample_poisson(vol_scale * n_eq[id], rng);
                    for (int d = 0; d < n_draw; ++d) {
                        const int after_rem = (rem_p - Z) + (rem_n - N) + static_cast<int>(out_species.size()) + 1;
                        const double m_cand = m_out_pole + sp.pole_mass_gev +
                                              (rem_p - Z) * phys::MASS_PROTON + (rem_n - N) * phys::MASS_NEUTRON;
                        if (rem_p >= Z && rem_n >= N && after_rem >= 2 && m_cand < Mc - 1.0e-3) {
                            out_species.push_back(id);
                            m_out_pole += sp.pole_mass_gev;
                            rem_p -= Z; rem_n -= N;
                            ++n_cl_formed;
                        }
                    }
                }
            }
            rec.clusters_formed = n_cl_formed;
            records_.push_back(rec);
            total_baryons_converted_ += Bc;
            total_clusters_formed_ += n_cl_formed;

            if (n_cl_formed == 0) {
                for (std::size_t idx : idxs) host_p[idx].freezeout_tag = 1;
                mutated = true;
                continue;
            }
            for (int k = 0; k < rem_p; ++k) out_species.push_back(PROTON);
            for (int k = 0; k < rem_n; ++k) out_species.push_back(NEUTRON);

            // Sample Maxwell-Juttner momenta in cell CM frame, enforce exact sum(q_k)==0 & sum(E_k*)==Mc,
            // and Lorentz-boost back by (beta_c, gamma_c) -> 100% exact E, P, B, Q conservation!
            const int K = static_cast<int>(out_species.size());
            std::vector<double> mk(K), qx(K), qy(K), qz(K);
            double m_sum = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
            for (int k = 0; k < K; ++k) {
                const SpeciesInfo sp = get_species_info(out_species[k]);
                mk[k] = sp.is_excited ? sample_thermal_breit_wigner_mass(out_species[k], T_gev, rng) : sp.pole_mass_gev;
                m_sum += mk[k];
                const FourVector q4 = sample_maxwell_juttner(mk[k], T_gev, rng);
                qx[k] = q4.x; qy[k] = q4.y; qz[k] = q4.z;
                sx += qx[k]; sy += qy[k]; sz += qz[k];
            }
            if (m_sum >= Mc - 1.0e-5) {
                // Fallback to ground-state pole masses if wide BW draw exceeded Mc
                m_sum = 0.0;
                for (int k = 0; k < K; ++k) { mk[k] = get_species_info(out_species[k]).pole_mass_gev; m_sum += mk[k]; }
            }
            for (int k = 0; k < K; ++k) {
                const double wf = mk[k] / m_sum;
                qx[k] -= wf * sx; qy[k] -= wf * sy; qz[k] -= wf * sz;
            }
            double eta = 1.0;
            for (int it = 0; it < 10; ++it) {
                double f = -Mc, df = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double q2 = qx[k] * qx[k] + qy[k] * qy[k] + qz[k] * qz[k];
                    const double ek = std::sqrt(mk[k] * mk[k] + eta * eta * q2);
                    f += ek;
                    df += (eta * q2) / ek;
                }
                if (std::fabs(f) < 1.0e-13 || df <= 1.0e-20) break;
                eta = std::max(1.0e-4, eta - f / df);
            }

            const double bx = Pxc / Ec, by = Pyc / Ec, bz = Pzc / Ec;
            const double gam = Ec / Mc;
            for (int k = 0; k < K; ++k) {
                const double px_cm = eta * qx[k], py_cm = eta * qy[k], pz_cm = eta * qz[k];
                const double e_cm = std::sqrt(mk[k] * mk[k] + px_cm * px_cm + py_cm * py_cm + pz_cm * pz_cm);
                const FourVector plab = lorentz_boost_from_cm(mk[k], e_cm, px_cm, py_cm, pz_cm, bx, by, bz, gam);
                HostParticle hp = host_p[idxs[k % idxs.size()]];
                hp.species = out_species[k];
                hp.mass = mk[k];
                hp.E = plab.t; hp.px = plab.x; hp.py = plab.y; hp.pz = plab.z;
                hp.collided = 5;
                hp.freezeout_tag = 1;
                spawned.push_back(hp);
            }
            for (std::size_t idx : idxs) remove_mask[idx] = 1;
            mutated = true;
        }

        if (mutated) {
            std::vector<HostParticle> next_p;
            next_p.reserve(host_p.size() + spawned.size());
            for (std::size_t i = 0; i < host_p.size(); ++i) {
                if (!remove_mask[i]) next_p.push_back(host_p[i]);
            }
            for (const auto& sp : spawned) next_p.push_back(sp);
            particles.upload(next_p);
        }
        return snap;
    }

    void write_freezeout_cells_csv() const {
        if (cfg_.cells_csv_path.empty()) return;
        std::ofstream f(cfg_.cells_csv_path);
        f << "step,cell_id,t_fm,x_fm,y_fm,z_fm,nB_lrf_fm3,eps_lrf_gev_fm3,T_mev,muB_mev,muQ_mev,vx,vy,vz,baryons_converted,clusters_formed\n";
        for (const auto& r : records_) {
            f << r.step << "," << r.cell_id << "," << r.t_fm << ","
              << r.x_fm << "," << r.y_fm << "," << r.z_fm << ","
              << r.nB_lrf_fm3 << "," << r.eps_lrf_gev_fm3 << ","
              << r.T_mev << "," << r.muB_mev << "," << r.muQ_mev << ","
              << r.vx << "," << r.vy << "," << r.vz << ","
              << r.baryons_converted << "," << r.clusters_formed << "\n";
        }
    }

    int total_cells_frozen() const { return static_cast<int>(records_.size()); }
    long long total_baryons_converted() const { return total_baryons_converted_; }
    long long total_clusters_formed() const { return total_clusters_formed_; }

private:
    FreezeoutConfig cfg_;
    GridConfig gcfg_;
    std::vector<double> peak_nb_;
    std::vector<FreezeoutCellRecord> records_;
    long long total_baryons_converted_ = 0;
    long long total_clusters_formed_ = 0;
};

}  // namespace transport
