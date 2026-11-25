#pragma once

#include <cmath>
#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/cross_sections.cuh"
#include "physics/inelastic_channels.cuh"
#include "physics/kinematics.cuh"
#include "physics/mean_field.cuh"
#include "physics/resonances.cuh"
#include "sim/grid.cuh"
#include "sim/particles.cuh"

namespace transport {

enum class CollisionScheme : int {
    DisjointPairs = 0,  // Bird/BAMPS random disjoint pairs: O(N_p), lock-free parallel per cell
    AllPairs = 1        // Full cell pair traversal with single-scatter-per-substep guard
};

struct ReactionConfig {
    bool enable_elastic = true;
    bool enable_inelastic_2to2 = true;
    bool enable_resonance_2to1_and_decays = true;
    double inelastic_scale = 1.0;
    SpectralNormTable norms{};
};

struct StepperConfig {
    double dt_fm = 0.2;             // Time step [fm/c]
    int n_test = 1;                 // Test-particle oversampling factor N_test
    int collision_substeps = 1;     // Sub-cycles per spatial cell build (reduces max P_pair)
    CollisionScheme scheme = CollisionScheme::DisjointPairs;
    std::uint64_t seed = 20261124ULL;
};

struct StepStats {
    unsigned long long pairs_tested = 0;
    unsigned long long collisions_elastic = 0;
    unsigned long long collisions_inelastic_2to2 = 0;
    unsigned long long formations_2to1 = 0;
    unsigned long long decays_1to2 = 0;
    unsigned long long prob_overflows = 0;
    double max_prob_seen = 0.0;

    unsigned long long total_binary_reactions() const {
        return collisions_elastic + collisions_inelastic_2to2 + formations_2to1;
    }
};

namespace detail {

HD_INLINE void drift_single_particle(ParticleSoAView& p, int i, double dt, const GridConfig& gcfg) {
    if (!p.active[i]) return;
    const double inv_E = 1.0 / p.E[i];
    double nx = p.x[i] + (p.px[i] * inv_E) * dt;
    double ny = p.y[i] + (p.py[i] * inv_E) * dt;
    double nz = p.z[i] + (p.pz[i] * inv_E) * dt;

    if (gcfg.boundary == BoundaryMode::PeriodicBox) {
        nx = wrap_periodic(nx, gcfg.xmin, gcfg.lx());
        ny = wrap_periodic(ny, gcfg.ymin, gcfg.ly());
        nz = wrap_periodic(nz, gcfg.zmin, gcfg.lz());
    }

    p.x[i] = nx;
    p.y[i] = ny;
    p.z[i] = nz;
    p.t[i] += dt;
}

HD_INLINE int compact_and_shuffle_cell_indices(const ParticleSoAView& p,
                                               int* indices,
                                               int raw_count,
                                               PhiloxRng& rng) {
    int live = 0;
    for (int a = 0; a < raw_count; ++a) {
        const int idx = indices[a];
        if (p.active[idx]) {
            indices[live++] = idx;
        }
    }
    for (int k = live - 1; k > 0; --k) {
        const int j = static_cast<int>(rng.uniform_int(static_cast<std::uint32_t>(k + 1)));
        const int tmp = indices[k];
        indices[k] = indices[j];
        indices[j] = tmp;
    }
    return live;
}

// Unified pair reaction evaluator:
// Returns 0 (none), 1 (elastic 2->2), 2 (inelastic 2->2), or 3 (resonance formation 2->1)
HD_INLINE int try_pair_reaction(ParticleSoAView& p,
                                int idx1,
                                int idx2,
                                double rate_prefactor,
                                const GridConfig& gcfg,
                                const CrossSectionConfig& xs_cfg,
                                const ReactionConfig& rxn_cfg,
                                PhiloxRng& rng,
                                unsigned long long* d_overflows,
                                double* d_max_prob) {
    if (!p.active[idx1] || !p.active[idx2]) return 0;

    const double m1 = p.mass[idx1];
    const double m2 = p.mass[idx2];
    const FourVector p1_in{p.E[idx1], p.px[idx1], p.py[idx1], p.pz[idx1]};
    const FourVector p2_in{p.E[idx2], p.px[idx2], p.py[idx2], p.pz[idx2]};

    const PairKinematics kin = compute_pair_kinematics(p1_in, m1, p2_in, m2);
    if (kin.p_cm <= 1.0e-12 || kin.v_moller <= 1.0e-15) {
        return 0;
    }

    const int id1 = p.species[idx1];
    const int id2 = p.species[idx2];

    // 1. Elastic 2 -> 2 v_moller * sigma_el [fm^2]
    const double w_el = rxn_cfg.enable_elastic
                            ? (kin.v_moller * elastic_cross_section_fm2(id1, id2, kin, xs_cfg))
                            : 0.0;

    // 2. Binary inelastic 2 -> 2 channels (up to 20 matching channels for N + 4He)
    int inel_out3[20];
    int inel_out4[20];
    double inel_w[20];
    int n_inel = 0;
    double w_inel_sum = 0.0;

    if (rxn_cfg.enable_inelastic_2to2) {
        for (int ch = 0; ch < NUM_INELASTIC_2TO2_CHANNELS; ++ch) {
            int o3 = -1, o4 = -1;
            const double w = inelastic_2to2_vmoller_sigma_fm2(
                ch, id1, id2, p1_in, p2_in, kin.sqrt_s,
                rxn_cfg.norms, rxn_cfg.inelastic_scale, o3, o4);
            if (w > 0.0 && n_inel < 20) {
                inel_out3[n_inel] = o3;
                inel_out4[n_inel] = o4;
                inel_w[n_inel] = w;
                w_inel_sum += w;
                ++n_inel;
            }
        }
    }

    // 3. s-channel 2 -> 1 Breit-Wigner resonance formation (a + b -> R)
    int form_res[18];
    double form_w[18];
    int n_form = 0;
    double w_form_sum = 0.0;

    if (rxn_cfg.enable_resonance_2to1_and_decays) {
        const SpeciesInfo sp1 = get_species_info(id1);
        const SpeciesInfo sp2 = get_species_info(id2);
        if (!sp1.is_excited && !sp2.is_excited) {
            const int B_tot = sp1.baryon_num + sp2.baryon_num;
            const int Z_tot = sp1.charge + sp2.charge;

            if (B_tot == 2) {
                int cand = -1;
                if (Z_tot == 0) cand = D_STAR_NN;
                else if (Z_tot == 1) cand = D_STAR_PN;
                else if (Z_tot == 2) cand = D_STAR_PP;
                if (cand >= 0) {
                    const double w = formation_vmoller_sigma_fm2(
                        id1, id2, p1_in, p2_in, kin.sqrt_s, cand, rxn_cfg.norms);
                    if (w > 0.0) {
                        form_res[n_form] = cand;
                        form_w[n_form] = w;
                        w_form_sum += w;
                        ++n_form;
                    }
                }
            } else if (B_tot == 4) {
                if (Z_tot == 2) {
                    for (int r = HE4_2021; r <= HE4_2989; ++r) {
                        const double w = formation_vmoller_sigma_fm2(
                            id1, id2, p1_in, p2_in, kin.sqrt_s, r, rxn_cfg.norms);
                        if (w > 0.0 && n_form < 18) {
                            form_res[n_form] = r;
                            form_w[n_form] = w;
                            w_form_sum += w;
                            ++n_form;
                        }
                    }
                } else if (Z_tot == 1) {
                    const double w = formation_vmoller_sigma_fm2(
                        id1, id2, p1_in, p2_in, kin.sqrt_s, H4_RES, rxn_cfg.norms);
                    if (w > 0.0) {
                        form_res[n_form] = H4_RES;
                        form_w[n_form] = w;
                        w_form_sum += w;
                        ++n_form;
                    }
                } else if (Z_tot == 3) {
                    const double w = formation_vmoller_sigma_fm2(
                        id1, id2, p1_in, p2_in, kin.sqrt_s, LI4_RES, rxn_cfg.norms);
                    if (w > 0.0) {
                        form_res[n_form] = LI4_RES;
                        form_w[n_form] = w;
                        w_form_sum += w;
                        ++n_form;
                    }
                }
            }
        }
    }

    const double w_tot = w_el + w_inel_sum + w_form_sum;
    const double prob_tot = w_tot * rate_prefactor;
    if (prob_tot <= 0.0) return 0;

    portable_atomic_max_double(d_max_prob, prob_tot);
    if (prob_tot > 1.0) {
        portable_atomic_add_ull(d_overflows, 1ULL);
    }

    const double u_draw = rng.uniform();
    if (u_draw >= prob_tot) {
        return 0;
    }

    // If prob_tot > 1 (overflow), normalize channel selection within w_tot
    const double sel = (prob_tot <= 1.0) ? (u_draw / rate_prefactor) : (rng.uniform() * w_tot);

    // Branch A: Elastic 2 -> 2
    if (sel < w_el) {
        double cos_theta = 0.0, phi = 0.0;
        sample_elastic_angles(id1, id2, kin, xs_cfg, rng, cos_theta, phi);

        FourVector p1_out{}, p2_out{};
        scatter_elastic_2to2(p1_in, m1, p2_in, m2, kin, cos_theta, phi, p1_out, p2_out);

        p.E[idx1] = p1_out.t; p.px[idx1] = p1_out.x; p.py[idx1] = p1_out.y; p.pz[idx1] = p1_out.z;
        p.E[idx2] = p2_out.t; p.px[idx2] = p2_out.x; p.py[idx2] = p2_out.y; p.pz[idx2] = p2_out.z;
        p.collided[idx1] = 1;
        p.collided[idx2] = 1;
        return 1;
    }

    // Branch B: Inelastic 2 -> 2 (including off-shell Breit-Wigner mass sampling!)
    if (sel < w_el + w_inel_sum && n_inel > 0) {
        const double target = sel - w_el;
        double acc = 0.0;
        int chosen = n_inel - 1;
        for (int k = 0; k < n_inel; ++k) {
            acc += inel_w[k];
            if (target <= acc) {
                chosen = k;
                break;
            }
        }

        const int out3 = inel_out3[chosen];
        const int out4 = inel_out4[chosen];
        const SpeciesInfo sp3 = get_species_info(out3);
        const SpeciesInfo sp4 = get_species_info(out4);

        double m3 = sp3.pole_mass_gev;
        double m4 = sp4.pole_mass_gev;
        if (sp3.is_excited && !sp4.is_excited) {
            m3 = sample_breit_wigner_2to2_mass(out3, kin.sqrt_s, m4, rng);
        } else if (!sp3.is_excited && sp4.is_excited) {
            m4 = sample_breit_wigner_2to2_mass(out4, kin.sqrt_s, m3, rng);
        }

        const double p_cm_out = two_body_p_cm(kin.sqrt_s, m3, m4);
        if (p_cm_out <= 0.0) return 0;

        const double cos_theta = 1.0 - 2.0 * rng.uniform();
        const double phi = phys::TWO_PI * rng.uniform();
        FourVector p3_out{}, p4_out{};
        scatter_general_2to2(p1_in, p2_in, kin.sqrt_s, m3, m4, p_cm_out,
                             cos_theta, phi, p3_out, p4_out);

        p.species[idx1] = out3;
        p.mass[idx1] = m3;
        p.E[idx1] = p3_out.t; p.px[idx1] = p3_out.x; p.py[idx1] = p3_out.y; p.pz[idx1] = p3_out.z;
        p.collided[idx1] = 2;

        p.species[idx2] = out4;
        p.mass[idx2] = m4;
        p.E[idx2] = p4_out.t; p.px[idx2] = p4_out.x; p.py[idx2] = p4_out.y; p.pz[idx2] = p4_out.z;
        p.collided[idx2] = 2;
        return 2;
    }

    // Branch C: 2 -> 1 s-channel Resonance Formation (a + b -> R(m_R = sqrt_s))
    if (n_form > 0) {
        const double target = sel - w_el - w_inel_sum;
        double acc = 0.0;
        int chosen = n_form - 1;
        for (int k = 0; k < n_form; ++k) {
            acc += form_w[k];
            if (target <= acc) {
                chosen = k;
                break;
            }
        }
        const int res_id = form_res[chosen];

        const double E_R = p1_in.t + p2_in.t;
        const double Px_R = p1_in.x + p2_in.x;
        const double Py_R = p1_in.y + p2_in.y;
        const double Pz_R = p1_in.z + p2_in.z;
        const double m_R_sq = E_R * E_R - (Px_R * Px_R + Py_R * Py_R + Pz_R * Pz_R);
        if (m_R_sq <= 0.0) return 0;
        const double m_R = ::sqrt(m_R_sq);

        double rx = (p1_in.t * p.x[idx1] + p2_in.t * p.x[idx2]) / E_R;
        double ry = (p1_in.t * p.y[idx1] + p2_in.t * p.y[idx2]) / E_R;
        double rz = (p1_in.t * p.z[idx1] + p2_in.t * p.z[idx2]) / E_R;
        if (gcfg.boundary == BoundaryMode::PeriodicBox) {
            rx = wrap_periodic(rx, gcfg.xmin, gcfg.lx());
            ry = wrap_periodic(ry, gcfg.ymin, gcfg.ly());
            rz = wrap_periodic(rz, gcfg.zmin, gcfg.lz());
        }

        p.species[idx1] = res_id;
        p.mass[idx1] = m_R;
        p.E[idx1] = E_R;
        p.px[idx1] = Px_R;
        p.py[idx1] = Py_R;
        p.pz[idx1] = Pz_R;
        p.x[idx1] = rx;
        p.y[idx1] = ry;
        p.z[idx1] = rz;
        p.collided[idx1] = 3;

        // Deactivate second slot cleanly
        p.active[idx2] = 0;
        p.cell_id[idx2] = -1;
        p.collided[idx2] = 0;
        return 3;
    }

    return 0;
}

// Evaluate 1 -> 2 decay of a single excited resonance particle i over interval dt
HD_INLINE bool try_resonance_decay(ParticleSoAView& p,
                                   int i,
                                   double dt,
                                   PhiloxRng& rng) {
    if (!p.active[i]) return false;
    const int res_id = p.species[i];
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return false;

    const double m_R = p.mass[i];
    const double E_R = p.E[i];
    const double gamma_tot = total_width_gev(res_id, m_R);
    if (gamma_tot <= 0.0) return false;

    // Relativistic time-dilated decay probability P = 1 - exp(-(Gamma / hbar*c) * (m_R / E_R) * dt)
    const double decay_arg = (gamma_tot / phys::HBARC) * (m_R / E_R) * dt;
    const double prob_decay = -::expm1(-decay_arg);
    if (rng.uniform() >= prob_decay) return false;

    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    double br_w[3] = {0.0, 0.0, 0.0};
    double br_sum = 0.0;
    for (int b = 0; b < dspec.n_branches; ++b) {
        br_w[b] = branch_partial_width_gev(res_id, b, m_R);
        br_sum += br_w[b];
    }
    if (br_sum <= 0.0) return false;

    const double target = rng.uniform() * br_sum;
    double acc = 0.0;
    int chosen = 0;
    for (int b = 0; b < dspec.n_branches; ++b) {
        acc += br_w[b];
        if (target <= acc) {
            chosen = b;
            break;
        }
    }

    const int d1_id = dspec.branches[chosen].d1;
    const int d2_id = dspec.branches[chosen].d2;
    const double ma = get_species_info(d1_id).pole_mass_gev;
    const double mb = get_species_info(d2_id).pole_mass_gev;
    if (m_R <= ma + mb + 1.0e-12) return false;

    // Reserve output slot atomically before mutating parent slot
    const int slot_b = portable_atomic_add_int(p.d_n_slots, 1);
    if (slot_b >= p.capacity) {
        // Roll back if capacity exhausted (prevented in practice by ensure_headroom)
        portable_atomic_add_int(p.d_n_slots, -1);
        return false;
    }

    const FourVector p_res{E_R, p.px[i], p.py[i], p.pz[i]};
    const double cos_theta = 1.0 - 2.0 * rng.uniform();
    const double phi = phys::TWO_PI * rng.uniform();
    FourVector pa_out{}, pb_out{};
    decay_isotropic_1to2(p_res, m_R, ma, mb, cos_theta, phi, pa_out, pb_out);

    const double rx = p.x[i], ry = p.y[i], rz = p.z[i], rt = p.t[i];

    // Overwrite slot i with daughter a
    p.species[i] = d1_id;
    p.mass[i] = ma;
    p.E[i] = pa_out.t;
    p.px[i] = pa_out.x;
    p.py[i] = pa_out.y;
    p.pz[i] = pa_out.z;
    p.collided[i] = 4;

    // Populate newly allocated slot_b with daughter b
    p.active[slot_b] = 1;
    p.t[slot_b] = rt;
    p.x[slot_b] = rx;
    p.y[slot_b] = ry;
    p.z[slot_b] = rz;
    p.species[slot_b] = d2_id;
    p.mass[slot_b] = mb;
    p.E[slot_b] = pb_out.t;
    p.px[slot_b] = pb_out.x;
    p.py[slot_b] = pb_out.y;
    p.pz[slot_b] = pb_out.z;
    p.cell_id[slot_b] = -1;
    p.collided[slot_b] = 4;

    return true;
}

#if TRANSPORT_HAS_CUDA

__global__ void drift_particles_kernel(ParticleSoAView p, double dt, GridConfig gcfg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    drift_single_particle(p, i, dt, gcfg);
}

__global__ void decay_resonances_kernel(ParticleSoAView p,
                                        int n_initial_slots,
                                        double dt,
                                        std::uint64_t seed,
                                        std::uint64_t decay_step_id,
                                        unsigned long long* d_decays) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_initial_slots) return;
    PhiloxRng rng(seed, decay_step_id, static_cast<std::uint64_t>(i) + 0x9E3779B9ULL);
    if (try_resonance_decay(p, i, dt, rng)) {
        atomicAdd(d_decays, 1ULL);
    }
}

__global__ void collide_disjoint_pairs_kernel(ParticleSoAView p,
                                              GridView g,
                                              CrossSectionConfig xs_cfg,
                                              ReactionConfig rxn_cfg,
                                              double sub_dt,
                                              int n_test,
                                              std::uint64_t seed,
                                              std::uint64_t step_counter,
                                              unsigned long long* d_stats,  // [5]: pairs, el, inel, form, overf
                                              double* d_max_prob) {
    __shared__ int sh_live_count;
    const int cid = blockIdx.x;
    if (cid >= g.cfg.num_cells()) return;

    const int raw_count = g.cell_counts[cid];
    if (raw_count < 2) return;

    int* cell_indices = g.sorted_indices + g.cell_start[cid];

    if (threadIdx.x == 0) {
        PhiloxRng shuf_rng(seed, step_counter, static_cast<std::uint64_t>(cid) * 2ULL);
        sh_live_count = compact_and_shuffle_cell_indices(p, cell_indices, raw_count, shuf_rng);
    }
    __syncthreads();

    const int count = sh_live_count;
    if (count < 2) return;

    const int num_disjoint = count / 2;
    const double total_pairs = 0.5 * static_cast<double>(count) * static_cast<double>(count - 1);
    const double pair_multiplicity_weight = total_pairs / static_cast<double>(num_disjoint);
    const double dV = g.cfg.cell_volume_fm3();
    const double rate_prefactor =
        pair_multiplicity_weight * sub_dt / (static_cast<double>(n_test) * dV);

    unsigned long long l_tested = 0, l_el = 0, l_inel = 0, l_form = 0;

    for (int k = threadIdx.x; k < num_disjoint; k += blockDim.x) {
        const int idx1 = cell_indices[2 * k];
        const int idx2 = cell_indices[2 * k + 1];

        const std::uint64_t pair_stream =
            (static_cast<std::uint64_t>(cid) << 24) ^ static_cast<std::uint64_t>(k + 1);
        PhiloxRng rng(seed, step_counter, pair_stream);

        ++l_tested;
        const int outcome = try_pair_reaction(
            p, idx1, idx2, rate_prefactor, g.cfg, xs_cfg, rxn_cfg,
            rng, &d_stats[4], d_max_prob);
        if (outcome == 1) ++l_el;
        else if (outcome == 2) ++l_inel;
        else if (outcome == 3) ++l_form;
    }

    if (l_tested > 0) atomicAdd(&d_stats[0], l_tested);
    if (l_el > 0)     atomicAdd(&d_stats[1], l_el);
    if (l_inel > 0)   atomicAdd(&d_stats[2], l_inel);
    if (l_form > 0)   atomicAdd(&d_stats[3], l_form);
}

__global__ void collide_all_pairs_kernel(ParticleSoAView p,
                                         GridView g,
                                         CrossSectionConfig xs_cfg,
                                         ReactionConfig rxn_cfg,
                                         double sub_dt,
                                         int n_test,
                                         std::uint64_t seed,
                                         std::uint64_t step_counter,
                                         unsigned long long* d_stats,
                                         double* d_max_prob) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= g.cfg.num_cells()) return;

    const int raw_count = g.cell_counts[cid];
    if (raw_count < 2) return;

    int* cell_indices = g.sorted_indices + g.cell_start[cid];
    PhiloxRng rng(seed, step_counter, static_cast<std::uint64_t>(cid));
    const int count = compact_and_shuffle_cell_indices(p, cell_indices, raw_count, rng);
    if (count < 2) return;

    for (int a = 0; a < count; ++a) {
        p.collided[cell_indices[a]] = 0;
    }

    const double dV = g.cfg.cell_volume_fm3();
    const double rate_prefactor = sub_dt / (static_cast<double>(n_test) * dV);

    unsigned long long l_tested = 0, l_el = 0, l_inel = 0, l_form = 0;

    for (int a = 0; a < count - 1; ++a) {
        const int idx1 = cell_indices[a];
        if (!p.active[idx1] || p.collided[idx1]) continue;
        for (int b = a + 1; b < count; ++b) {
            const int idx2 = cell_indices[b];
            if (!p.active[idx2] || p.collided[idx2]) continue;

            ++l_tested;
            const int outcome = try_pair_reaction(
                p, idx1, idx2, rate_prefactor, g.cfg, xs_cfg, rxn_cfg,
                rng, &d_stats[4], d_max_prob);
            if (outcome > 0) {
                if (outcome == 1) ++l_el;
                else if (outcome == 2) ++l_inel;
                else if (outcome == 3) ++l_form;
                break;
            }
        }
    }

    if (l_tested > 0) atomicAdd(&d_stats[0], l_tested);
    if (l_el > 0)     atomicAdd(&d_stats[1], l_el);
    if (l_inel > 0)   atomicAdd(&d_stats[2], l_inel);
    if (l_form > 0)   atomicAdd(&d_stats[3], l_form);
}

#endif  // TRANSPORT_HAS_CUDA

}  // namespace detail

class TransportStepper {
public:
    TransportStepper(const StepperConfig& step_cfg,
                     const CrossSectionConfig& xs_cfg,
                     ReactionConfig rxn_cfg = ReactionConfig{})
        : step_cfg_(step_cfg), xs_cfg_(xs_cfg), rxn_cfg_(rxn_cfg) {
        rxn_cfg_.norms = compute_spectral_norm_table();
        d_ull_stats_.resize(6);  // [pairs, el, inel, form, overf, decays]
        d_max_prob_.resize(1);
        reset_stats();
    }

    void reset_stats() {
        d_ull_stats_.zero();
        d_max_prob_.zero();
    }

    // Time-reversal symmetric Vlasov-BUU Velocity-Verlet + Strang-split collision step:
    //   Kick(dt/2) -> Drift(dt/2) -> Decay(dt/2) -> Collide(dt) -> Decay(dt/2) -> Drift(dt/2) -> Kick(dt/2)
    void step(ParticleSystem& particles, SpatialGrid& grid, MeanFieldGrid* mf = nullptr) {
        const double half_dt = 0.5 * step_cfg_.dt_fm;
        const bool use_mf = (mf != nullptr && mf->config().enabled);

        if (use_mf) {
            mf->rebuild_potentials(particles, step_cfg_.n_test);
            mf->kick(particles, half_dt);
            mf->drift(particles, half_dt);
        } else {
            drift(particles, grid.config(), half_dt);
        }

        if (rxn_cfg_.enable_resonance_2to1_and_decays) {
            decay_resonances(particles, half_dt, step_index_ * 2ULL);
        }
        grid.build(particles);
        collide(particles, grid);
        if (rxn_cfg_.enable_resonance_2to1_and_decays) {
            decay_resonances(particles, half_dt, step_index_ * 2ULL + 1ULL);
        }

        if (use_mf) {
            mf->rebuild_potentials(particles, step_cfg_.n_test);
            mf->drift(particles, half_dt);
            mf->rebuild_potentials(particles, step_cfg_.n_test);
            mf->kick(particles, half_dt);
        } else {
            drift(particles, grid.config(), half_dt);
        }

        particles.compact_if_fragmented();
        ++step_index_;
    }

    void drift(ParticleSystem& particles, const GridConfig& gcfg, double dt) {
        const int np = particles.slot_size();
        if (np == 0) return;
        ParticleSoAView pview = particles.view();
#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        const int blocks = (np + TPB - 1) / TPB;
        detail::drift_particles_kernel<<<blocks, TPB>>>(pview, dt, gcfg);
        CUDA_CHECK(cudaGetLastError());
#else
        for (int i = 0; i < np; ++i) {
            detail::drift_single_particle(pview, i, dt, gcfg);
        }
#endif
    }

    void decay_resonances(ParticleSystem& particles, double dt, std::uint64_t decay_id) {
        const int n_est = particles.slot_size();
        if (n_est == 0) return;
        particles.ensure_headroom(n_est + 256);
        const int n_cur = particles.slot_size();
        if (n_cur == 0) return;

        ParticleSoAView pview = particles.view();
        unsigned long long* d_decays = d_ull_stats_.data() + 5;

#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        const int blocks = (n_cur + TPB - 1) / TPB;
        detail::decay_resonances_kernel<<<blocks, TPB>>>(
            pview, n_cur, dt, step_cfg_.seed ^ 0xA5A5A5A5ULL, decay_id, d_decays);
        CUDA_CHECK(cudaGetLastError());
#else
        for (int i = 0; i < n_cur; ++i) {
            PhiloxRng rng(step_cfg_.seed ^ 0xA5A5A5A5ULL, decay_id,
                          static_cast<std::uint64_t>(i) + 0x9E3779B9ULL);
            if (detail::try_resonance_decay(pview, i, dt, rng)) {
                *d_decays += 1ULL;
            }
        }
#endif
    }

    void collide(ParticleSystem& particles, SpatialGrid& grid) {
        const int n_cells = grid.config().num_cells();
        if (particles.slot_size() < 2 || n_cells == 0) return;

        const int n_sub = (step_cfg_.collision_substeps > 0) ? step_cfg_.collision_substeps : 1;
        const double sub_dt = step_cfg_.dt_fm / static_cast<double>(n_sub);

        ParticleSoAView pview = particles.view();
        GridView gview = grid.view();

        unsigned long long* d_stats = d_ull_stats_.data();
        double* d_maxp = d_max_prob_.data();

        for (int sub = 0; sub < n_sub; ++sub) {
            const std::uint64_t sub_counter =
                step_index_ * static_cast<std::uint64_t>(n_sub) + static_cast<std::uint64_t>(sub);

#if TRANSPORT_HAS_CUDA
            if (step_cfg_.scheme == CollisionScheme::DisjointPairs) {
                constexpr int BLOCK_THREADS = 64;
                detail::collide_disjoint_pairs_kernel<<<n_cells, BLOCK_THREADS>>>(
                    pview, gview, xs_cfg_, rxn_cfg_, sub_dt, step_cfg_.n_test,
                    step_cfg_.seed, sub_counter, d_stats, d_maxp);
                CUDA_CHECK(cudaGetLastError());
            } else {
                constexpr int TPB = 128;
                const int blocks = (n_cells + TPB - 1) / TPB;
                detail::collide_all_pairs_kernel<<<blocks, TPB>>>(
                    pview, gview, xs_cfg_, rxn_cfg_, sub_dt, step_cfg_.n_test,
                    step_cfg_.seed, sub_counter, d_stats, d_maxp);
                CUDA_CHECK(cudaGetLastError());
            }
#else
            const double dV = gview.cfg.cell_volume_fm3();
            for (int cid = 0; cid < n_cells; ++cid) {
                const int raw_count = gview.cell_counts[cid];
                if (raw_count < 2) continue;
                int* cell_indices = gview.sorted_indices + gview.cell_start[cid];

                if (step_cfg_.scheme == CollisionScheme::DisjointPairs) {
                    PhiloxRng shuf_rng(step_cfg_.seed, sub_counter,
                                       static_cast<std::uint64_t>(cid) * 2ULL);
                    const int count = detail::compact_and_shuffle_cell_indices(
                        pview, cell_indices, raw_count, shuf_rng);
                    if (count < 2) continue;

                    const int num_disjoint = count / 2;
                    const double total_pairs =
                        0.5 * static_cast<double>(count) * static_cast<double>(count - 1);
                    const double weight = total_pairs / static_cast<double>(num_disjoint);
                    const double prefactor =
                        weight * sub_dt / (static_cast<double>(step_cfg_.n_test) * dV);

                    for (int k = 0; k < num_disjoint; ++k) {
                        const int idx1 = cell_indices[2 * k];
                        const int idx2 = cell_indices[2 * k + 1];
                        const std::uint64_t pair_stream =
                            (static_cast<std::uint64_t>(cid) << 24) ^
                            static_cast<std::uint64_t>(k + 1);
                        PhiloxRng rng(step_cfg_.seed, sub_counter, pair_stream);

                        d_stats[0] += 1ULL;
                        const int outcome = detail::try_pair_reaction(
                            pview, idx1, idx2, prefactor, gview.cfg, xs_cfg_, rxn_cfg_,
                            rng, &d_stats[4], d_maxp);
                        if (outcome == 1) d_stats[1] += 1ULL;
                        else if (outcome == 2) d_stats[2] += 1ULL;
                        else if (outcome == 3) d_stats[3] += 1ULL;
                    }
                } else {
                    PhiloxRng rng(step_cfg_.seed, sub_counter, static_cast<std::uint64_t>(cid));
                    const int count = detail::compact_and_shuffle_cell_indices(
                        pview, cell_indices, raw_count, rng);
                    if (count < 2) continue;
                    for (int a = 0; a < count; ++a) pview.collided[cell_indices[a]] = 0;

                    const double prefactor =
                        sub_dt / (static_cast<double>(step_cfg_.n_test) * dV);

                    for (int a = 0; a < count - 1; ++a) {
                        const int idx1 = cell_indices[a];
                        if (!pview.active[idx1] || pview.collided[idx1]) continue;
                        for (int b = a + 1; b < count; ++b) {
                            const int idx2 = cell_indices[b];
                            if (!pview.active[idx2] || pview.collided[idx2]) continue;
                            d_stats[0] += 1ULL;
                            const int outcome = detail::try_pair_reaction(
                                pview, idx1, idx2, prefactor, gview.cfg, xs_cfg_, rxn_cfg_,
                                rng, &d_stats[4], d_maxp);
                            if (outcome > 0) {
                                if (outcome == 1) d_stats[1] += 1ULL;
                                else if (outcome == 2) d_stats[2] += 1ULL;
                                else if (outcome == 3) d_stats[3] += 1ULL;
                                break;
                            }
                        }
                    }
                }
            }
#endif
        }
    }

    StepStats read_stats() const {
        StepStats st{};
        unsigned long long h_ull[6] = {};
        double h_maxp = 0.0;
        d_ull_stats_.copy_to_host(h_ull, 6);
        d_max_prob_.copy_to_host(&h_maxp, 1);
        st.pairs_tested = h_ull[0];
        st.collisions_elastic = h_ull[1];
        st.collisions_inelastic_2to2 = h_ull[2];
        st.formations_2to1 = h_ull[3];
        st.prob_overflows = h_ull[4];
        st.decays_1to2 = h_ull[5];
        st.max_prob_seen = h_maxp;
        return st;
    }

    std::uint64_t step_index() const { return step_index_; }
    const StepperConfig& stepper_config() const { return step_cfg_; }
    const CrossSectionConfig& xs_config() const { return xs_cfg_; }
    const ReactionConfig& reaction_config() const { return rxn_cfg_; }

private:
    StepperConfig step_cfg_;
    CrossSectionConfig xs_cfg_;
    ReactionConfig rxn_cfg_;
    std::uint64_t step_index_ = 0;
    DeviceBuffer<unsigned long long> d_ull_stats_;
    DeviceBuffer<double> d_max_prob_;
};

}  // namespace transport
