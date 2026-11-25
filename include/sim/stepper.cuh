#pragma once

#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/cross_sections.cuh"
#include "physics/kinematics.cuh"
#include "sim/grid.cuh"
#include "sim/particles.cuh"

namespace transport {

enum class CollisionScheme : int {
    DisjointPairs = 0,  // Bird/BAMPS random disjoint pairs: O(N_p), lock-free parallel per cell
    AllPairs = 1        // Full cell pair traversal with single-scatter-per-substep guard
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
    unsigned long long collisions_accepted = 0;
    unsigned long long prob_overflows = 0;
    double max_prob_seen = 0.0;
};

namespace detail {

HD_INLINE void drift_single_particle(ParticleSoAView& p, int i, double dt, const GridConfig& gcfg) {
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

// Fisher-Yates shuffle of indices[0 .. count-1]
HD_INLINE void shuffle_cell_indices(int* indices, int count, PhiloxRng& rng) {
    for (int k = count - 1; k > 0; --k) {
        const int j = static_cast<int>(rng.uniform_int(static_cast<std::uint32_t>(k + 1)));
        const int tmp = indices[k];
        indices[k] = indices[j];
        indices[j] = tmp;
    }
}

// Evaluate and optionally scatter a candidate particle pair (idx1, idx2)
HD_INLINE bool try_elastic_pair(ParticleSoAView& p,
                                int idx1,
                                int idx2,
                                double rate_prefactor,
                                const CrossSectionConfig& xs_cfg,
                                PhiloxRng& rng,
                                unsigned long long* d_overflows,
                                double* d_max_prob) {
    const double m1 = p.mass[idx1];
    const double m2 = p.mass[idx2];
    const FourVector p1_in{p.E[idx1], p.px[idx1], p.py[idx1], p.pz[idx1]};
    const FourVector p2_in{p.E[idx2], p.px[idx2], p.py[idx2], p.pz[idx2]};

    const PairKinematics kin = compute_pair_kinematics(p1_in, m1, p2_in, m2);
    if (kin.p_cm <= 1.0e-12 || kin.v_moller <= 1.0e-15) {
        return false;
    }

    const int id1 = p.species[idx1];
    const int id2 = p.species[idx2];
    const double sigma_fm2 = elastic_cross_section_fm2(id1, id2, kin, xs_cfg);

    const double prob = kin.v_moller * sigma_fm2 * rate_prefactor;
    portable_atomic_max_double(d_max_prob, prob);
    if (prob > 1.0) {
        portable_atomic_add_ull(d_overflows, 1ULL);
    }

    if (rng.uniform() >= prob) {
        return false;
    }

    double cos_theta = 0.0, phi = 0.0;
    sample_elastic_angles(id1, id2, kin, xs_cfg, rng, cos_theta, phi);

    FourVector p1_out{}, p2_out{};
    scatter_elastic_2to2(p1_in, m1, p2_in, m2, kin, cos_theta, phi, p1_out, p2_out);

    p.E[idx1] = p1_out.t;
    p.px[idx1] = p1_out.x;
    p.py[idx1] = p1_out.y;
    p.pz[idx1] = p1_out.z;

    p.E[idx2] = p2_out.t;
    p.px[idx2] = p2_out.x;
    p.py[idx2] = p2_out.y;
    p.pz[idx2] = p2_out.z;

    p.collided[idx1] = 1;
    p.collided[idx2] = 1;

    return true;
}

#if TRANSPORT_HAS_CUDA

__global__ void drift_particles_kernel(ParticleSoAView p, double dt, GridConfig gcfg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    drift_single_particle(p, i, dt, gcfg);
}

// 1 CUDA block per spatial cell:
// - Thread 0 performs Fisher-Yates shuffle on the cell's slice of sorted_indices in L1/L2
// - All threads in the block evaluate disjoint pairs (2*k, 2*k+1) in parallel with zero locks!
__global__ void collide_disjoint_pairs_kernel(ParticleSoAView p,
                                              GridView g,
                                              CrossSectionConfig xs_cfg,
                                              double sub_dt,
                                              int n_test,
                                              std::uint64_t seed,
                                              std::uint64_t step_counter,
                                              unsigned long long* d_pairs_tested,
                                              unsigned long long* d_collisions,
                                              unsigned long long* d_overflows,
                                              double* d_max_prob) {
    const int cid = blockIdx.x;
    if (cid >= g.cfg.num_cells()) return;

    const int count = g.cell_counts[cid];
    if (count < 2) return;

    int* cell_indices = g.sorted_indices + g.cell_start[cid];

    if (threadIdx.x == 0) {
        PhiloxRng shuf_rng(seed, step_counter, static_cast<std::uint64_t>(cid) * 2ULL);
        shuffle_cell_indices(cell_indices, count, shuf_rng);
    }
    __syncthreads();

    const int num_disjoint = count / 2;
    const double total_pairs = 0.5 * static_cast<double>(count) * static_cast<double>(count - 1);
    const double pair_multiplicity_weight = total_pairs / static_cast<double>(num_disjoint);
    const double dV = g.cfg.cell_volume_fm3();
    const double rate_prefactor =
        pair_multiplicity_weight * sub_dt / (static_cast<double>(n_test) * dV);

    unsigned long long local_tested = 0;
    unsigned long long local_accepted = 0;

    for (int k = threadIdx.x; k < num_disjoint; k += blockDim.x) {
        const int idx1 = cell_indices[2 * k];
        const int idx2 = cell_indices[2 * k + 1];

        const std::uint64_t pair_stream =
            (static_cast<std::uint64_t>(cid) << 24) ^ static_cast<std::uint64_t>(k + 1);
        PhiloxRng rng(seed, step_counter, pair_stream);

        ++local_tested;
        if (try_elastic_pair(p, idx1, idx2, rate_prefactor, xs_cfg, rng,
                             d_overflows, d_max_prob)) {
            ++local_accepted;
        }
    }

    if (local_tested > 0) portable_atomic_add_ull(d_pairs_tested, local_tested);
    if (local_accepted > 0) portable_atomic_add_ull(d_collisions, local_accepted);
}

// All-pairs cell kernel: shuffles cell order, then traverses all N_c*(N_c-1)/2 pairs
// enforcing at-most-one scatter per particle per substep.
__global__ void collide_all_pairs_kernel(ParticleSoAView p,
                                         GridView g,
                                         CrossSectionConfig xs_cfg,
                                         double sub_dt,
                                         int n_test,
                                         std::uint64_t seed,
                                         std::uint64_t step_counter,
                                         unsigned long long* d_pairs_tested,
                                         unsigned long long* d_collisions,
                                         unsigned long long* d_overflows,
                                         double* d_max_prob) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= g.cfg.num_cells()) return;

    const int count = g.cell_counts[cid];
    if (count < 2) return;

    int* cell_indices = g.sorted_indices + g.cell_start[cid];
    PhiloxRng rng(seed, step_counter, static_cast<std::uint64_t>(cid));
    shuffle_cell_indices(cell_indices, count, rng);

    for (int a = 0; a < count; ++a) {
        p.collided[cell_indices[a]] = 0;
    }

    const double dV = g.cfg.cell_volume_fm3();
    const double rate_prefactor = sub_dt / (static_cast<double>(n_test) * dV);

    unsigned long long local_tested = 0;
    unsigned long long local_accepted = 0;

    for (int a = 0; a < count - 1; ++a) {
        const int idx1 = cell_indices[a];
        if (p.collided[idx1]) continue;
        for (int b = a + 1; b < count; ++b) {
            const int idx2 = cell_indices[b];
            if (p.collided[idx2]) continue;

            ++local_tested;
            if (try_elastic_pair(p, idx1, idx2, rate_prefactor, xs_cfg, rng,
                                 d_overflows, d_max_prob)) {
                p.collided[idx1] = 1;
                p.collided[idx2] = 1;
                ++local_accepted;
                break;
            }
        }
    }

    if (local_tested > 0) portable_atomic_add_ull(d_pairs_tested, local_tested);
    if (local_accepted > 0) portable_atomic_add_ull(d_collisions, local_accepted);
}

#endif  // TRANSPORT_HAS_CUDA

}  // namespace detail

class TransportStepper {
public:
    TransportStepper(const StepperConfig& step_cfg, const CrossSectionConfig& xs_cfg)
        : step_cfg_(step_cfg), xs_cfg_(xs_cfg) {
        d_ull_stats_.resize(3);
        d_max_prob_.resize(1);
        reset_stats();
    }

    void reset_stats() {
        d_ull_stats_.zero();
        d_max_prob_.zero();
    }

    // Advance the transport system by one Strang-split step:
    //   Drift(dt/2) -> Spatial Cell Binning -> Stochastic 2->2 Collisions(dt) -> Drift(dt/2)
    void step(ParticleSystem& particles, SpatialGrid& grid) {
        const double half_dt = 0.5 * step_cfg_.dt_fm;
        drift(particles, grid.config(), half_dt);
        grid.build(particles);
        collide(particles, grid);
        drift(particles, grid.config(), half_dt);
        ++step_index_;
    }

    void drift(ParticleSystem& particles, const GridConfig& gcfg, double dt) {
        const int np = particles.size();
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

    void collide(ParticleSystem& particles, SpatialGrid& grid) {
        const int n_cells = grid.config().num_cells();
        if (particles.size() < 2 || n_cells == 0) return;

        const int n_sub = (step_cfg_.collision_substeps > 0) ? step_cfg_.collision_substeps : 1;
        const double sub_dt = step_cfg_.dt_fm / static_cast<double>(n_sub);

        ParticleSoAView pview = particles.view();
        GridView gview = grid.view();

        unsigned long long* d_pairs = d_ull_stats_.data() + 0;
        unsigned long long* d_colls = d_ull_stats_.data() + 1;
        unsigned long long* d_overf = d_ull_stats_.data() + 2;
        double* d_maxp = d_max_prob_.data();

        for (int sub = 0; sub < n_sub; ++sub) {
            const std::uint64_t sub_counter =
                step_index_ * static_cast<std::uint64_t>(n_sub) + static_cast<std::uint64_t>(sub);

#if TRANSPORT_HAS_CUDA
            if (step_cfg_.scheme == CollisionScheme::DisjointPairs) {
                constexpr int BLOCK_THREADS = 64;
                detail::collide_disjoint_pairs_kernel<<<n_cells, BLOCK_THREADS>>>(
                    pview, gview, xs_cfg_, sub_dt, step_cfg_.n_test,
                    step_cfg_.seed, sub_counter, d_pairs, d_colls, d_overf, d_maxp);
                CUDA_CHECK(cudaGetLastError());
            } else {
                constexpr int TPB = 128;
                const int blocks = (n_cells + TPB - 1) / TPB;
                detail::collide_all_pairs_kernel<<<blocks, TPB>>>(
                    pview, gview, xs_cfg_, sub_dt, step_cfg_.n_test,
                    step_cfg_.seed, sub_counter, d_pairs, d_colls, d_overf, d_maxp);
                CUDA_CHECK(cudaGetLastError());
            }
#else
            const double dV = gview.cfg.cell_volume_fm3();
            for (int cid = 0; cid < n_cells; ++cid) {
                const int count = gview.cell_counts[cid];
                if (count < 2) continue;
                int* cell_indices = gview.sorted_indices + gview.cell_start[cid];

                if (step_cfg_.scheme == CollisionScheme::DisjointPairs) {
                    PhiloxRng shuf_rng(step_cfg_.seed, sub_counter,
                                       static_cast<std::uint64_t>(cid) * 2ULL);
                    detail::shuffle_cell_indices(cell_indices, count, shuf_rng);

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

                        *d_pairs += 1ULL;
                        if (detail::try_elastic_pair(pview, idx1, idx2, prefactor, xs_cfg_,
                                                     rng, d_overf, d_maxp)) {
                            *d_colls += 1ULL;
                        }
                    }
                } else {
                    PhiloxRng rng(step_cfg_.seed, sub_counter, static_cast<std::uint64_t>(cid));
                    detail::shuffle_cell_indices(cell_indices, count, rng);
                    for (int a = 0; a < count; ++a) pview.collided[cell_indices[a]] = 0;

                    const double prefactor =
                        sub_dt / (static_cast<double>(step_cfg_.n_test) * dV);

                    for (int a = 0; a < count - 1; ++a) {
                        const int idx1 = cell_indices[a];
                        if (pview.collided[idx1]) continue;
                        for (int b = a + 1; b < count; ++b) {
                            const int idx2 = cell_indices[b];
                            if (pview.collided[idx2]) continue;
                            *d_pairs += 1ULL;
                            if (detail::try_elastic_pair(pview, idx1, idx2, prefactor, xs_cfg_,
                                                         rng, d_overf, d_maxp)) {
                                pview.collided[idx1] = 1;
                                pview.collided[idx2] = 1;
                                *d_colls += 1ULL;
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
        unsigned long long h_ull[3] = {0, 0, 0};
        double h_maxp = 0.0;
        d_ull_stats_.copy_to_host(h_ull, 3);
        d_max_prob_.copy_to_host(&h_maxp, 1);
        st.pairs_tested = h_ull[0];
        st.collisions_accepted = h_ull[1];
        st.prob_overflows = h_ull[2];
        st.max_prob_seen = h_maxp;
        return st;
    }

    std::uint64_t step_index() const { return step_index_; }
    const StepperConfig& stepper_config() const { return step_cfg_; }
    const CrossSectionConfig& xs_config() const { return xs_cfg_; }

private:
    StepperConfig step_cfg_;
    CrossSectionConfig xs_cfg_;
    std::uint64_t step_index_ = 0;
    DeviceBuffer<unsigned long long> d_ull_stats_;
    DeviceBuffer<double> d_max_prob_;
};

}  // namespace transport
