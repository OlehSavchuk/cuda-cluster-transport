#pragma once

#include <array>
#include <cmath>
#include "backend/cuda_compat.cuh"
#include "physics/species.cuh"
#include "sim/particles.cuh"

namespace transport {

struct GlobalObservables {
    int n_active = 0;
    double E_tot = 0.0;
    double Px_tot = 0.0;
    double Py_tot = 0.0;
    double Pz_tot = 0.0;
    long long baryon_tot = 0;
    long long charge_tot = 0;

    double Txx_sum = 0.0;
    double Tyy_sum = 0.0;
    double Tzz_sum = 0.0;
    double r_sq_sum = 0.0;

    std::array<int, NUM_SPECIES> counts{};
    std::array<double, NUM_SPECIES> kinetic_energy_sum{};
    std::array<double, NUM_SPECIES> mass_sum{};

    double pressure_anisotropy() const {
        const double p_perp = 0.5 * (Txx_sum + Tyy_sum);
        return (p_perp > 1.0e-15) ? (Tzz_sum / p_perp) : 0.0;
    }

    double rms_radius() const {
        return (n_active > 0) ? std::sqrt(r_sq_sum / static_cast<double>(n_active)) : 0.0;
    }

    double mean_mass(int sp_id) const {
        return (counts[sp_id] > 0)
                   ? (mass_sum[sp_id] / static_cast<double>(counts[sp_id]))
                   : get_species_info(sp_id).pole_mass_gev;
    }
};

namespace detail {

#if TRANSPORT_HAS_CUDA
__global__ void reduce_observables_kernel(ParticleSoAView p,
                                          double* d_scalars,          // [8]: E, Px, Py, Pz, Txx, Tyy, Tzz, r2
                                          unsigned long long* d_bq,   // [3]: B, Q, active_count
                                          int* d_counts,              // [NUM_SPECIES]
                                          double* d_ke_sum,           // [NUM_SPECIES]
                                          double* d_mass_sum) {       // [NUM_SPECIES]
    __shared__ double sh_sc[8];
    if (threadIdx.x < 8) sh_sc[threadIdx.x] = 0.0;
    __syncthreads();

    double lE = 0.0, lPx = 0.0, lPy = 0.0, lPz = 0.0;
    double lTxx = 0.0, lTyy = 0.0, lTzz = 0.0, lr2 = 0.0;
    unsigned long long lB = 0ULL, lQ = 0ULL, lAct = 0ULL;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += blockDim.x * gridDim.x) {
        if (!p.active[i]) continue;

        const double E = p.E[i];
        const double px = p.px[i];
        const double py = p.py[i];
        const double pz = p.pz[i];
        const double m = p.mass[i];
        const double inv_E = 1.0 / E;
        const int sp_id = p.species[i];
        const SpeciesInfo sp = get_species_info(sp_id);

        lE += E;
        lPx += px;
        lPy += py;
        lPz += pz;
        lTxx += px * px * inv_E;
        lTyy += py * py * inv_E;
        lTzz += pz * pz * inv_E;
        lr2 += p.x[i] * p.x[i] + p.y[i] * p.y[i] + p.z[i] * p.z[i];

        lB += static_cast<unsigned long long>(sp.baryon_num);
        lQ += static_cast<unsigned long long>(sp.charge);
        lAct += 1ULL;

        atomicAdd(&d_counts[sp_id], 1);
        portable_atomic_add_double(&d_ke_sum[sp_id], E - m);
        portable_atomic_add_double(&d_mass_sum[sp_id], m);
    }

    portable_atomic_add_double(&sh_sc[0], lE);
    portable_atomic_add_double(&sh_sc[1], lPx);
    portable_atomic_add_double(&sh_sc[2], lPy);
    portable_atomic_add_double(&sh_sc[3], lPz);
    portable_atomic_add_double(&sh_sc[4], lTxx);
    portable_atomic_add_double(&sh_sc[5], lTyy);
    portable_atomic_add_double(&sh_sc[6], lTzz);
    portable_atomic_add_double(&sh_sc[7], lr2);

    if (lB > 0) atomicAdd(&d_bq[0], lB);
    if (lQ > 0) atomicAdd(&d_bq[1], lQ);
    if (lAct > 0) atomicAdd(&d_bq[2], lAct);
    __syncthreads();

    if (threadIdx.x < 8) {
        portable_atomic_add_double(&d_scalars[threadIdx.x], sh_sc[threadIdx.x]);
    }
}
#endif

}  // namespace detail

class ObservableCalculator {
public:
    ObservableCalculator() {
        d_scalars_.resize(8);
        d_bq_.resize(3);
        d_counts_.resize(NUM_SPECIES);
        d_ke_sum_.resize(NUM_SPECIES);
        d_mass_sum_.resize(NUM_SPECIES);
    }

    GlobalObservables compute(ParticleSystem& particles) {
        GlobalObservables obs{};
        const int np = particles.slot_size();
        if (np == 0) return obs;

        ParticleSoAView pview = particles.view();

#if TRANSPORT_HAS_CUDA
        d_scalars_.zero();
        d_bq_.zero();
        d_counts_.zero();
        d_ke_sum_.zero();
        d_mass_sum_.zero();

        constexpr int TPB = 256;
        int blocks = (np + TPB - 1) / TPB;
        if (blocks > 512) blocks = 512;

        detail::reduce_observables_kernel<<<blocks, TPB>>>(
            pview, d_scalars_.data(), d_bq_.data(),
            d_counts_.data(), d_ke_sum_.data(), d_mass_sum_.data());
        CUDA_CHECK(cudaGetLastError());

        double h_sc[8] = {};
        unsigned long long h_bq[3] = {};
        d_scalars_.copy_to_host(h_sc, 8);
        d_bq_.copy_to_host(h_bq, 3);
        d_counts_.copy_to_host(obs.counts.data(), NUM_SPECIES);
        d_ke_sum_.copy_to_host(obs.kinetic_energy_sum.data(), NUM_SPECIES);
        d_mass_sum_.copy_to_host(obs.mass_sum.data(), NUM_SPECIES);

        obs.E_tot = h_sc[0];
        obs.Px_tot = h_sc[1];
        obs.Py_tot = h_sc[2];
        obs.Pz_tot = h_sc[3];
        obs.Txx_sum = h_sc[4];
        obs.Tyy_sum = h_sc[5];
        obs.Tzz_sum = h_sc[6];
        obs.r_sq_sum = h_sc[7];
        obs.baryon_tot = static_cast<long long>(h_bq[0]);
        obs.charge_tot = static_cast<long long>(h_bq[1]);
        obs.n_active = static_cast<int>(h_bq[2]);
#else
        for (int i = 0; i < np; ++i) {
            if (!pview.active[i]) continue;
            const double E = pview.E[i];
            const double px = pview.px[i];
            const double py = pview.py[i];
            const double pz = pview.pz[i];
            const double m = pview.mass[i];
            const double inv_E = 1.0 / E;
            const int sp_id = pview.species[i];
            const SpeciesInfo sp = get_species_info(sp_id);

            obs.n_active += 1;
            obs.E_tot += E;
            obs.Px_tot += px;
            obs.Py_tot += py;
            obs.Pz_tot += pz;
            obs.Txx_sum += px * px * inv_E;
            obs.Tyy_sum += py * py * inv_E;
            obs.Tzz_sum += pz * pz * inv_E;
            obs.r_sq_sum += pview.x[i] * pview.x[i] +
                            pview.y[i] * pview.y[i] +
                            pview.z[i] * pview.z[i];
            obs.baryon_tot += sp.baryon_num;
            obs.charge_tot += sp.charge;
            obs.counts[sp_id] += 1;
            obs.kinetic_energy_sum[sp_id] += (E - m);
            obs.mass_sum[sp_id] += m;
        }
#endif
        return obs;
    }

private:
    DeviceBuffer<double> d_scalars_;
    DeviceBuffer<unsigned long long> d_bq_;
    DeviceBuffer<int> d_counts_;
    DeviceBuffer<double> d_ke_sum_;
    DeviceBuffer<double> d_mass_sum_;
};

}  // namespace transport
