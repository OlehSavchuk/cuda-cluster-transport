#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/particles.cuh"

namespace transport {

// Exact relativistic Maxwell-Juttner 4-momentum sampler (~90% acceptance across all m/T)
// Samples f(p) d^3p ~ exp(-sqrt(m^2 + p^2) / T) d^3p using a 4-Gamma mixture envelope.
HD_INLINE FourVector sample_maxwell_juttner(double mass, double T_gev, PhiloxRng& rng) {
    using namespace phys;
    if (T_gev <= 1.0e-12) {
        return FourVector{mass, 0.0, 0.0, 0.0};
    }

    const double sqrt_pi = 1.7724538509055160273;
    const double rt_2m = ::sqrt(2.0 * mass);
    const double t15 = T_gev * ::sqrt(T_gev);
    const double t20 = T_gev * T_gev;
    const double t25 = t20 * ::sqrt(T_gev);
    const double t30 = t20 * T_gev;

    const double w0 = mass * rt_2m * (0.5 * sqrt_pi) * t15;
    const double w1 = mass * t20;
    const double w2 = rt_2m * (0.75 * sqrt_pi) * t25;
    const double w3 = 2.0 * t30;
    const double w_sum = w0 + w1 + w2 + w3;

    double K = 0.0;
    while (true) {
        const double sel = w_sum * rng.uniform();
        double y = 0.0;
        if (sel < w0) {
            const double z = rng.normal();
            y = 0.5 * z * z - ::log(rng.uniform());
        } else if (sel < w0 + w1) {
            y = -::log(rng.uniform()) - ::log(rng.uniform());
        } else if (sel < w0 + w1 + w2) {
            const double z = rng.normal();
            y = 0.5 * z * z - ::log(rng.uniform()) - ::log(rng.uniform());
        } else {
            y = -::log(rng.uniform()) - ::log(rng.uniform()) - ::log(rng.uniform());
        }
        K = T_gev * y;
        const double denom = rt_2m * ::sqrt(K) + K;
        if (denom <= 0.0) continue;
        const double accept = ::sqrt(K * (K + 2.0 * mass)) / denom;
        if (rng.uniform() < accept) break;
    }

    const double E = mass + K;
    const double p_mag = ::sqrt(K * (K + 2.0 * mass));
    const double cos_th = 1.0 - 2.0 * rng.uniform();
    const double sin_th = ::sqrt(
        (1.0 - cos_th * cos_th > 0.0) ? (1.0 - cos_th * cos_th) : 0.0);
    const double phi = TWO_PI * rng.uniform();

    return FourVector{
        E,
        p_mag * sin_th * ::cos(phi),
        p_mag * sin_th * ::sin(phi),
        p_mag * cos_th
    };
}

// Compute grand-canonical Maxwell-Juttner equilibrium density n_i [fm^-3]
// via 32-point Gauss-Laguerre-like quadrature on u = K / T in [0, inf).
inline double equilibrium_density_fm3(int species_id,
                                      double T_gev,
                                      double mu_B_gev,
                                      double mu_Q_gev) {
    using namespace phys;
    const SpeciesInfo sp = get_species_info(species_id);
    const double m = sp.pole_mass_gev;
    const double mu = sp.baryon_num * mu_B_gev + sp.charge * mu_Q_gev;
    const double fugacity_arg = (mu - m) / T_gev;
    if (fugacity_arg < -500.0) return 0.0;

    // Integrate int_0^inf p * E * exp(-K/T) dK = T^2 int_0^inf sqrt(T u (T u + 2m)) (m + T u) e^{-u} du
    // Substitute u = s^2 (du = 2 s ds) on s in [0, 8] with midpoint rule (128 points -> <1e-9 accuracy)
    constexpr int N_PTS = 128;
    constexpr double S_MAX = 8.0;
    const double ds = S_MAX / static_cast<double>(N_PTS);
    double integral = 0.0;
    for (int j = 0; j < N_PTS; ++j) {
        const double s = (static_cast<double>(j) + 0.5) * ds;
        const double u = s * s;
        const double K = T_gev * u;
        const double E = m + K;
        const double p = std::sqrt(K * (K + 2.0 * m));
        integral += (2.0 * s) * p * E * std::exp(-u) * ds;
    }

    const double prefactor =
        static_cast<double>(sp.degeneracy) / (2.0 * PI * PI * HBARC * HBARC * HBARC);
    return prefactor * T_gev * integral * std::exp(fugacity_arg);
}

struct MixtureConfig {
    bool include_nucleons = true;
    bool include_ground_clusters = true;
    bool include_excited_dstar = true;
    bool include_excited_he4 = true;
    // Optional statistical enhancement of clusters/excited states for finite-statistics tests
    // (Set both to 1.0 for pure grand-canonical SHM abundances)
    double cluster_yield_boost = 1.0;
    double excited_yield_boost = 1.0;
};

inline std::vector<double> build_species_probabilities(double T_gev,
                                                       double mu_B_gev,
                                                       double mu_Q_gev,
                                                       const MixtureConfig& mix) {
    std::vector<double> weights(NUM_SPECIES, 0.0);
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (sp.baryon_num == 1 && !mix.include_nucleons) continue;
        if (sp.baryon_num >= 2 && !sp.is_excited && !mix.include_ground_clusters) continue;
        if (sp.baryon_num == 2 && sp.is_excited && !mix.include_excited_dstar) continue;
        if (sp.baryon_num == 4 && sp.is_excited && !mix.include_excited_he4) continue;

        double w = equilibrium_density_fm3(id, T_gev, mu_B_gev, mu_Q_gev);
        if (sp.baryon_num >= 2) w *= mix.cluster_yield_boost;
        if (sp.is_excited) w *= mix.excited_yield_boost;
        weights[id] = w;
    }

    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sum <= 0.0) {
        weights[PROTON] = 0.5;
        weights[NEUTRON] = 0.5;
        sum = 1.0;
    }
    for (double& w : weights) w /= sum;
    return weights;
}

inline int sample_species_from_cdf(const std::vector<double>& cdf, PhiloxRng& rng) {
    const double u = rng.uniform();
    for (int id = 0; id < static_cast<int>(cdf.size()); ++id) {
        if (u <= cdf[id]) return id;
    }
    return static_cast<int>(cdf.size()) - 1;
}

// 1. Periodic 3D box in thermal or momentum-anisotropic (T_perp vs T_z) initial state
inline std::vector<HostParticle> init_box_ensemble(int n_particles,
                                                   const GridConfig& grid,
                                                   double T_perp_gev,
                                                   double T_z_gev,
                                                   double mu_B_gev,
                                                   double mu_Q_gev,
                                                   const MixtureConfig& mix,
                                                   std::uint64_t seed) {
    const auto probs = build_species_probabilities(
        (2.0 * T_perp_gev + T_z_gev) / 3.0, mu_B_gev, mu_Q_gev, mix);
    std::vector<double> cdf(NUM_SPECIES, 0.0);
    double acc = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
        acc += probs[i];
        cdf[i] = acc;
    }
    cdf.back() = 1.0;

    std::vector<HostParticle> out(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        PhiloxRng rng(seed, 100ULL, static_cast<std::uint64_t>(i));
        const int sp_id = sample_species_from_cdf(cdf, rng);
        const SpeciesInfo sp = get_species_info(sp_id);
        const double m = sp.pole_mass_gev;

        HostParticle hp;
        hp.t = 0.0;
        hp.x = grid.xmin + grid.lx() * rng.uniform();
        hp.y = grid.ymin + grid.ly() * rng.uniform();
        hp.z = grid.zmin + grid.lz() * rng.uniform();
        hp.species = sp_id;
        hp.mass = m;

        if (std::fabs(T_perp_gev - T_z_gev) < 1.0e-12) {
            const FourVector p4 = sample_maxwell_juttner(m, T_perp_gev, rng);
            hp.E = p4.t;
            hp.px = p4.x;
            hp.py = p4.y;
            hp.pz = p4.z;
        } else {
            // Anisotropic momentum initialization: draw transverse components at T_perp
            // and longitudinal component at T_z, then place strictly on mass shell.
            const FourVector pt = sample_maxwell_juttner(m, T_perp_gev, rng);
            const FourVector pl = sample_maxwell_juttner(m, T_z_gev, rng);
            hp.px = pt.x;
            hp.py = pt.y;
            hp.pz = pl.z;
            hp.E = std::sqrt(m * m + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
        }
        out[i] = hp;
    }
    return out;
}

// 2. Expanding 3D heavy-ion hadronic fireball with radial Hubble flow beta_r = beta_s * (r / R0)
inline std::vector<HostParticle> init_expanding_fireball(int n_particles,
                                                         double radius_rms_fm,
                                                         double beta_surface,
                                                         double T_local_gev,
                                                         double mu_B_gev,
                                                         double mu_Q_gev,
                                                         const MixtureConfig& mix,
                                                         std::uint64_t seed) {
    const auto probs = build_species_probabilities(T_local_gev, mu_B_gev, mu_Q_gev, mix);
    std::vector<double> cdf(NUM_SPECIES, 0.0);
    double acc = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
        acc += probs[i];
        cdf[i] = acc;
    }
    cdf.back() = 1.0;

    const double sigma_r = radius_rms_fm / std::sqrt(3.0);
    std::vector<HostParticle> out(n_particles);

    for (int i = 0; i < n_particles; ++i) {
        PhiloxRng rng(seed, 200ULL, static_cast<std::uint64_t>(i));
        const int sp_id = sample_species_from_cdf(cdf, rng);
        const SpeciesInfo sp = get_species_info(sp_id);
        const double m = sp.pole_mass_gev;

        const double x = sigma_r * rng.normal();
        const double y = sigma_r * rng.normal();
        const double z = sigma_r * rng.normal();
        const double r = std::sqrt(x * x + y * y + z * z);

        // Smoothly saturating radial flow profile beta(r) = beta_s * tanh(r / R0)
        const double beta_mag =
            (r > 1.0e-9) ? (beta_surface * std::tanh(r / radius_rms_fm)) : 0.0;
        const double bx = (r > 1.0e-9) ? (beta_mag * x / r) : 0.0;
        const double by = (r > 1.0e-9) ? (beta_mag * y / r) : 0.0;
        const double bz = (r > 1.0e-9) ? (beta_mag * z / r) : 0.0;
        const double gamma = 1.0 / std::sqrt(1.0 - beta_mag * beta_mag);

        // Local thermal 4-momentum boosted by collective radial flow velocity +beta
        const FourVector p_loc = sample_maxwell_juttner(m, T_local_gev, rng);
        const FourVector p_lab = lorentz_boost_from_cm(
            m, p_loc.t, p_loc.x, p_loc.y, p_loc.z, bx, by, bz, gamma);

        HostParticle hp;
        hp.t = 0.0;
        hp.x = x;
        hp.y = y;
        hp.z = z;
        hp.E = p_lab.t;
        hp.px = p_lab.x;
        hp.py = p_lab.y;
        hp.pz = p_lab.z;
        hp.mass = m;
        hp.species = sp_id;
        out[i] = hp;
    }
    return out;
}

// 3. Counter-streaming nuclear & excited-cluster slabs along +-x for visual shock/scattering tests
inline std::vector<HostParticle> init_colliding_slabs(int n_particles,
                                                      const GridConfig& grid,
                                                      double beta_beam,
                                                      double T_internal_gev,
                                                      double mu_B_gev,
                                                      double mu_Q_gev,
                                                      const MixtureConfig& mix,
                                                      std::uint64_t seed) {
    // Use freeze-out chemical temperature (~55 MeV) for species abundances while keeping
    // internal beam kinetic spread cold (T_internal_gev) so the two beams are well-collimated.
    const double T_chem = (T_internal_gev < 0.055) ? 0.055 : T_internal_gev;
    const auto probs = build_species_probabilities(T_chem, mu_B_gev, mu_Q_gev, mix);
    std::vector<double> cdf(NUM_SPECIES, 0.0);
    double acc = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
        acc += probs[i];
        cdf[i] = acc;
    }
    cdf.back() = 1.0;

    const double gamma = 1.0 / std::sqrt(1.0 - beta_beam * beta_beam);
    const double cx = 0.5 * (grid.xmin + grid.xmax);
    const double cy = 0.5 * (grid.ymin + grid.ymax);
    const double cz = 0.5 * (grid.zmin + grid.zmax);
    const double Lx = grid.lx();
    const double sig_perp = 0.16 * grid.ly();
    const double slab_half_thick = 0.065 * Lx;

    std::vector<HostParticle> out(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        PhiloxRng rng(seed, 300ULL, static_cast<std::uint64_t>(i));
        const int sp_id = sample_species_from_cdf(cdf, rng);
        const SpeciesInfo sp = get_species_info(sp_id);
        const double m = sp.pole_mass_gev;

        const bool left_slab = (i % 2 == 0);
        const double x_center = left_slab ? (cx - 0.30 * Lx) : (cx + 0.30 * Lx);
        const double bx = left_slab ? (+beta_beam) : (-beta_beam);

        double x = x_center + slab_half_thick * (2.0 * rng.uniform() - 1.0);
        double y = cy + sig_perp * rng.normal();
        double z = cz + sig_perp * rng.normal();
        x = detail::wrap_periodic(x, grid.xmin, grid.lx());
        y = detail::wrap_periodic(y, grid.ymin, grid.ly());
        z = detail::wrap_periodic(z, grid.zmin, grid.lz());

        const FourVector p_loc = sample_maxwell_juttner(m, T_internal_gev, rng);
        const FourVector p_lab = lorentz_boost_from_cm(
            m, p_loc.t, p_loc.x, p_loc.y, p_loc.z, bx, 0.0, 0.0, gamma);

        HostParticle hp;
        hp.t = 0.0;
        hp.x = x;
        hp.y = y;
        hp.z = z;
        hp.E = p_lab.t;
        hp.px = p_lab.x;
        hp.py = p_lab.y;
        hp.pz = p_lab.z;
        hp.mass = m;
        hp.species = sp_id;
        out[i] = hp;
    }
    return out;
}

}  // namespace transport
