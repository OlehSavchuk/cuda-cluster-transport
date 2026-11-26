#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/resonances.cuh"
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

// Fixed-mass Maxwell-Juttner density n(m; g, T, mu) [fm^-3]
inline double fixed_mass_mj_density_fm3(double m, int degeneracy, double T_gev, double mu_gev) {
    using namespace phys;
    const double fugacity_arg = (mu_gev - m) / T_gev;
    if (fugacity_arg < -500.0) return 0.0;

    constexpr int N_PTS = 96;
    constexpr double S_MAX = 7.5;
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
        static_cast<double>(degeneracy) / (2.0 * PI * PI * HBARC * HBARC * HBARC);
    return prefactor * T_gev * integral * std::exp(fugacity_arg);
}

// Grand-canonical equilibrium density n_i [fm^-3]:
// When use_breit_wigner == true, integrates A_R(m) * n_MJ(m, T, mu) over [m_min, m_max]
// so off-shell detailed balance holds strictly in grand-canonical box tests!
inline double equilibrium_density_fm3(int species_id,
                                      double T_gev,
                                      double mu_B_gev,
                                      double mu_Q_gev,
                                      bool use_breit_wigner = true,
                                      const SpectralNormTable* norms_ptr = nullptr) {
    const SpeciesInfo sp = get_species_info(species_id);
    const double mu = sp.baryon_num * mu_B_gev + sp.charge * mu_Q_gev;
    if (!sp.is_excited || !use_breit_wigner) {
        return fixed_mass_mj_density_fm3(sp.pole_mass_gev, sp.degeneracy, T_gev, mu);
    }

    const SpectralNormTable local_norms =
        (norms_ptr != nullptr) ? (*norms_ptr) : compute_spectral_norm_table();

    const double m_min = resonance_min_mass(species_id);
    const double m_max = resonance_max_mass(species_id);
    const double m0 = sp.pole_mass_gev;
    const double split = (m0 > m_min && m0 < m_max) ? m0 : 0.5 * (m_min + m_max);
    const double segs[3] = {m_min, split, m_max};

    constexpr double gx[4] = {
        0.183434642495649805, 0.525532409916328986,
        0.796666477413626740, 0.960289856497536232
    };
    constexpr double gw[4] = {
        0.362683783378361983, 0.313706645877887287,
        0.222381034453374471, 0.101228536290376259
    };

    double sum = 0.0;
    for (int s = 0; s < 2; ++s) {
        constexpr int N_SUB = 32;
        const double h = (segs[s + 1] - segs[s]) / static_cast<double>(N_SUB);
        for (int k = 0; k < N_SUB; ++k) {
            const double mid = segs[s] + (k + 0.5) * h;
            const double half = 0.5 * h;
            for (int q = 0; q < 4; ++q) {
                const double m1 = mid - half * gx[q];
                const double m2 = mid + half * gx[q];
                sum += half * gw[q] * (
                    spectral_function(species_id, m1, local_norms) *
                        fixed_mass_mj_density_fm3(m1, sp.degeneracy, T_gev, mu) +
                    spectral_function(species_id, m2, local_norms) *
                        fixed_mass_mj_density_fm3(m2, sp.degeneracy, T_gev, mu));
            }
        }
    }
    return sum;
}

// Sample initial thermal mass m_R of an excited resonance from A_R(m) * n_MJ(m, T)
inline double sample_thermal_breit_wigner_mass(int res_id, double T_gev, PhiloxRng& rng) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return sp.pole_mass_gev;

    const double m_min = resonance_min_mass(res_id);
    const double m_max = resonance_max_mass(res_id);
    const double m0 = sp.pole_mass_gev;
    const double w0 = m0 * sp.width_gev;
    const double th_min = std::atan((m_min * m_min - m0 * m0) / w0);
    const double th_max = std::atan((m_max * m_max - m0 * m0) / w0);

    const double n_ref = fixed_mass_mj_density_fm3(m_min, 1, T_gev, 0.0);
    constexpr double RATIO_BOUND = 8.5;

    for (int iter = 0; iter < 128; ++iter) {
        const double th = th_min + (th_max - th_min) * rng.uniform();
        const double m2 = m0 * m0 + w0 * std::tan(th);
        if (m2 <= m_min * m_min || m2 >= m_max * m_max) continue;
        const double m = std::sqrt(m2);
        const double g_tot = total_width_gev(res_id, m);
        if (g_tot <= 0.0) continue;

        const double dm2 = m2 - m0 * m0;
        const double mg = m * g_tot;
        const double bw_ratio = (mg / w0) * ((dm2 * dm2 + w0 * w0) / (dm2 * dm2 + mg * mg));
        const double mj_ratio = (n_ref > 0.0)
                                    ? (fixed_mass_mj_density_fm3(m, 1, T_gev, 0.0) / n_ref)
                                    : std::exp(-(m - m_min) / T_gev);
        if (rng.uniform() < (bw_ratio * mj_ratio) / RATIO_BOUND) {
            return m;
        }
    }
    return m0;
}

struct MixtureConfig {
    bool include_nucleons = true;
    bool include_ground_clusters = true;
    bool include_excited_dstar = true;
    bool include_excited_he4 = true;
    bool include_a56_cascade = false;
    bool include_pions = false;
    bool include_deltas = false;
    bool use_breit_wigner = true;
    double cluster_yield_boost = 1.0;
    double excited_yield_boost = 1.0;
};

inline std::vector<double> build_species_probabilities(double T_gev,
                                                       double mu_B_gev,
                                                       double mu_Q_gev,
                                                       const MixtureConfig& mix) {
    const SpectralNormTable norms = compute_spectral_norm_table();
    std::vector<double> weights(NUM_SPECIES, 0.0);
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (is_pion(id) && !mix.include_pions) continue;
        if (is_delta(id) && !mix.include_deltas) continue;
        if (sp.baryon_num == 1 && !sp.is_excited && !mix.include_nucleons) continue;
        if (sp.baryon_num >= 2 && !sp.is_excited && !mix.include_ground_clusters) continue;
        if (sp.baryon_num == 2 && sp.is_excited && !mix.include_excited_dstar) continue;
        if (sp.baryon_num >= 4 && sp.is_excited && !mix.include_excited_he4) continue;
        if (is_a56_excited(id) && !mix.include_a56_cascade) continue;

        double w = equilibrium_density_fm3(
            id, T_gev, mu_B_gev, mu_Q_gev, mix.use_breit_wigner, &norms);
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

// Compute exact grand-canonical total test-particle count N_test * V * sum_i n_i^eq(T, mu_B, mu_Q)
// Required whenever testing 2 <-> 1 chemical equilibrium so physical density matches mu_B!
inline int grand_canonical_test_particles(const GridConfig& grid,
                                          int n_test,
                                          double T_gev,
                                          double mu_B_gev,
                                          double mu_Q_gev,
                                          const MixtureConfig& mix) {
    const SpectralNormTable norms = compute_spectral_norm_table();
    double n_tot_fm3 = 0.0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (is_pion(id) && !mix.include_pions) continue;
        if (is_delta(id) && !mix.include_deltas) continue;
        if (sp.baryon_num == 1 && !sp.is_excited && !mix.include_nucleons) continue;
        if (sp.baryon_num >= 2 && !sp.is_excited && !mix.include_ground_clusters) continue;
        if (sp.baryon_num == 2 && sp.is_excited && !mix.include_excited_dstar) continue;
        if (sp.baryon_num >= 4 && sp.is_excited && !mix.include_excited_he4) continue;
        if (is_a56_excited(id) && !mix.include_a56_cascade) continue;

        double w = equilibrium_density_fm3(
            id, T_gev, mu_B_gev, mu_Q_gev, mix.use_breit_wigner, &norms);
        if (sp.baryon_num >= 2) w *= mix.cluster_yield_boost;
        if (sp.is_excited) w *= mix.excited_yield_boost;
        n_tot_fm3 += w;
    }
    const double vol = grid.lx() * grid.ly() * grid.lz();
    return static_cast<int>(std::llround(static_cast<double>(n_test) * vol * n_tot_fm3));
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
    const double T_eff = (2.0 * T_perp_gev + T_z_gev) / 3.0;
    const auto probs = build_species_probabilities(T_eff, mu_B_gev, mu_Q_gev, mix);
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
        const double m = (sp.is_excited && mix.use_breit_wigner)
                             ? sample_thermal_breit_wigner_mass(sp_id, T_eff, rng)
                             : sp.pole_mass_gev;

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

// 2. Expanding 3D heavy-ion hadronic fireball with radial Hubble flow beta_r = beta_s * tanh(r / R0)
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
        const double m = (sp.is_excited && mix.use_breit_wigner)
                             ? sample_thermal_breit_wigner_mass(sp_id, T_local_gev, rng)
                             : sp.pole_mass_gev;

        const double x = sigma_r * rng.normal();
        const double y = sigma_r * rng.normal();
        const double z = sigma_r * rng.normal();
        const double r = std::sqrt(x * x + y * y + z * z);

        const double beta_mag =
            (r > 1.0e-9) ? (beta_surface * std::tanh(r / radius_rms_fm)) : 0.0;
        const double bx = (r > 1.0e-9) ? (beta_mag * x / r) : 0.0;
        const double by = (r > 1.0e-9) ? (beta_mag * y / r) : 0.0;
        const double bz = (r > 1.0e-9) ? (beta_mag * z / r) : 0.0;
        const double gamma = 1.0 / std::sqrt(1.0 - beta_mag * beta_mag);

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
        const double m = (sp.is_excited && mix.use_breit_wigner)
                             ? sample_thermal_breit_wigner_mass(sp_id, T_chem, rng)
                             : sp.pole_mass_gev;

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

// 4. Symmetric Heavy-Ion Collision (e.g. FOPI Au+Au @ 250 AMeV) in equal-velocity NN CM frame
// Initializes two Woods-Saxon + local Thomas-Fermi nuclei (A_nuc, Z_nuc) boosted by +-beta_cm along z
// at impact parameter x = +-b/2 (reaction plane = x-z plane).
inline std::vector<HostParticle> init_heavy_ion_collision(int A_nuc,
                                                          int Z_nuc,
                                                          int n_test,
                                                          double elab_kin_amev,
                                                          double impact_b_fm,
                                                          std::uint64_t seed,
                                                          double& out_y_cm_beam) {
    using namespace phys;
    const double T_lab_gev = elab_kin_amev * MEV_TO_GEV;
    const double E_lab = MASS_NUCLEON_AVG + T_lab_gev;
    const double p_lab = std::sqrt(E_lab * E_lab - MASS_NUCLEON_AVG * MASS_NUCLEON_AVG);
    const double y_lab = 0.5 * std::log((E_lab + p_lab) / (E_lab - p_lab));
    out_y_cm_beam = 0.5 * y_lab;

    const double beta_cm = std::tanh(out_y_cm_beam);
    const double gamma_cm = std::cosh(out_y_cm_beam);

    const double R_ws = 1.12 * std::cbrt(static_cast<double>(A_nuc));
    const double a_ws = 0.54;
    const double r_max = R_ws + 8.0 * a_ws;
    const double z0 = R_ws + 2.5;  // Touching distance + 2.5 fm nuclear skin clearance

    const int n_per_nucleus = A_nuc * n_test;
    const int z_per_nucleus = Z_nuc * n_test;
    std::vector<HostParticle> out;
    out.reserve(2 * n_per_nucleus);

    for (int side = 0; side < 2; ++side) {
        std::vector<HostParticle> nuc(n_per_nucleus);
        double sx = 0.0, sy = 0.0, sz = 0.0;
        double spx = 0.0, spy = 0.0, spz = 0.0;

        for (int i = 0; i < n_per_nucleus; ++i) {
            PhiloxRng rng(seed, 400ULL + side, static_cast<std::uint64_t>(i));
            const bool is_p = (i < z_per_nucleus);
            const int sp_id = is_p ? PROTON : NEUTRON;
            const double m = get_species_info(sp_id).pole_mass_gev;

            // Sample Woods-Saxon radius r^2 / (1 + exp((r - R)/a))
            double r = 0.0, ws_f = 0.0;
            while (true) {
                r = r_max * rng.uniform();
                ws_f = 1.0 / (1.0 + std::exp((r - R_ws) / a_ws));
                if (rng.uniform() < (r * r / (r_max * r_max)) * ws_f * 2.4) break;
            }
            const double cth = 1.0 - 2.0 * rng.uniform();
            const double sth = std::sqrt((1.0 - cth * cth > 0.0) ? (1.0 - cth * cth) : 0.0);
            const double phi = TWO_PI * rng.uniform();
            const double rx = r * sth * std::cos(phi);
            const double ry = r * sth * std::sin(phi);
            const double rz = r * cth;

            // Local Thomas-Fermi momentum sphere p_F,tau(r) = hbarc * (3 pi^2 rho_tau(r))^(1/3)
            const double frac_iso = is_p ? (static_cast<double>(Z_nuc) / A_nuc)
                                         : (static_cast<double>(A_nuc - Z_nuc) / A_nuc);
            const double rho_tau = 0.168 * frac_iso * ws_f;
            const double pF = HBARC * std::cbrt(3.0 * PI * PI * rho_tau);
            const double pmag = pF * std::cbrt(rng.uniform());
            const double pcth = 1.0 - 2.0 * rng.uniform();
            const double psth = std::sqrt((1.0 - pcth * pcth > 0.0) ? (1.0 - pcth * pcth) : 0.0);
            const double pphi = TWO_PI * rng.uniform();

            nuc[i].x = rx; nuc[i].y = ry; nuc[i].z = rz;
            nuc[i].px = pmag * psth * std::cos(pphi);
            nuc[i].py = pmag * psth * std::sin(pphi);
            nuc[i].pz = pmag * pcth;
            nuc[i].mass = m;
            nuc[i].species = sp_id;
            sx += rx; sy += ry; sz += rz;
            spx += nuc[i].px; spy += nuc[i].py; spz += nuc[i].pz;
        }

        const double inv_n = 1.0 / static_cast<double>(n_per_nucleus);
        const double x_shift = (side == 0) ? (+0.5 * impact_b_fm) : (-0.5 * impact_b_fm);
        const double z_shift = (side == 0) ? (-z0) : (+z0);
        const double bz = (side == 0) ? (+beta_cm) : (-beta_cm);

        for (int i = 0; i < n_per_nucleus; ++i) {
            HostParticle& hp = nuc[i];
            hp.x = (hp.x - sx * inv_n) + x_shift;
            hp.y = (hp.y - sy * inv_n);
            // Lorentz-contracted longitudinal profile in CM frame!
            hp.z = (hp.z - sz * inv_n) / gamma_cm + z_shift;

            const double px0 = hp.px - spx * inv_n;
            const double py0 = hp.py - spy * inv_n;
            const double pz0 = hp.pz - spz * inv_n;
            const double E0 = std::sqrt(hp.mass * hp.mass + px0 * px0 + py0 * py0 + pz0 * pz0);
            const FourVector plab = lorentz_boost_from_cm(hp.mass, E0, px0, py0, pz0, 0.0, 0.0, bz, gamma_cm);
            hp.t = 0.0;
            hp.E = plab.t; hp.px = plab.x; hp.py = plab.y; hp.pz = plab.z;
            out.push_back(hp);
        }
    }

    // Remove tiny O(1/sqrt(N)) Fermi-energy boost asymmetry so total 3-momentum P_cm == 0 to machine precision
    double tot_px = 0.0, tot_py = 0.0, tot_pz = 0.0;
    for (const auto& hp : out) { tot_px += hp.px; tot_py += hp.py; tot_pz += hp.pz; }
    const double inv_tot = 1.0 / static_cast<double>(out.size());
    for (auto& hp : out) {
        hp.px -= tot_px * inv_tot;
        hp.py -= tot_py * inv_tot;
        hp.pz -= tot_pz * inv_tot;
        hp.E = std::sqrt(hp.mass * hp.mass + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
    }
    return out;
}

}  // namespace transport
