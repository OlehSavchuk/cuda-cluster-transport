#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/cross_sections.cuh"
#include "physics/kinematics.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/init_conditions.cuh"
#include "sim/observables.cuh"
#include "sim/particles.cuh"
#include "sim/stepper.cuh"

namespace {

using namespace transport;

#define TEST_ASSERT(cond, msg)                                                         \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg));     \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

void test_species_table() {
    int n_excited = 0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        TEST_ASSERT(sp.id == id, "Species ID mismatch");
        TEST_ASSERT(sp.baryon_num >= 1 && sp.baryon_num <= 4, "Baryon number out of range");
        TEST_ASSERT(sp.charge >= 0 && sp.charge <= 3, "Charge out of range");
        TEST_ASSERT(sp.degeneracy == sp.two_J + 1, "Spin degeneracy mismatch g != 2J + 1");
        TEST_ASSERT(sp.pole_mass_gev > 0.90 * sp.baryon_num, "Unphysical pole mass");
        TEST_ASSERT(sp.rms_radius_fm > 0.5, "Unphysical RMS radius");
        if (sp.is_excited) {
            ++n_excited;
            TEST_ASSERT(sp.width_gev > 0.0, "Excited state must have positive width");
        }
    }
    // 3 d* states + 17 excited 4He* states + 4H* + 4Li* = 22 excited states
    TEST_ASSERT(n_excited == 22, "Expected 22 excited A=2 and A=4 cluster states");
    std::printf("[PASS] test_species_table (28 species: 6 ground + 22 excited states)\n");
}

void test_relativistic_kinematics() {
    PhiloxRng rng(42ULL, 1ULL, 1ULL);

    // 1. Ultra-low energy threshold test: 4He*(20.21) + d*(0+) at keV relative energy
    //    boosted to relativistic lab frame (beta_z = 0.8)
    const double m1 = get_species_info(HE4_2021).pole_mass_gev;
    const double m2 = get_species_info(D_STAR_PN).pole_mass_gev;
    const double p_cm_true = 0.002;  // 2 MeV/c -> sub-keV kinetic energy for 4He*!
    const double E1_cm = std::sqrt(m1 * m1 + p_cm_true * p_cm_true);
    const double E2_cm = std::sqrt(m2 * m2 + p_cm_true * p_cm_true);

    const double bx = 0.25, by = -0.35, bz = 0.65;
    const double b2 = bx * bx + by * by + bz * bz;
    const double gamma = 1.0 / std::sqrt(1.0 - b2);

    const FourVector p1_lab = lorentz_boost_from_cm(m1, E1_cm, 0.0, 0.0,  p_cm_true, bx, by, bz, gamma);
    const FourVector p2_lab = lorentz_boost_from_cm(m2, E2_cm, 0.0, 0.0, -p_cm_true, bx, by, bz, gamma);

    const PairKinematics kin = compute_pair_kinematics(p1_lab, m1, p2_lab, m2);
    TEST_ASSERT(std::fabs(kin.p_cm - p_cm_true) / p_cm_true < 1.0e-11,
                "Cancellation-free p_cm failed near threshold under Lorentz boost");

    for (int trial = 0; trial < 200; ++trial) {
        const double cos_th = 1.0 - 2.0 * rng.uniform();
        const double phi = phys::TWO_PI * rng.uniform();
        FourVector out1{}, out2{};
        scatter_elastic_2to2(p1_lab, m1, p2_lab, m2, kin, cos_th, phi, out1, out2);

        const double dE = std::fabs((out1.t + out2.t) - (p1_lab.t + p2_lab.t));
        const double dPx = std::fabs((out1.x + out2.x) - (p1_lab.x + p2_lab.x));
        const double dPy = std::fabs((out1.y + out2.y) - (p1_lab.y + p2_lab.y));
        const double dPz = std::fabs((out1.z + out2.z) - (p1_lab.z + p2_lab.z));

        TEST_ASSERT(dE < 1.0e-13 && dPx < 1.0e-13 && dPy < 1.0e-13 && dPz < 1.0e-13,
                    "4-momentum conservation violated in elastic 2 -> 2 scattering");

        const double m1_check = std::sqrt(out1.t * out1.t - out1.x * out1.x - out1.y * out1.y - out1.z * out1.z);
        const double m2_check = std::sqrt(out2.t * out2.t - out2.x * out2.x - out2.y * out2.y - out2.z * out2.z);
        TEST_ASSERT(std::fabs(m1_check - m1) < 1.0e-12 && std::fabs(m2_check - m2) < 1.0e-12,
                    "Mass-shell condition violated after Lorentz boost");
    }
    std::printf("[PASS] test_relativistic_kinematics (sub-keV threshold + boosted 2->2 exact to 1e-13)\n");
}

void test_cross_section_hierarchy() {
    CrossSectionConfig cfg{};
    cfg.model = CrossSectionModel::ParametricGlauber;

    const double mp = phys::MASS_PROTON;
    const FourVector p_beam{std::sqrt(mp * mp + 0.45 * 0.45), 0.0, 0.0, 0.45};

    auto sig_with_proton = [&](int target_id) {
        const double mt = get_species_info(target_id).pole_mass_gev;
        const FourVector p_targ{mt, 0.0, 0.0, 0.0};
        const PairKinematics kin = compute_pair_kinematics(p_beam, mp, p_targ, mt);
        return elastic_cross_section_fm2(PROTON, target_id, kin, cfg) * phys::FM2_TO_MB;
    };

    const double sig_pp = sig_with_proton(PROTON);
    const double sig_pn = sig_with_proton(NEUTRON);
    const double sig_pd = sig_with_proton(DEUTERON);
    const double sig_pdstar = sig_with_proton(D_STAR_PN);
    const double sig_phe4 = sig_with_proton(ALPHA);
    const double sig_phe4_exc = sig_with_proton(HE4_2021);

    TEST_ASSERT(sig_pn > sig_pp, "Expected sigma_pn > sigma_pp in nuclear energy regime");
    TEST_ASSERT(sig_pd > sig_pp, "Expected sigma_pd > sigma_pp");
    TEST_ASSERT(sig_phe4 > sig_pd, "Expected sigma_p+4He > sigma_p+d");
    TEST_ASSERT(sig_pdstar > sig_pd,
                "Expected diffuse singlet d*(0+) to have larger Glauber cross section than ground d");
    TEST_ASSERT(sig_phe4_exc > sig_phe4,
                "Expected breathing-mode 4He*(20.21) to have larger cross section than ground 4He");

    std::printf("[PASS] test_cross_section_hierarchy (pp=%.1f, pn=%.1f, pd=%.1f, pd*=%.1f, p4He=%.1f, p4He*=%.1f mb)\n",
                sig_pp, sig_pn, sig_pd, sig_pdstar, sig_phe4, sig_phe4_exc);
}

void test_stochastic_boltzmann_rate() {
    // Analytic collision rate test in a single periodic cell (including odd particle count N_c = 501!)
    // Verifies exact agreement between simulated collision counts (both DisjointPairs & AllPairs)
    // and theoretical Boltzmann expectation sum_{i<j} v_moller(i,j) * sigma_0 * dt / (N_test * V).
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 1; gcfg.ny = 1; gcfg.nz = 1;
    gcfg.xmin = 0.0; gcfg.xmax = 10.0;
    gcfg.ymin = 0.0; gcfg.ymax = 10.0;
    gcfg.zmin = 0.0; gcfg.zmax = 10.0;

    MixtureConfig mix{};
    mix.include_ground_clusters = false;
    mix.include_excited_dstar = false;
    mix.include_excited_he4 = false;

    const int n_part = 251;  // Deliberately odd to test exact odd-N_c pair multiplicity weighting!
    auto host_p = init_box_ensemble(n_part, gcfg, 0.050, 0.050, 0.900, 0.0, mix, 777ULL);

    CrossSectionConfig xs_cfg{};
    xs_cfg.model = CrossSectionModel::ConstantIsotropic;
    xs_cfg.angular_mode = AngularMode::Isotropic;
    xs_cfg.sigma_const_mb = 30.0;
    const double sigma_fm2 = 30.0 * phys::MB_TO_FM2;

    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.05;
    step_cfg.n_test = 5;
    step_cfg.collision_substeps = 1;

    // Exact expected collisions in a single frozen-distribution ensemble over M independent trials
    double exact_rate_per_step = 0.0;
    for (int i = 0; i < n_part; ++i) {
        const FourVector pi{host_p[i].E, host_p[i].px, host_p[i].py, host_p[i].pz};
        for (int j = i + 1; j < n_part; ++j) {
            const FourVector pj{host_p[j].E, host_p[j].px, host_p[j].py, host_p[j].pz};
            const PairKinematics kin = compute_pair_kinematics(pi, host_p[i].mass, pj, host_p[j].mass);
            exact_rate_per_step += kin.v_moller * sigma_fm2 * step_cfg.dt_fm /
                                   (step_cfg.n_test * gcfg.cell_volume_fm3());
        }
    }

    for (CollisionScheme scheme : {CollisionScheme::DisjointPairs, CollisionScheme::AllPairs}) {
        step_cfg.scheme = scheme;
        TransportStepper stepper(step_cfg, xs_cfg);
        SpatialGrid grid(gcfg);
        ParticleSystem particles;

        constexpr int N_TRIALS = 3000;
        for (int t = 0; t < N_TRIALS; ++t) {
            particles.upload(host_p);
            grid.build(particles);
            stepper.step(particles, grid);
        }
        const StepStats st = stepper.read_stats();
        const double expected_total = exact_rate_per_step * N_TRIALS;
        const double observed_total = static_cast<double>(st.collisions_accepted);
        const double rel_diff = std::fabs(observed_total - expected_total) / expected_total;

        // With expected_total ~ 2800, 1/sqrt(N) ~ 1.9%; 4-sigma tolerance = 8%
        TEST_ASSERT(rel_diff < 0.08, "Stochastic Boltzmann collision rate deviated from analytic expectation");
        std::printf("[PASS] test_stochastic_boltzmann_rate [%s]: obs=%.1f vs exact=%.1f (diff=%.2f%%)\n",
                    (scheme == CollisionScheme::DisjointPairs) ? "DisjointPairs" : "AllPairs",
                    observed_total, expected_total, 100.0 * rel_diff);
    }
}

void test_multispecies_isotropization_and_conservation() {
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 6; gcfg.ny = 6; gcfg.nz = 6;
    gcfg.xmin = 0.0; gcfg.xmax = 12.0;
    gcfg.ymin = 0.0; gcfg.ymax = 12.0;
    gcfg.zmin = 0.0; gcfg.zmax = 12.0;

    MixtureConfig mix{};
    mix.cluster_yield_boost = 8.0;
    mix.excited_yield_boost = 20.0;

    const int n_part = 12000;
    auto host_p = init_box_ensemble(
        n_part, gcfg, /*T_perp=*/0.025, /*T_z=*/0.095, 0.860, -0.005, mix, 9999ULL);

    CrossSectionConfig xs_cfg{};
    xs_cfg.model = CrossSectionModel::ParametricGlauber;
    xs_cfg.angular_mode = AngularMode::Isotropic;

    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.25;
    step_cfg.n_test = 5;
    step_cfg.collision_substeps = 2;
    step_cfg.scheme = CollisionScheme::DisjointPairs;

    ParticleSystem particles;
    particles.upload(host_p);
    SpatialGrid grid(gcfg);
    TransportStepper stepper(step_cfg, xs_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    const double aniso0 = obs0.pressure_anisotropy();
    TEST_ASSERT(aniso0 > 2.5, "Initial anisotropic state should have Pz/P_perp > 2.5");

    for (int s = 0; s < 60; ++s) {
        stepper.step(particles, grid);
    }

    const GlobalObservables obs1 = obs_calc.compute(particles);
    const double aniso1 = obs1.pressure_anisotropy();
    const double rel_dE = std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot;

    TEST_ASSERT(rel_dE < 1.0e-12, "Total energy conservation violated in multi-species run");
    TEST_ASSERT(obs1.baryon_tot == obs0.baryon_tot, "Total baryon number changed");
    TEST_ASSERT(obs1.charge_tot == obs0.charge_tot, "Total electric charge changed");
    TEST_ASSERT(aniso1 < 1.35 && aniso1 < aniso0,
                "Multi-species elastic collisions failed to isotropize pressure tensor");

    std::printf("[PASS] test_multispecies_isotropization (Pz/P_perp: %.3f -> %.3f, |dE|/E = %.2e)\n",
                aniso0, aniso1, rel_dE);
}

void test_open_expanding_fireball() {
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 8; gcfg.ny = 8; gcfg.nz = 8;
    gcfg.xmin = -12.0; gcfg.xmax = 12.0;
    gcfg.ymin = -12.0; gcfg.ymax = 12.0;
    gcfg.zmin = -12.0; gcfg.zmax = 12.0;

    MixtureConfig mix{};
    mix.cluster_yield_boost = 5.0;
    mix.excited_yield_boost = 15.0;

    auto host_p = init_expanding_fireball(6000, 5.0, 0.35, 0.035, 0.860, 0.0, mix, 31415ULL);
    ParticleSystem particles;
    particles.upload(host_p);

    SpatialGrid grid(gcfg);
    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.30;
    step_cfg.n_test = 5;
    step_cfg.collision_substeps = 2;

    TransportStepper stepper(step_cfg, xs_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    for (int s = 0; s < 40; ++s) {
        stepper.step(particles, grid);
    }
    const GlobalObservables obs1 = obs_calc.compute(particles);
    const StepStats st = stepper.read_stats();

    TEST_ASSERT(obs1.rms_radius(6000) > obs0.rms_radius(6000) + 1.5,
                "Expanding fireball RMS radius did not expand as expected");
    TEST_ASSERT(std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot < 1.0e-12,
                "Energy conservation violated in expanding fireball");
    TEST_ASSERT(st.collisions_accepted > 100, "Expected elastic collisions during fireball expansion");

    std::printf("[PASS] test_open_expanding_fireball (R_rms: %.2f -> %.2f fm, %llu collisions)\n",
                obs0.rms_radius(6000), obs1.rms_radius(6000), st.collisions_accepted);
}

}  // namespace

int main() {
    std::printf("=== Running CUDA Cluster Transport Verification Suite ===\n");
    std::printf("Backend: %s\n", transport::backend_description().c_str());

    test_species_table();
    test_relativistic_kinematics();
    test_cross_section_hierarchy();
    test_stochastic_boltzmann_rate();
    test_multispecies_isotropization_and_conservation();
    test_open_expanding_fireball();

    std::printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
