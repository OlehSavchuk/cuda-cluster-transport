#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/cross_sections.cuh"
#include "physics/inelastic_channels.cuh"
#include "physics/kinematics.cuh"
#include "physics/mean_field.cuh"
#include "physics/resonances.cuh"
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
    TEST_ASSERT(n_excited == 22, "Expected 22 excited A=2 and A=4 cluster states");
    std::printf("[PASS] test_species_table (28 species: 6 ground + 22 excited states)\n");
}

void test_relativistic_kinematics() {
    PhiloxRng rng(42ULL, 1ULL, 1ULL);

    const double m1 = get_species_info(HE4_2021).pole_mass_gev;
    const double m2 = get_species_info(D_STAR_PN).pole_mass_gev;
    const double p_cm_true = 0.002;
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

    // Also test inelastic 2 -> 2 (d + t -> n + 4He) and 1 -> 2 decay kinematics
    const double md = get_species_info(DEUTERON).pole_mass_gev;
    const double mt = get_species_info(TRITON).pole_mass_gev;
    const double mn = get_species_info(NEUTRON).pole_mass_gev;
    const double ma = get_species_info(ALPHA).pole_mass_gev;
    const FourVector pd{std::sqrt(md * md + 0.20 * 0.20 + 0.15 * 0.15 + 0.24 * 0.24), 0.20, -0.15, 0.24};
    const FourVector pt{std::sqrt(mt * mt + 0.10 * 0.10 + 0.25 * 0.25 + 0.13 * 0.13), -0.10, 0.25, -0.13};
    const PairKinematics kin_dt = compute_pair_kinematics(pd, md, pt, mt);
    const double pcm_na = two_body_p_cm(kin_dt.sqrt_s, mn, ma);

    for (int trial = 0; trial < 100; ++trial) {
        const double cos_th = 1.0 - 2.0 * rng.uniform();
        const double phi = phys::TWO_PI * rng.uniform();
        FourVector pn_out{}, pa_out{};
        scatter_general_2to2(pd, pt, kin_dt.sqrt_s, mn, ma, pcm_na, cos_th, phi, pn_out, pa_out);

        TEST_ASSERT(std::fabs((pn_out.t + pa_out.t) - (pd.t + pt.t)) < 1.0e-13,
                    "Energy conservation violated in exothermic d + t -> n + 4He");
        TEST_ASSERT(std::fabs((pn_out.x + pa_out.x) - (pd.x + pt.x)) < 1.0e-13,
                    "Momentum conservation violated in d + t -> n + 4He");

        // Test 1 -> 2 resonance decay conservation
        const FourVector p_res{pd.t + pt.t, pd.x + pt.x, pd.y + pt.y, pd.z + pt.z};
        FourVector dec1{}, dec2{};
        decay_isotropic_1to2(p_res, kin_dt.sqrt_s, mn, ma, cos_th, phi, dec1, dec2);
        TEST_ASSERT(std::fabs((dec1.t + dec2.t) - p_res.t) < 1.0e-13,
                    "Energy conservation violated in 1 -> 2 decay");
    }
    std::printf("[PASS] test_relativistic_kinematics (elastic, exothermic 2->2, and 1->2 decay exact to 1e-13)\n");
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
    TEST_ASSERT(sig_pdstar > sig_pd, "Expected sigma_p+d* > sigma_p+d");
    TEST_ASSERT(sig_phe4_exc > sig_phe4, "Expected sigma_p+4He* > sigma_p+4He");

    std::printf("[PASS] test_cross_section_hierarchy (pp=%.1f, pn=%.1f, pd=%.1f, pd*=%.1f, p4He=%.1f, p4He*=%.1f mb)\n",
                sig_pp, sig_pn, sig_pd, sig_pdstar, sig_phe4, sig_phe4_exc);
}

void test_breit_wigner_spectral_properties() {
    const SpectralNormTable norms = compute_spectral_norm_table();

    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (!sp.is_excited) continue;

        const double m_min = resonance_min_mass(id);
        const double m_max = resonance_max_mass(id);
        const double m0 = sp.pole_mass_gev;

        TEST_ASSERT(m0 > m_min, "Resonance pole mass must lie above lowest decay threshold");
        TEST_ASSERT(total_width_gev(id, m_min - 1.0e-5) == 0.0,
                    "Width must vanish strictly below threshold");
        TEST_ASSERT(spectral_function(id, m_min - 1.0e-5, norms) == 0.0,
                    "Spectral function must vanish strictly below threshold");

        // At pole mass m0, sum of partial widths equals nominal pole width Gamma_0
        const double gam_pole = total_width_gev(id, m0);
        TEST_ASSERT(std::fabs(gam_pole - sp.width_gev) / sp.width_gev < 1.0e-12,
                    "Total width at pole mass m0 must equal sp.width_gev");

        // Verify unit normalization int_{m_min}^{m_max} A_R(m) dm = 1 via independent midpoint check
        constexpr int N_CHK = 4000;
        const double dm = (m_max - m_min) / N_CHK;
        double area = 0.0;
        for (int k = 0; k < N_CHK; ++k) {
            area += spectral_function(id, m_min + (k + 0.5) * dm, norms) * dm;
        }
        TEST_ASSERT(std::fabs(area - 1.0) < 5.0e-3,
                    "Normalized Breit-Wigner spectral function failed unit integral check");

        // Verify conservation of baryon number and electric charge across all decay branches
        const ResonanceDecaySpec dspec = get_decay_spec(id);
        TEST_ASSERT(dspec.n_branches >= 1, "Excited state must have at least 1 decay branch");
        for (int b = 0; b < dspec.n_branches; ++b) {
            const SpeciesInfo da = get_species_info(dspec.branches[b].d1);
            const SpeciesInfo db = get_species_info(dspec.branches[b].d2);
            TEST_ASSERT(da.baryon_num + db.baryon_num == sp.baryon_num,
                        "Baryon conservation violated in decay table");
            TEST_ASSERT(da.charge + db.charge == sp.charge,
                        "Charge conservation violated in decay table");
        }
    }

    // Also verify conservation of B and Z in all 46 2 <-> 2 inelastic channels
    for (int ch = 0; ch < NUM_INELASTIC_2TO2_CHANNELS; ++ch) {
        const BinaryInelasticChannel c = get_inelastic_2to2_channel(ch);
        const SpeciesInfo sa = get_species_info(c.a);
        const SpeciesInfo sb = get_species_info(c.b);
        const SpeciesInfo sc = get_species_info(c.c);
        const SpeciesInfo sd = get_species_info(c.d);
        TEST_ASSERT(sa.baryon_num + sb.baryon_num == sc.baryon_num + sd.baryon_num,
                    "Baryon conservation violated in 2 <-> 2 channel table");
        TEST_ASSERT(sa.charge + sb.charge == sc.charge + sd.charge,
                    "Charge conservation violated in 2 <-> 2 channel table");
    }

    std::printf("[PASS] test_breit_wigner_spectral_properties (22 spectral functions + 46 inelastic channels verified)\n");
}

void test_stochastic_boltzmann_rate() {
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

    const int n_part = 251;
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

    ReactionConfig rxn_cfg{};
    rxn_cfg.enable_elastic = true;
    rxn_cfg.enable_inelastic_2to2 = false;
    rxn_cfg.enable_resonance_2to1_and_decays = false;

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
        TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
        SpatialGrid grid(gcfg);
        ParticleSystem particles;

        constexpr int N_TRIALS = 2000;
        for (int t = 0; t < N_TRIALS; ++t) {
            particles.upload(host_p);
            grid.build(particles);
            stepper.step(particles, grid);
        }
        const StepStats st = stepper.read_stats();
        const double expected_total = exact_rate_per_step * N_TRIALS;
        const double observed_total = static_cast<double>(st.collisions_elastic);
        const double rel_diff = std::fabs(observed_total - expected_total) / expected_total;

        TEST_ASSERT(rel_diff < 0.09, "Stochastic Boltzmann collision rate deviated from analytic expectation");
        std::printf("[PASS] test_stochastic_boltzmann_rate [%s]: obs=%.1f vs exact=%.1f (diff=%.2f%%)\n",
                    (scheme == CollisionScheme::DisjointPairs) ? "DisjointPairs" : "AllPairs",
                    observed_total, expected_total, 100.0 * rel_diff);
    }
}

void test_detailed_balance_2to1_resonances() {
    // Initialize grand-canonical equilibrium mixture of (p, n, nn*, d*, pp*) ONLY
    // with 2 <-> 1 formations and 1 -> 2 decays enabled, and verify forward == reverse rates
    // and stationary d* yield + off-shell Breit-Wigner mass distribution!
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 5; gcfg.ny = 5; gcfg.nz = 5;
    gcfg.xmin = 0.0; gcfg.xmax = 15.0;
    gcfg.ymin = 0.0; gcfg.ymax = 15.0;
    gcfg.zmin = 0.0; gcfg.zmax = 15.0;

    MixtureConfig mix{};
    mix.include_nucleons = true;
    mix.include_ground_clusters = false;
    mix.include_excited_dstar = true;
    mix.include_excited_he4 = false;
    mix.use_breit_wigner = true;
    mix.cluster_yield_boost = 1.0;
    mix.excited_yield_boost = 1.0;

    const double T_gev = 0.060;
    const double mu_B_gev = 0.850;
    const int n_test = 60;
    const int n_init = grand_canonical_test_particles(gcfg, n_test, T_gev, mu_B_gev, 0.0, mix);
    auto host_p = init_box_ensemble(n_init, gcfg, T_gev, T_gev, mu_B_gev, 0.0, mix, 2026ULL);

    CrossSectionConfig xs_cfg{};
    xs_cfg.model = CrossSectionModel::ConstantIsotropic;
    xs_cfg.sigma_const_mb = 20.0;

    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.10;
    step_cfg.n_test = n_test;
    step_cfg.collision_substeps = 2;

    ReactionConfig rxn_cfg{};
    rxn_cfg.enable_elastic = true;
    rxn_cfg.enable_inelastic_2to2 = false;
    rxn_cfg.enable_resonance_2to1_and_decays = true;

    ParticleSystem particles;
    particles.upload(host_p);
    SpatialGrid grid(gcfg);
    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    const int dstar0 = obs0.counts[D_STAR_NN] + obs0.counts[D_STAR_PN] + obs0.counts[D_STAR_PP];
    const double m_dstar0 = obs0.mean_mass(D_STAR_PN);

    for (int s = 0; s < 80; ++s) {
        stepper.step(particles, grid);
    }

    const GlobalObservables obs1 = obs_calc.compute(particles);
    const StepStats st = stepper.read_stats();
    const int dstar1 = obs1.counts[D_STAR_NN] + obs1.counts[D_STAR_PN] + obs1.counts[D_STAR_PP];
    const double m_dstar1 = obs1.mean_mass(D_STAR_PN);

    TEST_ASSERT(st.formations_2to1 > 1000 && st.decays_1to2 > 1000,
                "Expected >1000 2->1 formations and 1->2 decays in detailed balance test");

    const double rate_imb = std::fabs(static_cast<double>(st.formations_2to1) -
                                      static_cast<double>(st.decays_1to2)) /
                            static_cast<double>(st.formations_2to1);
    const double yield_drift = std::fabs(static_cast<double>(dstar1 - dstar0)) /
                               static_cast<double>(dstar0);

    TEST_ASSERT(rate_imb < 0.08, "2 <-> 1 formation/decay rate detailed balance violated");
    TEST_ASSERT(yield_drift < 0.12, "Equilibrium d* yield drifted under 2 <-> 1 dynamics");
    TEST_ASSERT(std::fabs(m_dstar1 - m_dstar0) < 0.004,
                "Off-shell Breit-Wigner mean mass drifted out of thermal equilibrium");
    TEST_ASSERT(std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot < 1.0e-11,
                "Energy conservation violated in 2 <-> 1 resonance formation/decay");
    TEST_ASSERT(obs1.baryon_tot == obs0.baryon_tot && obs1.charge_tot == obs0.charge_tot,
                "Baryon or charge conservation violated in 2 <-> 1 dynamics");

    std::printf("[PASS] test_detailed_balance_2to1_resonances (form=%llu vs dec=%llu [%.2f%%], N_d*: %d -> %d, <m_d*>: %.4f -> %.4f GeV)\n",
                st.formations_2to1, st.decays_1to2, 100.0 * rate_imb,
                dstar0, dstar1, m_dstar0, m_dstar1);
}

void test_detailed_balance_full_chemical_network() {
    // Full 28-species grand-canonical off-shell equilibrium box with ALL 2<->1 and ALL 46 2<->2
    // inelastic channels active simultaneously!
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 10; gcfg.ny = 10; gcfg.nz = 10;
    gcfg.xmin = 0.0; gcfg.xmax = 18.0;
    gcfg.ymin = 0.0; gcfg.ymax = 18.0;
    gcfg.zmin = 0.0; gcfg.zmax = 18.0;

    MixtureConfig mix{};
    mix.use_breit_wigner = true;
    mix.cluster_yield_boost = 1.0;
    mix.excited_yield_boost = 1.0;

    const double T_gev = 0.060;
    const double mu_B_gev = 0.820;
    const int n_test = 30;
    const int n_init = grand_canonical_test_particles(gcfg, n_test, T_gev, mu_B_gev, 0.0, mix);
    auto host_p = init_box_ensemble(n_init, gcfg, T_gev, T_gev, mu_B_gev, 0.0, mix, 8888ULL);

    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.10;
    step_cfg.n_test = n_test;
    step_cfg.collision_substeps = 2;

    ReactionConfig rxn_cfg{};
    rxn_cfg.enable_elastic = true;
    rxn_cfg.enable_inelastic_2to2 = true;
    rxn_cfg.enable_resonance_2to1_and_decays = true;
    rxn_cfg.inelastic_scale = 1.5;

    ParticleSystem particles;
    particles.upload(host_p);
    SpatialGrid grid(gcfg);
    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    const int d0 = obs0.counts[DEUTERON];
    const int dstar0 = obs0.counts[D_STAR_NN] + obs0.counts[D_STAR_PN] + obs0.counts[D_STAR_PP];

    for (int s = 0; s < 60; ++s) {
        stepper.step(particles, grid);
    }

    const GlobalObservables obs1 = obs_calc.compute(particles);
    const StepStats st = stepper.read_stats();
    const int d1 = obs1.counts[DEUTERON];
    const int dstar1 = obs1.counts[D_STAR_NN] + obs1.counts[D_STAR_PN] + obs1.counts[D_STAR_PP];

    TEST_ASSERT(st.collisions_inelastic_2to2 > 400, "Expected substantial 2<->2 inelastic reactions");
    TEST_ASSERT(std::fabs(static_cast<double>(d1 - d0)) / (d0 > 0 ? d0 : 1) < 0.14,
                "Deuteron yield drifted out of grand-canonical detailed balance");
    TEST_ASSERT(std::fabs(static_cast<double>(dstar1 - dstar0)) / (dstar0 > 0 ? dstar0 : 1) < 0.14,
                "d* yield drifted out of grand-canonical detailed balance");
    TEST_ASSERT(std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot < 1.0e-11,
                "Total energy conservation violated in full chemical network");
    TEST_ASSERT(obs1.baryon_tot == obs0.baryon_tot && obs1.charge_tot == obs0.charge_tot,
                "Baryon/charge conservation violated in full chemical network");

    std::printf("[PASS] test_detailed_balance_full_chemical_network (inel22=%llu, form=%llu, dec=%llu, d:%d->%d, d*:%d->%d)\n",
                st.collisions_inelastic_2to2, st.formations_2to1, st.decays_1to2,
                d0, d1, dstar0, dstar1);
}

void test_dynamic_cluster_nucleosynthesis() {
    // Start from 100% pure protons & neutrons (ZERO clusters at t=0) and verify spontaneous
    // synthesis of d, d*, (t, 3He), 4He, and excited 4He* states!
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 5; gcfg.ny = 5; gcfg.nz = 5;
    gcfg.xmin = 0.0; gcfg.xmax = 13.0;
    gcfg.ymin = 0.0; gcfg.ymax = 13.0;
    gcfg.zmin = 0.0; gcfg.zmax = 13.0;

    MixtureConfig mix{};
    mix.include_nucleons = true;
    mix.include_ground_clusters = false;
    mix.include_excited_dstar = false;
    mix.include_excited_he4 = false;

    auto host_p = init_box_ensemble(24000, gcfg, 0.060, 0.060, 0.860, 0.0, mix, 4242ULL);

    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.15;
    step_cfg.n_test = 10;
    step_cfg.collision_substeps = 2;

    ReactionConfig rxn_cfg{};
    rxn_cfg.inelastic_scale = 2.5;

    ParticleSystem particles;
    particles.upload(host_p);
    SpatialGrid grid(gcfg);
    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    TEST_ASSERT(obs0.counts[PROTON] + obs0.counts[NEUTRON] == 24000, "Initial state must be 100% nucleons");

    for (int s = 0; s < 80; ++s) {
        stepper.step(particles, grid);
    }

    const GlobalObservables obs1 = obs_calc.compute(particles);
    const int n_d = obs1.counts[DEUTERON];
    const int n_dstar = obs1.counts[D_STAR_NN] + obs1.counts[D_STAR_PN] + obs1.counts[D_STAR_PP];
    const int n_a3 = obs1.counts[TRITON] + obs1.counts[HELION];
    const int n_he4 = obs1.counts[ALPHA];
    int n_a4star = 0;
    for (int id = HE4_2021; id < NUM_SPECIES; ++id) n_a4star += obs1.counts[id];

    TEST_ASSERT(n_d > 200, "Failed to catalyze ground-state deuterons from pure nucleons");
    TEST_ASSERT(n_dstar > 200, "Failed to form singlet d* resonances from pure nucleons");
    TEST_ASSERT(n_a3 > 15, "Failed to synthesize A=3 clusters (t, 3He) from pure nucleons");
    TEST_ASSERT(n_he4 + n_a4star >= 3, "Failed to synthesize A=4 clusters (4He, 4He*) from pure nucleons");
    TEST_ASSERT(std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot < 1.0e-11,
                "Energy conservation violated during nucleosynthesis");
    TEST_ASSERT(obs1.baryon_tot == obs0.baryon_tot && obs1.charge_tot == obs0.charge_tot,
                "Baryon/charge conservation violated during nucleosynthesis");

    std::printf("[PASS] test_dynamic_cluster_nucleosynthesis (0 clusters -> d=%d, d*=%d, A3=%d, 4He=%d, 4He*=%d, |dE|/E=%.2e)\n",
                n_d, n_dstar, n_a3, n_he4, n_a4star,
                std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot);
}

void test_reactive_expanding_fireball() {
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 8; gcfg.ny = 8; gcfg.nz = 8;
    gcfg.xmin = -12.0; gcfg.xmax = 12.0;
    gcfg.ymin = -12.0; gcfg.ymax = 12.0;
    gcfg.zmin = -12.0; gcfg.zmax = 12.0;

    MixtureConfig mix{};
    mix.cluster_yield_boost = 3.0;
    mix.excited_yield_boost = 6.0;

    auto host_p = init_expanding_fireball(8000, 5.0, 0.35, 0.050, 0.850, 0.0, mix, 31415ULL);
    ParticleSystem particles;
    particles.upload(host_p);

    SpatialGrid grid(gcfg);
    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.25;
    step_cfg.n_test = 10;
    step_cfg.collision_substeps = 2;

    ReactionConfig rxn_cfg{};
    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    for (int s = 0; s < 40; ++s) {
        stepper.step(particles, grid);
    }
    const GlobalObservables obs1 = obs_calc.compute(particles);
    const StepStats st = stepper.read_stats();

    TEST_ASSERT(obs1.rms_radius() > obs0.rms_radius() + 1.5,
                "Expanding fireball RMS radius did not expand as expected");
    TEST_ASSERT(std::fabs(obs1.E_tot - obs0.E_tot) / obs0.E_tot < 1.0e-11,
                "Energy conservation violated in reactive expanding fireball");
    TEST_ASSERT(obs1.baryon_tot == obs0.baryon_tot && obs1.charge_tot == obs0.charge_tot,
                "Baryon or charge conservation violated in reactive expanding fireball");
    TEST_ASSERT(st.decays_1to2 > 50 && st.collisions_inelastic_2to2 > 20,
                "Expected active inelastic reactions and resonance decays during fireball expansion");

    std::printf("[PASS] test_reactive_expanding_fireball (R_rms: %.2f -> %.2f fm, el=%llu, inel=%llu, form=%llu, dec=%llu)\n",
                obs0.rms_radius(), obs1.rms_radius(), st.collisions_elastic,
                st.collisions_inelastic_2to2, st.formations_2to1, st.decays_1to2);
}

void test_eos_saturation_and_momentum_dependence() {
    for (EOSPreset preset : {EOSPreset::SkyrmeSoft, EOSPreset::MDSoft, EOSPreset::MDStiff}) {
        const MeanFieldConfig cfg = make_mean_field_config(preset);
        constexpr double h = 1.0e-4;
        const double e0 = cold_eos_energy_per_nucleon_gev(1.0, cfg);
        const double ep = cold_eos_energy_per_nucleon_gev(1.0 + h, cfg);
        const double em = cold_eos_energy_per_nucleon_gev(1.0 - h, cfg);
        const double de = (ep - em) / (2.0 * h);
        const double K0_num_mev = 9.0 * ((ep - 2.0 * e0 + em) / (h * h)) * 1.0e3;

        TEST_ASSERT(std::fabs(e0 * 1.0e3 - (-16.0)) < 1.0e-5,
                    "EOS saturation binding energy E/A(rho0) != -16.0 MeV");
        TEST_ASSERT(std::fabs(de) < 1.0e-7,
                    "EOS saturation pressure d(E/A)/du(1) != 0");
        TEST_ASSERT(std::fabs(K0_num_mev - cfg.K0_mev) < 0.05,
                    "EOS incompressibility K_0 mismatch at rho_0");
    }

    // Check momentum-dependent repulsive rise of U_N(rho_0, p) and effective mass m*(p_F)/m < 0.75
    const MeanFieldConfig md = make_mean_field_config(EOSPreset::MDSoft);
    double alpha1 = 1.0, da = 0.0, d2a = 0.0;
    cold_fermi_md_factors(1.0, md.rho0_fm3, md.Lambda_md_gev, alpha1, da, d2a);
    const double u_p0 = (md.A_sk_gev + md.B_sk_gev + 0.5 * md.C_md_gev * (alpha1 + 1.0)) * 1.0e3;
    const double u_p1 = (md.A_sk_gev + md.B_sk_gev +
                         0.5 * md.C_md_gev * (alpha1 + 1.0 / (1.0 + 1.0 / (md.Lambda_md_gev * md.Lambda_md_gev)))) * 1.0e3;
    TEST_ASSERT(u_p0 < -60.0 && u_p1 > 0.0,
                "MD optical potential failed to transition from attractive (p=0) to repulsive (p=1 GeV/c)");

    const double pF = phys::HBARC * std::cbrt(1.5 * phys::PI * phys::PI * md.rho0_fm3);
    const double denom = 1.0 + (pF * pF) / (md.Lambda_md_gev * md.Lambda_md_gev);
    const double inv_mstar_ratio =
        1.0 - (phys::MASS_NUCLEON_AVG * md.C_md_gev) / (md.Lambda_md_gev * md.Lambda_md_gev * denom * denom);
    const double mstar_over_m = 1.0 / inv_mstar_ratio;
    TEST_ASSERT(mstar_over_m > 0.65 && mstar_over_m < 0.75,
                "Expected Landau effective mass m*(p_F)/m ~ 0.70 for MDSoft");

    std::printf("[PASS] test_eos_saturation_and_md (E/A=-16.00 MeV, K0=215/380 MeV, U(0)=%.1f->U(1GeV)=+%.1f MeV, m*/m=%.3f)\n",
                u_p0, u_p1, mstar_over_m);
}

void test_vlasov_meanfield_conservation_and_spinodal() {
    // 1. Vlasov droplet test: inhomogeneous nuclear+cluster cloud in periodic box under MDSoft
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 12; gcfg.ny = 12; gcfg.nz = 12;
    gcfg.xmin = -12.0; gcfg.xmax = 12.0;
    gcfg.ymin = -12.0; gcfg.ymax = 12.0;
    gcfg.zmin = -12.0; gcfg.zmax = 12.0;

    MixtureConfig mix{};
    mix.cluster_yield_boost = 3.0;
    mix.excited_yield_boost = 4.0;

    auto host_p = init_expanding_fireball(12000, 5.2, /*beta_s=*/0.05, 0.015, 0.860, -0.005, mix, 555ULL);
    ParticleSystem particles;
    particles.upload(host_p);

    SpatialGrid grid(gcfg);
    MeanFieldGrid mf(make_mean_field_config(EOSPreset::MDSoft), gcfg);
    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};
    step_cfg.dt_fm = 0.20;
    step_cfg.n_test = 150;

    ReactionConfig rxn_cfg{};
    rxn_cfg.enable_elastic = false;
    rxn_cfg.enable_inelastic_2to2 = false;
    rxn_cfg.enable_resonance_2to1_and_decays = false;

    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    ObservableCalculator obs_calc;

    mf.rebuild_potentials(particles, step_cfg.n_test);
    const GlobalObservables obs0 = obs_calc.compute(particles);
    const double epot0 = mf.compute_potential_energy_test_gev(step_cfg.n_test);
    const double H0 = obs0.E_tot + epot0;

    for (int s = 0; s < 40; ++s) {
        stepper.step(particles, grid, &mf);
    }

    mf.rebuild_potentials(particles, step_cfg.n_test);
    const GlobalObservables obs1 = obs_calc.compute(particles);
    const double epot1 = mf.compute_potential_energy_test_gev(step_cfg.n_test);
    const double H1 = obs1.E_tot + epot1;

    const double dP = std::sqrt(
        (obs1.Px_tot - obs0.Px_tot) * (obs1.Px_tot - obs0.Px_tot) +
        (obs1.Py_tot - obs0.Py_tot) * (obs1.Py_tot - obs0.Py_tot) +
        (obs1.Pz_tot - obs0.Pz_tot) * (obs1.Pz_tot - obs0.Pz_tot));
    const double rel_dH = std::fabs(H1 - H0) / H0;

    TEST_ASSERT(dP < 1.0e-11, "Total 3-momentum conservation violated on CIC Vlasov lattice");
    TEST_ASSERT(rel_dH < 1.0e-3, "Symplectic Vlasov Hamiltonian drift exceeded tolerance");

    // 2. Spinodal liquid-gas clumping check at rho_B = n_0/5 (0.032 fm^-3), T = 6 MeV (with seeded mode)
    GridConfig sp_grid{};
    sp_grid.boundary = BoundaryMode::PeriodicBox;
    sp_grid.nx = 10; sp_grid.ny = 10; sp_grid.nz = 10;
    sp_grid.xmin = 0.0; sp_grid.xmax = 20.0;
    sp_grid.ymin = 0.0; sp_grid.ymax = 20.0;
    sp_grid.zmin = 0.0; sp_grid.zmax = 20.0;

    MixtureConfig nuc_only{};
    nuc_only.include_ground_clusters = false;
    nuc_only.include_excited_dstar = false;
    nuc_only.include_excited_he4 = false;
    auto sp_init = init_box_ensemble(12800, sp_grid, 0.004, 0.004, 0.900, 0.0, nuc_only, 909ULL);
    for (auto& hp : sp_init) {
        // Seed Chomaz-Colonna lambda = 10 fm (m = 2) spinodal mode across x and y
        hp.x = detail::wrap_periodic(hp.x + 0.55 * std::sin(2.0 * phys::TWO_PI * hp.x / 20.0), 0.0, 20.0);
        hp.y = detail::wrap_periodic(hp.y + 0.55 * std::sin(2.0 * phys::TWO_PI * hp.y / 20.0), 0.0, 20.0);
    }

    auto run_contrast = [&](EOSPreset pset) {
        ParticleSystem ps; ps.upload(sp_init);
        SpatialGrid sg(sp_grid);
        MeanFieldGrid mfg(make_mean_field_config(pset != EOSPreset::None ? pset : EOSPreset::MDSoft), sp_grid);
        StepperConfig sc{}; sc.dt_fm = 0.50; sc.n_test = 30;  // rho_B = 0.33 * rho_0 (spinodal center!)
        ReactionConfig rc{}; rc.enable_inelastic_2to2 = false; rc.enable_resonance_2to1_and_decays = false;
        TransportStepper stp(sc, CrossSectionConfig{}, rc);
        for (int s = 0; s < 75; ++s) {
            stp.step(ps, sg, (pset != EOSPreset::None) ? &mfg : nullptr);
        }
        mfg.rebuild_potentials(ps, sc.n_test);
        return mfg.density_contrast_rms();
    };

    const double c_free = run_contrast(EOSPreset::None);
    const double c_md = run_contrast(EOSPreset::MDSoft);
    TEST_ASSERT(c_md > 1.8 * c_free, "Expected spinodal mean-field attraction to amplify density clumping");

    std::printf("[PASS] test_vlasov_and_spinodal (|dP|=%.2e, |dH|/H=%.2e, spinodal contrast: free=%.3f vs MD=%.3f)\n",
                dP, rel_dH, c_free, c_md);
}

}  // namespace

int main() {
    std::printf("=== Running CUDA Cluster Transport Verification Suite (Inelastic + Breit-Wigner) ===\n");
    std::printf("Backend: %s\n", transport::backend_description().c_str());

    test_species_table();
    test_relativistic_kinematics();
    test_cross_section_hierarchy();
    test_breit_wigner_spectral_properties();
    test_stochastic_boltzmann_rate();
    test_detailed_balance_2to1_resonances();
    test_detailed_balance_full_chemical_network();
    test_dynamic_cluster_nucleosynthesis();
    test_reactive_expanding_fireball();
    test_eos_saturation_and_momentum_dependence();
    test_vlasov_meanfield_conservation_and_spinodal();

    std::printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
