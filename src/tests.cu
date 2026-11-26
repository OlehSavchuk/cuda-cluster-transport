#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#include "sim/flow_observables.cuh"
#include "sim/freezeout_clusters.cuh"
#include "sim/grid.cuh"
#include "sim/hcgf_io.cuh"
#include "sim/hydro_io.cuh"
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
    int n_excited = 0, n_mesons = 0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        TEST_ASSERT(sp.id == id, "Species ID mismatch");
        TEST_ASSERT(sp.baryon_num >= 0 && sp.baryon_num <= 6, "Baryon number out of range");
        TEST_ASSERT(sp.charge >= -1 && sp.charge <= 4, "Charge out of range");
        TEST_ASSERT(sp.degeneracy == sp.two_J + 1, "Spin degeneracy mismatch g != 2J + 1");
        if (sp.baryon_num == 0) {
            ++n_mesons;
            TEST_ASSERT(sp.pole_mass_gev > 0.13 && sp.pole_mass_gev < 0.15, "Unphysical pion mass");
        } else {
            TEST_ASSERT(sp.pole_mass_gev > 0.90 * sp.baryon_num, "Unphysical baryon/cluster pole mass");
        }
        TEST_ASSERT(sp.rms_radius_fm > 0.5, "Unphysical RMS radius");
        double wp = -1.0, wn = -1.0;
        get_effective_pn_weights(id, wp, wn);
        TEST_ASSERT(wp >= 0.0 && wn >= 0.0 && std::fabs((wp + wn) - sp.baryon_num) < 1.0e-12,
                    "Effective p/n weights must be non-negative and sum to baryon number A");
        if (sp.is_excited) {
            ++n_excited;
            TEST_ASSERT(sp.width_gev > 0.0, "Excited state must have positive width");
        }
    }
    TEST_ASSERT(n_mesons == 3 && n_excited == 47,
                "Expected 3 pions + 47 excited states (4 Delta + 43 A=2..6 nuclear resonances)");
    std::printf("[PASS] test_species_table (58 species: 11 ground hadrons/nuclei [up to 6He,6Li] + 47 resonances)\n");
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

        // Verify unit normalization int_{m_min}^{m_max} A_R(m) dm = 1 via core-refined midpoint check
        const double w_c = 8.0 * sp.width_gev;
        const double c_lo = ((m0 - w_c) > m_min) ? (m0 - w_c) : m_min;
        const double c_hi = ((m0 + w_c) < m_max) ? (m0 + w_c) : m_max;
        const double segs[4] = {m_min, c_lo, c_hi, m_max};
        double area = 0.0;
        for (int sg = 0; sg < 3; ++sg) {
            if (segs[sg + 1] <= segs[sg]) continue;
            constexpr int N_CHK = 2000;
            const double dm = (segs[sg + 1] - segs[sg]) / N_CHK;
            for (int k = 0; k < N_CHK; ++k) {
                area += spectral_function(id, segs[sg] + (k + 0.5) * dm, norms) * dm;
            }
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
    for (int id = 0; id < NUM_SPECIES; ++id) {
        if (is_a4_excited(id)) n_a4star += obs1.counts[id];
    }

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

void test_hic_and_hydro_binary_output() {
    double y_cm = 0.0;
    const int A_nuc = 197, Z_nuc = 79, n_test = 15;
    auto hic_p = init_heavy_ion_collision(A_nuc, Z_nuc, n_test, 250.0, 5.5, 4242ULL, y_cm);
    TEST_ASSERT(static_cast<int>(hic_p.size()) == 2 * A_nuc * n_test, "HIC nucleon count mismatch");
    TEST_ASSERT(std::fabs(y_cm - 0.35723) < 1.0e-3, "Au+Au 250 AMeV y_cm_beam mismatch");

    ParticleSystem ps;
    ps.upload(hic_p);
    ObservableCalculator obs_calc;
    const GlobalObservables obs = obs_calc.compute(ps);
    TEST_ASSERT(obs.baryon_tot == 2LL * A_nuc * n_test && obs.charge_tot == 2LL * Z_nuc * n_test,
                "HIC baryon or charge number mismatch");
    const double p_cm_err = std::sqrt(obs.Px_tot * obs.Px_tot + obs.Py_tot * obs.Py_tot + obs.Pz_tot * obs.Pz_tot);
    TEST_ASSERT(p_cm_err < 1.0e-11, "HIC equal-velocity CM frame total 3-momentum non-zero");

    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 16; gcfg.ny = 16; gcfg.nz = 16;
    gcfg.xmin = -24.0; gcfg.xmax = 24.0;
    gcfg.ymin = -24.0; gcfg.ymax = 24.0;
    gcfg.zmin = -24.0; gcfg.zmax = 24.0;

    const std::string tmp_dir = "/tmp/test_chyd_dump";
    const MeanFieldConfig mf_cfg = make_mean_field_config(EOSPreset::MDSoft);
    write_hydro_grid_binary(tmp_dir, 0, 0, 0.0, n_test, gcfg, mf_cfg, hic_p);

    std::ifstream in(tmp_dir + "/hydro_grid_0000.bin", std::ios::binary);
    TEST_ASSERT(in.good(), "Failed to open binary CHYD hydro grid file");
    HydroGridHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(HydroGridHeader));
    TEST_ASSERT(hdr.magic == 0x44594843U && hdr.n_fields == 8 && hdr.nx == 16, "CHYD header invalid");

    const int nc = gcfg.num_cells();
    std::vector<float> fields(static_cast<std::size_t>(8 * nc));
    in.read(reinterpret_cast<char*>(fields.data()), static_cast<std::streamsize>(fields.size() * sizeof(float)));
    in.close();
    std::filesystem::remove_all(tmp_dir);

    double b_int = 0.0, max_nb_lrf = 0.0, max_eps_lrf = 0.0;
    const double dV = gcfg.cell_volume_fm3();
    for (int c = 0; c < nc; ++c) {
        b_int += static_cast<double>(fields[c]) * dV;
        if (fields[nc + c] > max_nb_lrf) max_nb_lrf = fields[nc + c];
        if (fields[4 * nc + c] > max_eps_lrf) max_eps_lrf = fields[4 * nc + c];
    }
    TEST_ASSERT(std::fabs(b_int - 2.0 * A_nuc) / (2.0 * A_nuc) < 2.0e-3,
                "Integrated binary rho_B_lab * dV != 2 * A_nuc");
    TEST_ASSERT(max_nb_lrf > 0.10 && max_eps_lrf > 0.09, "Expected nuclear-density core in CHYD fields");
    std::printf("[PASS] test_hic_and_hydro_binary_output (int rho_B dV = %.2f / %d, max n_B = %.3f fm^-3, max eps = %.3f GeV/fm^3)\n",
                b_int, 2 * A_nuc, max_nb_lrf, max_eps_lrf);
}

void test_in_medium_collision_energy_conservation() {
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 10; gcfg.ny = 10; gcfg.nz = 10;
    gcfg.xmin = 0.0; gcfg.xmax = 20.0;
    gcfg.ymin = 0.0; gcfg.ymax = 20.0;
    gcfg.zmin = 0.0; gcfg.zmax = 20.0;

    MixtureConfig mix{};
    mix.cluster_yield_boost = 2.0;
    mix.excited_yield_boost = 2.0;
    // Momentum-anisotropic nuclear + cluster matter at ~1.2 * rho_0 under stiff MD potential
    auto init_p = init_box_ensemble(20000, gcfg, /*T_perp=*/0.035, /*T_z=*/0.135,
                                    0.860, 0.0, mix, 7777ULL);

    auto run_aniso_md = [&](bool med_ec, bool exact_proj, double& out_rel_dH, double& out_dP) {
        ParticleSystem ps;
        ps.upload(init_p);
        SpatialGrid sg(gcfg);
        MeanFieldGrid mf(make_mean_field_config(EOSPreset::MDStiff), gcfg);
        StepperConfig sc{};
        sc.dt_fm = 0.20;
        sc.n_test = 16;  // Physical baryon density ~ 1.15 * rho_0!
        sc.collision_substeps = 2;
        sc.in_medium_energy_conservation = med_ec;
        sc.exact_hamiltonian_projection = exact_proj;
        TransportStepper stp(sc, CrossSectionConfig{}, ReactionConfig{});
        ObservableCalculator oc;

        mf.rebuild_potentials(ps, sc.n_test);
        const GlobalObservables o0 = oc.compute(ps);
        const double H0 = o0.E_tot + mf.compute_potential_energy_test_gev(sc.n_test);
        for (int s = 0; s < 45; ++s) stp.step(ps, sg, &mf);
        mf.rebuild_potentials(ps, sc.n_test);
        const GlobalObservables o1 = oc.compute(ps);
        const double H1 = o1.E_tot + mf.compute_potential_energy_test_gev(sc.n_test);

        TEST_ASSERT(o1.baryon_tot == o0.baryon_tot && o1.charge_tot == o0.charge_tot,
                    "Baryon/charge conservation violated in in-medium EC test");
        out_rel_dH = std::fabs(H1 - H0) / std::fabs(H0);
        out_dP = std::sqrt((o1.Px_tot - o0.Px_tot) * (o1.Px_tot - o0.Px_tot) +
                           (o1.Py_tot - o0.Py_tot) * (o1.Py_tot - o0.Py_tot) +
                           (o1.Pz_tot - o0.Pz_tot) * (o1.Pz_tot - o0.Pz_tot));
    };

    double dH_off = 0.0, dP_off = 0.0, dH_on = 0.0, dP_on = 0.0, dH_ex = 0.0, dP_ex = 0.0;
    run_aniso_md(false, false, dH_off, dP_off);
    run_aniso_md(true, false, dH_on, dP_on);
    run_aniso_md(true, true, dH_ex, dP_ex);
    TEST_ASSERT(dP_on < 1.0e-10 && dP_ex < 1.0e-10, "3-momentum drift in in-medium EC mode");
    TEST_ASSERT(dH_on < 0.35 * dH_off && dH_on < 1.0e-3,
                "In-medium energy conservation failed to suppress MD collision energy drift");
    TEST_ASSERT(dH_ex < 1.0e-11, "Exact Hamiltonian projection failed to lock |dH|/H < 1e-11");
    std::printf("[PASS] test_in_medium_collision_energy_conservation (|dH|/H: OFF=%.2e -> ON=%.2e -> EXACT=%.2e, |dP|=%.2e)\n",
                dH_off, dH_on, dH_ex, dP_ex);
}

void test_pion_delta_cross_sections_and_detailed_balance() {
    const SpectralNormTable norms = compute_spectral_norm_table();
    const double mp = get_species_info(PROTON).pole_mass_gev;
    const double mpi = get_species_info(PION_PLUS).pole_mass_gev;
    const double m0 = get_species_info(DELTA_PLUSPLUS).pole_mass_gev;
    const double q0 = two_body_p_cm(m0, mp, mpi);
    const FourVector p_p{std::sqrt(mp * mp + q0 * q0), 0.0, 0.0, q0};
    const FourVector p_pi{std::sqrt(mpi * mpi + q0 * q0), 0.0, 0.0, -q0};
    const PairKinematics kin = compute_pair_kinematics(p_p, mp, p_pi, mpi);

    const double sig_dpp_mb =
        (formation_vmoller_sigma_fm2(PROTON, PION_PLUS, p_p, p_pi, m0, DELTA_PLUSPLUS, norms) / kin.v_moller) *
        phys::FM2_TO_MB;
    const double sig_d0_mb =
        (formation_vmoller_sigma_fm2(PROTON, PION_MINUS, p_p, p_pi, m0, DELTA_ZERO, norms) / kin.v_moller) *
        phys::FM2_TO_MB;
    TEST_ASSERT(sig_dpp_mb > 180.0 && sig_dpp_mb < 220.0, "Expected PDG ~198 mb peak for pi+ p -> Delta++");
    TEST_ASSERT(std::fabs(sig_dpp_mb / sig_d0_mb - 3.0) < 0.02, "Expected 3:1 Clebsch-Gordan ratio (Delta++ / Delta0)");

    // Grand-canonical detailed balance of (N, d, pi-, pi0, pi+, Delta-, Delta0, Delta+, Delta++) at T = 105 MeV
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 8; gcfg.ny = 8; gcfg.nz = 8;
    gcfg.xmin = 0.0; gcfg.xmax = 16.0;
    gcfg.ymin = 0.0; gcfg.ymax = 16.0;
    gcfg.zmin = 0.0; gcfg.zmax = 16.0;

    MixtureConfig mix{};
    mix.include_ground_clusters = true;
    mix.include_excited_dstar = false;
    mix.include_excited_he4 = false;
    mix.include_pions = true;
    mix.include_deltas = true;

    const double T_gev = 0.105, mu_B_gev = 0.720;
    const int n_test = 25;
    const int n_init = grand_canonical_test_particles(gcfg, n_test, T_gev, mu_B_gev, 0.0, mix);
    auto host_p = init_box_ensemble(n_init, gcfg, T_gev, T_gev, mu_B_gev, 0.0, mix, 9191ULL);

    ParticleSystem ps; ps.upload(host_p);
    SpatialGrid grid(gcfg);
    StepperConfig sc{}; sc.dt_fm = 0.08; sc.n_test = n_test; sc.collision_substeps = 2;
    TransportStepper stepper(sc, CrossSectionConfig{}, ReactionConfig{});
    ObservableCalculator oc;

    const GlobalObservables o0 = oc.compute(ps);
    const int pi0_tot = o0.counts[PION_MINUS] + o0.counts[PION_ZERO] + o0.counts[PION_PLUS];
    const int del0_tot = o0.counts[DELTA_MINUS] + o0.counts[DELTA_ZERO] + o0.counts[DELTA_PLUS] + o0.counts[DELTA_PLUSPLUS];
    for (int s = 0; s < 50; ++s) stepper.step(ps, grid);
    const GlobalObservables o1 = oc.compute(ps);
    const StepStats st = stepper.read_stats();
    const int pi1_tot = o1.counts[PION_MINUS] + o1.counts[PION_ZERO] + o1.counts[PION_PLUS];
    const int del1_tot = o1.counts[DELTA_MINUS] + o1.counts[DELTA_ZERO] + o1.counts[DELTA_PLUS] + o1.counts[DELTA_PLUSPLUS];

    TEST_ASSERT(st.formations_2to1 > 1000 && st.decays_1to2 > 1000 && st.collisions_inelastic_2to2 > 300,
                "Expected active pi+N <-> Delta and N+N <-> N+Delta reactions");
    TEST_ASSERT(std::fabs(static_cast<double>(pi1_tot - pi0_tot)) / pi0_tot < 0.10, "Pion equilibrium yield drifted");
    TEST_ASSERT(std::fabs(static_cast<double>(del1_tot - del0_tot)) / del0_tot < 0.14, "Delta equilibrium yield drifted");
    TEST_ASSERT(o1.baryon_tot == o0.baryon_tot && o1.charge_tot == o0.charge_tot, "Signed Q or B violated with pions/Deltas");
    TEST_ASSERT(std::fabs(o1.E_tot - o0.E_tot) / o0.E_tot < 1.0e-11, "Energy conservation violated in pi-Delta box");
    std::printf("[PASS] test_pion_delta_detailed_balance (sig_pk=%.1f/%.1f mb [%.2f], pi:%d->%d, Delta:%d->%d)\n",
                sig_dpp_mb, sig_d0_mb, sig_dpp_mb / sig_d0_mb, pi0_tot, pi1_tot, del0_tot, del1_tot);
}

void test_hic_pion_and_delta_production() {
    double y_cm = 0.0;
    const int A_nuc = 197, Z_nuc = 79, n_test = 15;
    auto hic_p = init_heavy_ion_collision(A_nuc, Z_nuc, n_test, /*elab_amev=*/800.0, /*b=*/3.0, 2027ULL, y_cm);

    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 16; gcfg.ny = 16; gcfg.nz = 16;
    gcfg.xmin = -20.0; gcfg.xmax = 20.0;
    gcfg.ymin = -20.0; gcfg.ymax = 20.0;
    gcfg.zmin = -20.0; gcfg.zmax = 20.0;

    ParticleSystem ps; ps.upload(hic_p);
    SpatialGrid sg(gcfg);
    MeanFieldGrid mf(make_mean_field_config(EOSPreset::MDSoft), gcfg);
    StepperConfig sc{};
    sc.dt_fm = 0.25; sc.n_test = n_test; sc.collision_substeps = 2;
    sc.in_medium_energy_conservation = true;
    sc.exact_hamiltonian_projection = true;

    TransportStepper stp(sc, CrossSectionConfig{}, ReactionConfig{});
    ObservableCalculator oc;
    mf.rebuild_potentials(ps, n_test);
    const GlobalObservables o0 = oc.compute(ps);
    const double H0 = o0.E_tot + mf.compute_potential_energy_test_gev(n_test);

    int peak_delta = 0;
    for (int s = 0; s < 110; ++s) {
        stp.step(ps, sg, &mf);
        if (s % 10 == 0) {
            const GlobalObservables om = oc.compute(ps);
            const int nd = om.counts[DELTA_MINUS] + om.counts[DELTA_ZERO] +
                           om.counts[DELTA_PLUS] + om.counts[DELTA_PLUSPLUS];
            if (nd > peak_delta) peak_delta = nd;
        }
    }
    mf.rebuild_potentials(ps, n_test);
    const GlobalObservables o1 = oc.compute(ps);
    const double H1 = o1.E_tot + mf.compute_potential_energy_test_gev(n_test);

    const auto det = decay_all_unstable_for_detection(ps.download(), sc.seed);
    int n_pim = 0, n_pi0 = 0, n_pip = 0;
    for (const auto& hp : det) {
        if (hp.species == PION_MINUS) ++n_pim;
        else if (hp.species == PION_ZERO) ++n_pi0;
        else if (hp.species == PION_PLUS) ++n_pip;
    }
    TEST_ASSERT(peak_delta > 40 && (n_pim + n_pi0 + n_pip) > 150,
                "Expected robust Delta excitation and pion production in Au+Au @ 800 AMeV");
    TEST_ASSERT(n_pim > n_pip, "Expected neutron-rich Au+Au isospin excess pi- > pi+");
    TEST_ASSERT(o1.baryon_tot == o0.baryon_tot && o1.charge_tot == o0.charge_tot,
                "Baryon or signed electric charge violated in Au+Au pion production");
    TEST_ASSERT(std::fabs(H1 - H0) / std::fabs(H0) < 1.0e-11,
                "Exact MD Hamiltonian conservation violated with active Delta/pion production");
    std::printf("[PASS] test_hic_pion_and_delta_production (peak Delta=%d, freeze-out pi-=%.0d, pi0=%d, pi+=%d, pi-/pi+=%.2f, |dH|/H=%.2e)\n",
                peak_delta, n_pim, n_pi0, n_pip, static_cast<double>(n_pim) / (n_pip > 0 ? n_pip : 1),
                std::fabs(H1 - H0) / std::fabs(H0));
}

void test_hcgf_tmunu_and_landau_frame() {
    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 8; gcfg.ny = 8; gcfg.nz = 8;
    gcfg.xmin = 0.0; gcfg.xmax = 16.0;
    gcfg.ymin = 0.0; gcfg.ymax = 16.0;
    gcfg.zmin = 0.0; gcfg.zmax = 16.0;

    MixtureConfig mix{};
    mix.include_ground_clusters = false;
    mix.include_excited_dstar = false;
    mix.include_excited_he4 = false;
    mix.include_pions = false;
    mix.include_deltas = true;

    const int n_test = 20;
    auto host_p = init_box_ensemble(12000, gcfg, 0.060, 0.060, 0.780, -0.010, mix, 6161ULL);
    const double bx = 0.28, by = -0.18, bz = 0.42;
    const double bmag = std::sqrt(bx * bx + by * by + bz * bz);
    const double gam = 1.0 / std::sqrt(1.0 - bmag * bmag);
    double E_baryon = 0.0, Pz_baryon = 0.0, B_sum = 0.0, Q_baryon = 0.0;
    for (std::size_t idx = 0; idx < host_p.size(); ++idx) {
        auto& hp = host_p[idx];
        PhiloxRng jr(9090ULL, 1ULL, idx);
        FourVector p_cm{};
        while (true) {
            p_cm = sample_maxwell_juttner(hp.mass, 0.060, jr);
            const double b_dot_v = (bx * p_cm.x + by * p_cm.y + bz * p_cm.z) / p_cm.t;
            if (jr.uniform() * (1.0 + bmag) <= (1.0 + b_dot_v)) break;
        }
        const FourVector plab = lorentz_boost_from_cm(hp.mass, p_cm.t, p_cm.x, p_cm.y, p_cm.z, bx, by, bz, gam);
        hp.E = plab.t; hp.px = plab.x; hp.py = plab.y; hp.pz = plab.z;
        const SpeciesInfo sp = get_species_info(hp.species);
        E_baryon += hp.E / n_test; Pz_baryon += hp.pz / n_test;
        B_sum += static_cast<double>(sp.baryon_num) / n_test;
        Q_baryon += static_cast<double>(sp.charge) / n_test;
    }

    // Inject 4,000 hard counter-streaming pions (B=0) moving OPPOSITE to baryons to verify pions are excluded!
    for (int k = 0; k < 4000; ++k) {
        HostParticle pi{};
        pi.x = 8.0; pi.y = 8.0; pi.z = 8.0;
        pi.species = (k % 2 == 0) ? PION_MINUS : PION_PLUS;
        pi.mass = get_species_info(pi.species).pole_mass_gev;
        pi.px = -0.75; pi.py = 0.55; pi.pz = -1.10;
        pi.E = std::sqrt(pi.mass * pi.mass + pi.px * pi.px + pi.py * pi.py + pi.pz * pi.pz);
        host_p.push_back(pi);
    }

    const MeanFieldConfig mf_off = make_mean_field_config(EOSPreset::None);
    const HcgfGridSnapshot snap = compute_hcgf_tmunu_snapshot(n_test, gcfg, mf_off, host_p);
    const std::string tmp_dir = "/tmp/test_hcgf_tmunu";
    write_hcgf_tmunu_binary(tmp_dir, 0, 0, 0.0, gcfg, snap);
    std::ifstream in(tmp_dir + "/tmunu_grid_0000.bin", std::ios::binary);
    HcgfTmunuHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(HcgfTmunuHeader));
    in.close();
    std::filesystem::remove_all(tmp_dir);
    TEST_ASSERT(hdr.magic == 0x46474348U && hdr.n_fields == 24, "Invalid HCGF T^munu binary header");

    const double dV = gcfg.cell_volume_fm3();
    const int nc = gcfg.num_cells();
    double iT00 = 0.0, iT0z = 0.0, iJB0 = 0.0, iJQ0 = 0.0, vx_avg = 0.0, vy_avg = 0.0, vz_avg = 0.0;
    for (int c = 0; c < nc; ++c) {
        iT00 += snap.fields[0][c] * dV; iT0z += snap.fields[3][c] * dV;
        iJB0 += snap.fields[10][c] * dV; iJQ0 += snap.fields[14][c] * dV;
        vx_avg += snap.fields[21][c] / nc; vy_avg += snap.fields[22][c] / nc; vz_avg += snap.fields[23][c] / nc;
    }
    TEST_ASSERT(std::fabs(iT00 - E_baryon) / E_baryon < 1.0e-3 && std::fabs(iT0z - Pz_baryon) / Pz_baryon < 1.0e-3,
                "Baryon-only T^00 or T^0z leaked pion contributions");
    TEST_ASSERT(std::fabs(iJB0 - B_sum) / B_sum < 1.0e-3 && std::fabs(iJQ0 - Q_baryon) < 5.0e-2,
                "Integrated J_B^0 or baryonic J_Q^0 mismatch");
    TEST_ASSERT(std::fabs(vx_avg - bx) < 0.008 && std::fabs(vy_avg - by) < 0.008 && std::fabs(vz_avg - bz) < 0.008,
                "Eckart 4-velocity failed to recover baryon collective boost");

    // Verify with MDSoft potential enabled: int T^00 dV == E_kin_B + E_pot AND int T^0z dV == Pz_baryon!
    const MeanFieldConfig mf_md = make_mean_field_config(EOSPreset::MDSoft);
    ParticleSystem ps_md; ps_md.upload(host_p);
    MeanFieldGrid mfg(mf_md, gcfg);
    mfg.rebuild_potentials(ps_md, n_test);
    const double Epot_phys = mfg.compute_potential_energy_test_gev(n_test) / n_test;
    const HcgfGridSnapshot snap_md = compute_hcgf_tmunu_snapshot(n_test, gcfg, mf_md, host_p);
    double iT00_md = 0.0, iT0z_md = 0.0;
    for (int c = 0; c < nc; ++c) { iT00_md += snap_md.fields[0][c] * dV; iT0z_md += snap_md.fields[3][c] * dV; }
    TEST_ASSERT(std::fabs(iT00_md - (E_baryon + Epot_phys)) / std::fabs(E_baryon + Epot_phys) < 1.0e-4 &&
                std::fabs(iT0z_md - Pz_baryon) / std::fabs(Pz_baryon) < 1.0e-4,
                "T^00 failed exact E_kin + E_pot conservation or T^0z was corrupted by potential");

    std::printf("[PASS] test_hcgf_baryon_eckart_frame (int T00_B=%.2f/%.2f GeV [E_pot=%.2f in T00], v_E=(%.3f,%.3f,%.3f))\n",
                iT00_md, E_baryon + Epot_phys, Epot_phys, vx_avg, vy_avg, vz_avg);
}

void test_pure_hadron_and_grid_freezeout_clusters() {
    double y_cm = 0.0;
    const int A_nuc = 197, Z_nuc = 79, n_test = 20;
    const auto hic_init = init_heavy_ion_collision(A_nuc, Z_nuc, n_test, 250.0, 5.5, 3030ULL, y_cm);

    GridConfig gcfg{};
    gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 16; gcfg.ny = 16; gcfg.nz = 16;
    gcfg.xmin = -24.0; gcfg.xmax = 24.0;
    gcfg.ymin = -24.0; gcfg.ymax = 24.0;
    gcfg.zmin = -24.0; gcfg.zmax = 24.0;

    auto run_mode = [&](ClusterMode cmode, int& out_d, int& out_a34, double& out_rel_dE, double& out_dP) {
        ParticleSystem ps; ps.upload(hic_init);
        SpatialGrid sg(gcfg);
        MeanFieldConfig mf_off = make_mean_field_config(EOSPreset::None);
        StepperConfig sc{}; sc.dt_fm = 0.30; sc.n_test = n_test; sc.collision_substeps = 2;
        ReactionConfig rc{}; rc.enable_nuclear_clusters = (cmode == ClusterMode::Microscopic);
        TransportStepper stp(sc, CrossSectionConfig{}, rc);
        FreezeoutConfig fc{}; fc.mode = cmode; fc.n_fr_fm3 = 0.045; fc.min_time_fm = 25.0;
        FreezeoutCellClusterizer foc(fc, gcfg);
        ObservableCalculator oc;

        const GlobalObservables o0 = oc.compute(ps);
        for (int s = 1; s <= 150; ++s) {
            stp.step(ps, sg);
            if (cmode != ClusterMode::Microscopic && (s % 4 == 0 || s == 150)) {
                foc.process_step(ps, s, s * sc.dt_fm, n_test, mf_off, sc.seed, /*force_finalize=*/(s == 150));
            }
        }
        const GlobalObservables o1 = oc.compute(ps);
        TEST_ASSERT(o1.baryon_tot == o0.baryon_tot && o1.charge_tot == o0.charge_tot,
                    "Baryon or charge conservation violated in freeze-out cluster mode");
        out_rel_dE = std::fabs(o1.E_tot - o0.E_tot) / o0.E_tot;
        out_dP = std::sqrt((o1.Px_tot - o0.Px_tot) * (o1.Px_tot - o0.Px_tot) +
                           (o1.Py_tot - o0.Py_tot) * (o1.Py_tot - o0.Py_tot) +
                           (o1.Pz_tot - o0.Pz_tot) * (o1.Pz_tot - o0.Pz_tot));
        const auto det = decay_all_unstable_for_detection(ps.download(), sc.seed);
        out_d = 0; out_a34 = 0;
        for (const auto& hp : det) {
            if (hp.species == DEUTERON) ++out_d;
            else if (hp.species == TRITON || hp.species == HELION || hp.species == ALPHA) ++out_a34;
        }
    };

    int d_none = 0, a_none = 0, d_fo = 0, a_fo = 0;
    double dE_none = 0.0, dP_none = 0.0, dE_fo = 0.0, dP_fo = 0.0;
    run_mode(ClusterMode::None, d_none, a_none, dE_none, dP_none);
    run_mode(ClusterMode::FreezeoutThermal, d_fo, a_fo, dE_fo, dP_fo);

    TEST_ASSERT(d_none == 0 && a_none == 0, "Expected ZERO nuclear clusters in pure N+Delta+pi mode");
    TEST_ASSERT(d_fo > 150 && a_fo > 30, "Expected abundant thermal d, t, 3He, 4He from n < n_fr freeze-out cells");
    TEST_ASSERT(dE_fo < 1.0e-11 && dP_fo < 1.0e-10, "4-momentum violated in n < n_fr cell thermal clusterization");

    // Explicitly inject back-emitted freeze-out deuterons & alphas into a dense core (n_B > n_melt)
    // and verify Mott re-melting back into un-frozen p + n with exact E, P, B, Q conservation!
    GridConfig box_g{};
    box_g.boundary = BoundaryMode::PeriodicBox;
    box_g.nx = 4; box_g.ny = 4; box_g.nz = 4;
    box_g.xmin = 0.0; box_g.xmax = 8.0;
    box_g.ymin = 0.0; box_g.ymax = 8.0;
    box_g.zmin = 0.0; box_g.lz(); box_g.zmax = 8.0;
    MixtureConfig m_nuc{};
    m_nuc.include_ground_clusters = false; m_nuc.include_excited_dstar = false; m_nuc.include_excited_he4 = false;
    auto dense_p = init_box_ensemble(800, box_g, 0.035, 0.035, 0.880, 0.0, m_nuc, 1234ULL);
    for (int k = 0; k < 25; ++k) {
        HostParticle cl{};
        cl.x = 1.0 + 6.0 * ((k * 7) % 25) / 25.0;
        cl.y = 1.0 + 6.0 * ((k * 11) % 25) / 25.0;
        cl.z = 1.0 + 6.0 * ((k * 13) % 25) / 25.0;
        cl.species = (k % 2 == 0) ? DEUTERON : ALPHA;
        cl.mass = get_species_info(cl.species).pole_mass_gev;
        cl.px = -0.55; cl.py = 0.25; cl.pz = -0.45;
        cl.E = std::sqrt(cl.mass * cl.mass + cl.px * cl.px + cl.py * cl.py + cl.pz * cl.pz);
        cl.freezeout_tag = 1;
        dense_p.push_back(cl);
    }
    ParticleSystem ps_m; ps_m.upload(dense_p);
    ObservableCalculator oc_m;
    const GlobalObservables om0 = oc_m.compute(ps_m);
    FreezeoutConfig fcm{}; fcm.mode = ClusterMode::FreezeoutThermal; fcm.n_fr_fm3 = 0.040; fcm.min_time_fm = 0.0;
    FreezeoutCellClusterizer foc_m(fcm, box_g);
    foc_m.process_step(ps_m, 1, 1.0, /*n_test=*/15, make_mean_field_config(EOSPreset::None), 99ULL, false);
    const GlobalObservables om1 = oc_m.compute(ps_m);
    TEST_ASSERT(foc_m.remelted_clusters() == 25 && om1.counts[DEUTERON] == 0 && om1.counts[ALPHA] == 0,
                "Back-emitted clusters failed to Mott-dissociate inside dense core n_B > n_melt");
    TEST_ASSERT(om1.baryon_tot == om0.baryon_tot && om1.charge_tot == om0.charge_tot &&
                std::fabs(om1.E_tot - om0.E_tot) / om0.E_tot < 1.0e-13,
                "Conservation violated during back-emitted cluster Mott re-melting");

    std::printf("[PASS] test_eckart_freezeout_and_remelt (d=%d, A3+4=%d, remelt=25/25, |dE|/E=%.2e, |dP|=%.2e)\n",
                d_fo, a_fo, dE_fo, dP_fo);
}

void test_a456_isotopes_and_cascade_decays() {
    // 1. Verify multi-step cascade decays of 6H, 6He*(2+), 6Li*(3+), 6Li*(2+_1), 6Be, 6Be*(2+), 5H, 5He, 5Li
    std::vector<HostParticle> res_in;
    const int test_ids[] = {H6_GS, HE6_180, LI6_219, LI6_537, BE6_GS, BE6_167, H5_GS, HE5_GS, LI5_GS};
    long long b_in = 0, q_in = 0;
    double E_in = 0.0, Px_in = 0.0;
    for (int id : test_ids) {
        const SpeciesInfo sp = get_species_info(id);
        HostParticle hp{};
        hp.species = id;
        hp.mass = sp.pole_mass_gev + 0.001;
        hp.px = 0.30; hp.py = -0.20; hp.pz = 0.40;
        hp.E = std::sqrt(hp.mass * hp.mass + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
        res_in.push_back(hp);
        b_in += sp.baryon_num; q_in += sp.charge; E_in += hp.E; Px_in += hp.px;
    }
    const auto dec = decay_all_unstable_for_detection(res_in, 2028ULL);
    long long b_out = 0, q_out = 0;
    double E_out = 0.0, Px_out = 0.0;
    int n_alpha = 0;
    for (const auto& hp : dec) {
        const SpeciesInfo sp = get_species_info(hp.species);
        TEST_ASSERT(!sp.is_excited, "Multi-step cascade failed to reach stable ground-state daughters");
        b_out += sp.baryon_num; q_out += sp.charge; E_out += hp.E; Px_out += hp.px;
        if (hp.species == ALPHA) ++n_alpha;
    }
    TEST_ASSERT(b_out == b_in && q_out == q_in, "B or Q conservation violated in A=5,6 cascade decays");
    TEST_ASSERT(std::fabs(E_out - E_in) / E_in < 1.0e-13 && std::fabs(Px_out - Px_in) < 1.0e-13,
                "4-momentum conservation violated in A=5,6 cascade decays");
    TEST_ASSERT(n_alpha >= 7, "Expected >= 7 4He daughters from 5He, 5Li, 6He*, 6Li*, 6Be cascades");

    // 2. Compare max_cluster_A = 4 vs max_cluster_A = 6 in cool subsaturation freeze-out cell:
    //    verify A=5,6 isotopes boost 4He via feed-down and populate bound 6He & 6Li!
    double n4[NUM_SPECIES] = {}, n6[NUM_SPECIES] = {}, T4 = 0.0, T6 = 0.0, muB = 0.0, muQ = 0.0;
    fo_detail::invert_local_cluster_eos(0.020, 0.40, 0.008, true, 4, T4, muB, muQ, n4);
    fo_detail::invert_local_cluster_eos(0.020, 0.40, 0.008, true, 6, T6, muB, muQ, n6);
    const double alpha_feed6 = n6[ALPHA] + n6[HE5_GS] + n6[HE5_127] + n6[LI5_GS] + n6[LI5_149] +
                               n6[HE6_180] + n6[LI6_219] + n6[LI6_431] + n6[LI6_537] + n6[LI6_565] +
                               n6[BE6_GS] + n6[BE6_167];
    TEST_ASSERT(n6[HE6_GS] > 0.0 && n6[LI6_GS] > 0.0, "Expected positive bound 6He and 6Li densities");
    TEST_ASSERT(alpha_feed6 > 1.25 * n4[ALPHA], "Expected A=5,6 alpha-decaying isotopes to boost total 4He yield");
    std::printf("[PASS] test_a456_isotopes_and_cascade_decays (cascades->%zu ground daughters, 4He_tot(A<=6)/4He(A<=4)=%.2f)\n",
                dec.size(), alpha_feed6 / n4[ALPHA]);
}

void test_hcgf_pratt_torrieri_momentum_anisotropy() {
    // 1. Verify exact Lorentz stress projection P*_ij & det(Lambda) == 1 on boosted 3D shear tensor
    const double vx = 0.22, vy = -0.14, vz = 0.31;
    const double v2 = vx * vx + vy * vy + vz * vz;
    const double u0 = 1.0 / std::sqrt(1.0 - v2), ux = u0 * vx, uy = u0 * vy, uz = u0 * vz;
    const double ig = 1.0 / (u0 + 1.0);
    const double B[4][4] = {
        {u0, ux, uy, uz},
        {ux, 1.0 + ux * ux * ig, ux * uy * ig, ux * uz * ig},
        {uy, uy * ux * ig, 1.0 + uy * uy * ig, uy * uz * ig},
        {uz, uz * ux * ig, uz * uy * ig, 1.0 + uz * uz * ig}
    };
    const double T0[4][4] = {
        {0.020,  0.0,     0.0,     0.0},
        {0.0,    0.0010,  0.0,     0.00045},
        {0.0,    0.0,     0.0026,  0.0},
        {0.0,    0.00045, 0.0,     0.0009}
    };
    double Tlab[4][4] = {};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) Tlab[m][n] += B[m][a] * B[n][b] * T0[a][b];

    double nb_e = 0.0, eps_e = 0.0, P_e = 0.0, vxe = 0.0, vye = 0.0, vze = 0.0, Pstar[6] = {};
    hcgf_detail::eval_eckart_frame(0.020 * u0, 0.020 * ux, 0.020 * uy, 0.020 * uz,
                                   Tlab[0][0], Tlab[0][1], Tlab[0][2], Tlab[0][3],
                                   Tlab[1][1], Tlab[1][2], Tlab[1][3], Tlab[2][2], Tlab[2][3], Tlab[3][3],
                                   nb_e, eps_e, P_e, vxe, vye, vze);
    hcgf_detail::project_stress_to_rest_frame(vxe, vye, vze,
                                              Tlab[0][0], Tlab[0][1], Tlab[0][2], Tlab[0][3],
                                              Tlab[1][1], Tlab[1][2], Tlab[1][3], Tlab[2][2], Tlab[2][3], Tlab[3][3],
                                              Pstar);
    TEST_ASSERT(std::fabs(Pstar[0] - 0.0010) < 1.0e-14 && std::fabs(Pstar[3] - 0.0026) < 1.0e-14 &&
                std::fabs(Pstar[2] - 0.00045) < 1.0e-14 &&
                std::fabs((Pstar[0] + Pstar[3] + Pstar[5]) / 3.0 - P_e) < 1.0e-14,
                "Eckart rest-frame 3x3 stress projection mismatch");

    double L[9] = {}, alp[3] = {};
    hcgf_detail::compute_pratt_torrieri_lambda(Pstar, L, alp);
    const double detL = L[0] * (L[4] * L[8] - L[5] * L[7]) -
                        L[1] * (L[3] * L[8] - L[5] * L[6]) +
                        L[2] * (L[3] * L[7] - L[4] * L[6]);
    TEST_ASSERT(std::fabs(detL - 1.0) < 1.0e-14 && std::fabs(L[1] - L[3]) < 1.0e-15,
                "Pratt-Torrieri Lambda must be symmetric with exact det(Lambda) == 1");

    // 2. Verify FreezeoutCellClusterizer transfers local squeeze-out onto A>=2 clusters with exact P^mu conservation
    GridConfig gcfg{};
    gcfg.nx = 4; gcfg.ny = 4; gcfg.nz = 4;
    gcfg.xmin = 0.0; gcfg.xmax = 12.0; gcfg.ymin = 0.0; gcfg.ymax = 12.0; gcfg.zmin = 0.0; gcfg.zmax = 12.0;
    gcfg.boundary = BoundaryMode::PeriodicBox;
    PhiloxRng rng(9911ULL, 0ULL, 0ULL);
    std::vector<HostParticle> base_p;
    double E0 = 0.0, Px0 = 0.0;
    for (int i = 0; i < 3200; ++i) {
        HostParticle hp{};
        hp.species = (i % 5 < 2) ? PROTON : NEUTRON;
        hp.mass = get_species_info(hp.species).pole_mass_gev;
        hp.x = 12.0 * rng.uniform(); hp.y = 12.0 * rng.uniform(); hp.z = 12.0 * rng.uniform();
        const double qx = std::sqrt(hp.mass * 0.005) * rng.normal();
        const double qy = std::sqrt(hp.mass * 0.024) * rng.normal();
        const double qz = std::sqrt(hp.mass * 0.010) * rng.normal();
        const double eq = std::sqrt(hp.mass * hp.mass + qx * qx + qy * qy + qz * qz);
        const FourVector pl = lorentz_boost_from_cm(hp.mass, eq, qx, qy, qz, 0.14, 0.0, 0.22, 1.0 / std::sqrt(1.0 - 0.068));
        hp.E = pl.t; hp.px = pl.x; hp.py = pl.y; hp.pz = pl.z;
        base_p.push_back(hp);
        E0 += hp.E; Px0 += hp.px;
    }
    auto eval_cl_squeeze = [&](bool aniso_on, int& n_cl_out) {
        ParticleSystem ps; ps.upload(base_p);
        FreezeoutConfig fc{}; fc.mode = ClusterMode::FreezeoutThermal; fc.n_fr_fm3 = 0.030; fc.local_anisotropy = aniso_on;
        FreezeoutCellClusterizer clz(fc, gcfg);
        clz.process_step(ps, 1, 5.0, 100, make_mean_field_config(EOSPreset::None), 4242ULL, true);
        const auto out = ps.download();
        double E1 = 0.0, Px1 = 0.0, kxx = 0.0, kyy = 0.0;
        n_cl_out = 0;
        for (const auto& hp : out) {
            E1 += hp.E; Px1 += hp.px;
            if (get_species_info(hp.species).baryon_num >= 2) {
                ++n_cl_out;
                const FourVector pc = lorentz_boost_from_cm(hp.mass, hp.E, hp.px, hp.py, hp.pz, -0.14, 0.0, -0.22, 1.0 / std::sqrt(1.0 - 0.068));
                kxx += (pc.x * pc.x) / hp.mass; kyy += (pc.y * pc.y) / hp.mass;
            }
        }
        TEST_ASSERT(std::fabs(E1 - E0) / E0 < 1.0e-12 && std::fabs(Px1 - Px0) < 1.0e-10, "Anisotropic FO violated P^mu");
        return (kyy - kxx) / std::max(kyy + kxx, 1.0e-12);
    };
    int n_cl_iso = 0, n_cl_pt = 0;
    const double sq_iso = eval_cl_squeeze(false, n_cl_iso);
    const double sq_pt  = eval_cl_squeeze(true,  n_cl_pt);
    TEST_ASSERT(n_cl_pt == n_cl_iso && n_cl_pt > 50, "det(Lambda)=1 must preserve exact cluster multiplicities");
    TEST_ASSERT(std::fabs(sq_iso) < 0.10 && sq_pt > 0.42, "Pratt-Torrieri failed to imprint local squeeze-out onto clusters");
    std::printf("[PASS] test_hcgf_pratt_torrieri_momentum_anisotropy (detL=%.6f, N_cl=%d, eps_cl: iso=%+.3f -> PT=%+.3f)\n",
                detL, n_cl_pt, sq_iso, sq_pt);
}

void test_pauli_blocking_and_interacting_hcgf() {
    // 1. Cold Woods-Saxon + Thomas-Fermi 197Au nucleus at rest: verify core f_n ~ 1 and >80% Pauli blocking
    double y_dummy = 0.0;
    auto two_au = init_heavy_ion_collision(197, 79, 80, 0.01, 0.0, 7777ULL, y_dummy);
    std::vector<HostParticle> one_au(two_au.begin(), two_au.begin() + 197 * 80);
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (const auto& hp : one_au) { cx += hp.x; cy += hp.y; cz += hp.z; }
    cx /= one_au.size(); cy /= one_au.size(); cz /= one_au.size();
    for (auto& hp : one_au) { hp.x -= cx; hp.y -= cy; hp.z -= cz; hp.collided = 0; }

    ParticleSystem ps_chk; ps_chk.upload(one_au);
    PauliGridConfig pcfg{}; pcfg.enabled = true;
    PauliPhaseSpaceGrid pgrid(pcfg);
    pgrid.rebuild(ps_chk, 80);
    const double f_core = eval_pauli_f(pgrid.view(), NEUTRON, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const double f_high = eval_pauli_f(pgrid.view(), NEUTRON, 0.0, 0.0, 0.0, 0.0, 0.0, 0.48);
    TEST_ASSERT(f_core > 0.82 && f_high < 0.05, "6D Pauli grid failed cold Fermi-sea occupancy check");

    GridConfig gcfg{}; gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 16; gcfg.ny = 16; gcfg.nz = 16;
    gcfg.xmin = -16.0; gcfg.xmax = 16.0; gcfg.ymin = -16.0; gcfg.ymax = 16.0; gcfg.zmin = -16.0; gcfg.zmax = 16.0;

    auto run_cold_au = [&](bool pauli_on, unsigned long long& el_out, unsigned long long& blk_out) {
        ParticleSystem ps; ps.upload(one_au);
        SpatialGrid sg(gcfg);
        StepperConfig sc{}; sc.dt_fm = 0.25; sc.n_test = 80; sc.pauli_blocking = pauli_on; sc.in_medium_energy_conservation = false;
        ReactionConfig rc{}; rc.enable_inelastic_2to2 = false; rc.enable_resonance_2to1_and_decays = false;
        TransportStepper stp(sc, CrossSectionConfig{}, rc);
        for (int s = 0; s < 8; ++s) stp.step(ps, sg);
        const StepStats st = stp.read_stats();
        el_out = st.collisions_elastic; blk_out = st.pauli_blocked;
    };
    unsigned long long el_off = 0, blk_off = 0, el_on = 0, blk_on = 0;
    run_cold_au(false, el_off, blk_off);
    run_cold_au(true,  el_on,  blk_on);
    TEST_ASSERT(el_off > 500 && el_on < 0.22 * el_off && blk_on > 3 * el_on,
                "Pauli blocking failed to suppress cold intra-nuclear Fermi-sea collisions");

    // 2. Quantum INCG EoS at n_B = rho_0/8 with interacting cold/warm excitation e_tr = +2.0 MeV/B
    double T_q = 0.0, muB_q = 0.0, muQ_q = 0.0, n_q[NUM_SPECIES] = {};
    fo_detail::invert_local_cluster_eos(0.020, 0.401, 0.0020, true, 6, T_q, muB_q, muQ_q, n_q, true);
    const double alpha_tot = (340.0 / 0.020) * (n_q[ALPHA] + n_q[HE5_GS] + n_q[HE5_127] + n_q[LI5_GS] +
                             n_q[LI5_149] + n_q[HE6_180] + n_q[LI6_219] + n_q[LI6_431] + n_q[LI6_537] +
                             n_q[LI6_565] + n_q[BE6_GS] + n_q[BE6_167]);
    TEST_ASSERT(T_q * 1.0e3 > 6.5 && T_q * 1.0e3 < 13.5 && alpha_tot > 20.0,
                "Quantum INCG EoS failed expected T_fr ~ 9-11 MeV and enhanced 4He yield");
    std::printf("[PASS] test_pauli_blocking_and_interacting_hcgf (f_core=%.2f, cold NN blocked=%.1f%%, T_fr=%.2f MeV, 4He=%.1f)\n",
                f_core, 100.0 * static_cast<double>(blk_on) / std::max(1ULL, el_on + blk_on), T_q * 1.0e3, alpha_tot);
}

void test_poisson_coulomb_potential_and_energy_conservation() {
    // 1. Uniform Z = 79 charged sphere of radius R = 6.8 fm in open 3D space [-24, 24]^3 fm
    constexpr int Z = 79, N_NEUT = 118, NT = 120;
    constexpr double R_SPH = 6.8;
    std::vector<HostParticle> sph;
    sph.reserve((Z + N_NEUT) * NT);
    for (int i = 0; i < (Z + N_NEUT) * NT; ++i) {
        PhiloxRng rng(314159ULL, 0ULL, static_cast<std::uint64_t>(i));
        const bool is_p = (i < Z * NT);
        const double r = R_SPH * std::cbrt(rng.uniform());
        const double cth = 1.0 - 2.0 * rng.uniform();
        const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
        const double phi = phys::TWO_PI * rng.uniform();
        HostParticle hp{};
        hp.species = is_p ? PROTON : NEUTRON;
        hp.mass = get_species_info(hp.species).pole_mass_gev;
        hp.x = r * sth * std::cos(phi); hp.y = r * sth * std::sin(phi); hp.z = r * cth;
        hp.E = hp.mass;
        sph.push_back(hp);
    }

    GridConfig gcfg{}; gcfg.boundary = BoundaryMode::OpenExpanding;
    gcfg.nx = 24; gcfg.ny = 24; gcfg.nz = 24;
    gcfg.xmin = -24.0; gcfg.xmax = 24.0; gcfg.ymin = -24.0; gcfg.ymax = 24.0; gcfg.zmin = -24.0; gcfg.zmax = 24.0;

    MeanFieldConfig c_only{}; c_only.enabled = false; c_only.coulomb_enabled = true; c_only.gaussian_smooth_passes = 0;
    ParticleSystem ps0; ps0.upload(sph);
    MeanFieldGrid mfg0(c_only, gcfg);
    mfg0.rebuild_potentials(ps0, NT);

    const double phi_c_exact = 1.5 * Z * phys::E2_COULOMB_GEV_FM / R_SPH;
    const double phi_ext_exact = Z * phys::E2_COULOMB_GEV_FM / 15.0;
    const double ecoul_exact = 0.6 * Z * Z * phys::E2_COULOMB_GEV_FM / R_SPH;
    const double phi_c_num = mf_detail::sample_cic_scalar(mfg0.view().phi_coul, 0.0, 0.0, 0.0, gcfg);
    const double phi_ext_num = mf_detail::sample_cic_scalar(mfg0.view().phi_coul, 15.0, 0.0, 0.0, gcfg);
    const double ecoul_num = mfg0.compute_coulomb_energy_test_gev(NT) / NT;
    TEST_ASSERT(std::fabs(phi_c_num - phi_c_exact) / phi_c_exact < 0.05 &&
                std::fabs(phi_ext_num - phi_ext_exact) / phi_ext_exact < 0.03 &&
                std::fabs(ecoul_num - ecoul_exact) / ecoul_exact < 0.05,
                "3D Poisson open-boundary Coulomb potential/energy mismatch vs analytic sphere");

    // 2. Pure Vlasov (MDSoft, Coulomb ON vs OFF): verify pure Coulomb radial push & exact H, P conservation
    auto run_vlasov_sphere = [&](bool coul_on, double& pr_p_out, double& rel_dH_out, double& dP_out) {
        MeanFieldConfig mf_cfg = make_mean_field_config(EOSPreset::MDSoft);
        mf_cfg.coulomb_enabled = coul_on;
        ParticleSystem ps; ps.upload(sph);
        MeanFieldGrid mfg(mf_cfg, gcfg);
        SpatialGrid sg(gcfg);
        StepperConfig sc{}; sc.dt_fm = 0.20; sc.n_test = NT;
        ReactionConfig rc{}; rc.enable_elastic = false; rc.enable_inelastic_2to2 = false; rc.enable_resonance_2to1_and_decays = false;
        TransportStepper stp(sc, CrossSectionConfig{}, rc);
        ObservableCalculator oc;

        mfg.rebuild_potentials(ps, NT);
        const GlobalObservables o0 = oc.compute(ps);
        const double H0 = o0.E_tot + mfg.compute_potential_energy_test_gev(NT);
        for (int s = 0; s < 30; ++s) stp.step(ps, sg, &mfg);
        mfg.rebuild_potentials(ps, NT);
        const GlobalObservables o1 = oc.compute(ps);
        const double H1 = o1.E_tot + mfg.compute_potential_energy_test_gev(NT);

        pr_p_out = 0.0;
        for (const auto& hp : ps.download()) {
            if (hp.species != PROTON) continue;
            const double r = std::max(1.0e-6, std::sqrt(hp.x * hp.x + hp.y * hp.y + hp.z * hp.z));
            pr_p_out += ((hp.x * hp.px + hp.y * hp.py + hp.z * hp.pz) / r) / (Z * NT);
        }
        rel_dH_out = std::fabs(H1 - H0) / std::fabs(H0);
        dP_out = std::sqrt((o1.Px_tot - o0.Px_tot) * (o1.Px_tot - o0.Px_tot) +
                           (o1.Py_tot - o0.Py_tot) * (o1.Py_tot - o0.Py_tot) +
                           (o1.Pz_tot - o0.Pz_tot) * (o1.Pz_tot - o0.Pz_tot));
    };
    double pr_off = 0.0, dh_off = 0.0, dp_off = 0.0, pr_on = 0.0, dh_on = 0.0, dp_on = 0.0;
    run_vlasov_sphere(false, pr_off, dh_off, dp_off);
    run_vlasov_sphere(true,  pr_on,  dh_on,  dp_on);
    TEST_ASSERT(pr_on - pr_off > 0.006 && dh_on < 3.0e-4 && dp_on < 1.0e-10,
                "Coulomb Vlasov expansion failed proton radial push or Hamiltonian/momentum conservation");
    std::printf("[PASS] test_poisson_coulomb_potential_and_energy_conservation (Phi0=%.2f/%.2f MeV, E_C=%.3f/%.3f GeV, dp_r_Coul=%+.1f MeV/c, |dH|/H=%.2e)\n",
                phi_c_num * 1e3, phi_c_exact * 1e3, ecoul_num, ecoul_exact, (pr_on - pr_off) * 1e3, dh_on);
}

void test_grand_canonical_freezeout_sampling() {
    // Verify GrandCanonical Poisson sampling on sparse 1-fm^3 cells (Bc ~ 3) reproduces exact HRG <B>, <Q>, and 4He!
    GridConfig gcfg{}; gcfg.boundary = BoundaryMode::PeriodicBox;
    gcfg.nx = 10; gcfg.ny = 10; gcfg.nz = 10;
    gcfg.xmin = 0.0; gcfg.xmax = 10.0; gcfg.ymin = 0.0; gcfg.ymax = 10.0; gcfg.zmin = 0.0; gcfg.zmax = 10.0;

    constexpr int NT = 200;
    constexpr int B_PHYS = 15;  // n_B = 0.015 fm^-3, only 3.0 test-baryons per 1 fm^3 cell!
    constexpr int Z_PHYS = 6;   // Y_p = 0.40
    PhiloxRng rng(884422ULL, 0ULL, 0ULL);
    std::vector<HostParticle> init_p;
    for (int i = 0; i < B_PHYS * NT; ++i) {
        HostParticle hp{};
        hp.species = (i < Z_PHYS * NT) ? PROTON : NEUTRON;
        hp.mass = get_species_info(hp.species).pole_mass_gev;
        hp.x = 10.0 * rng.uniform(); hp.y = 10.0 * rng.uniform(); hp.z = 10.0 * rng.uniform();
        const double qx = std::sqrt(hp.mass * 0.004) * rng.normal();
        const double qy = std::sqrt(hp.mass * 0.014) * rng.normal();
        const double qz = std::sqrt(hp.mass * 0.006) * rng.normal();
        hp.E = std::sqrt(hp.mass * hp.mass + qx * qx + qy * qy + qz * qz);
        hp.px = qx; hp.py = qy; hp.pz = qz;
        init_p.push_back(hp);
    }

    ParticleSystem ps; ps.upload(init_p);
    FreezeoutConfig fc{};
    fc.mode = ClusterMode::FreezeoutThermal;
    fc.sampling = FreezeoutSampling::GrandCanonical;
    fc.n_fr_fm3 = 0.025;
    fc.local_anisotropy = true;
    FreezeoutCellClusterizer clz(fc, gcfg);
    clz.process_step(ps, 1, 5.0, NT, make_mean_field_config(EOSPreset::MDSoft), 2026ULL, true);

    const auto dec = decay_all_unstable_for_detection(ps.download(), 2026ULL);
    double b_out = 0.0, q_out = 0.0, n_d = 0.0, n_he4 = 0.0, kxx = 0.0, kyy = 0.0;
    for (const auto& hp : dec) {
        const SpeciesInfo sp = get_species_info(hp.species);
        b_out += sp.baryon_num / static_cast<double>(NT);
        q_out += sp.charge / static_cast<double>(NT);
        if (hp.species == DEUTERON) n_d += 1.0 / NT;
        if (hp.species == ALPHA) n_he4 += 1.0 / NT;
        if (sp.baryon_num >= 2) {
            kxx += (hp.px * hp.px) / hp.mass;
            kyy += (hp.py * hp.py) / hp.mass;
        }
    }
    const double sq_gc = (kyy - kxx) / std::max(kyy + kxx, 1.0e-12);
    TEST_ASSERT(std::fabs(b_out - B_PHYS) / B_PHYS < 0.05 && std::fabs(q_out - Z_PHYS) / Z_PHYS < 0.06,
                "GrandCanonical sampling violated ensemble <B> or <Q> consistency");
    TEST_ASSERT(n_d > 0.6 && n_he4 > 0.8 && sq_gc > 0.25,
                "GrandCanonical sampling failed cluster yield or Pratt-Torrieri anisotropy transfer on 1 fm^3 cells");
    std::printf("[PASS] test_grand_canonical_freezeout_sampling (<B>=%.2f/%d, <Q>=%.2f/%d, d=%.2f, 4He=%.2f, eps_cl=%+.3f)\n",
                b_out, B_PHYS, q_out, Z_PHYS, n_d, n_he4, sq_gc);
}

void test_freezeout_oversampling_and_backflow_remelt() {
    // 1. Verify M_over = 40 freeze-out oversampling produces 40x raw cluster stats with unbiased <B>, <Q>, yields
    GridConfig gcfg{};
    gcfg.nx = 10; gcfg.ny = 10; gcfg.nz = 10;
    gcfg.xmin = 0.0; gcfg.xmax = 10.0; gcfg.ymin = 0.0; gcfg.ymax = 10.0; gcfg.zmin = 0.0; gcfg.zmax = 10.0;
    gcfg.boundary = BoundaryMode::PeriodicBox;

    constexpr int NT = 15, B_PHYS = 15, Z_PHYS = 6, M_OVER = 40;
    PhiloxRng rng(991155ULL, 0ULL, 0ULL);
    std::vector<HostParticle> init_p;
    for (int i = 0; i < B_PHYS * NT; ++i) {
        HostParticle hp{};
        hp.species = (i < Z_PHYS * NT) ? PROTON : NEUTRON;
        hp.mass = get_species_info(hp.species).pole_mass_gev;
        hp.x = 10.0 * rng.uniform(); hp.y = 10.0 * rng.uniform(); hp.z = 10.0 * rng.uniform();
        const double qx = std::sqrt(hp.mass * 0.004) * rng.normal();
        const double qy = std::sqrt(hp.mass * 0.014) * rng.normal();
        const double qz = std::sqrt(hp.mass * 0.006) * rng.normal();
        hp.E = std::sqrt(hp.mass * hp.mass + qx * qx + qy * qy + qz * qz);
        hp.px = qx; hp.py = qy; hp.pz = qz;
        init_p.push_back(hp);
    }

    // Compare 1 single M_OVER = 40 oversampled pass against the mean of 40 independent M_OVER = 1 passes!
    double b_ens = 0.0, q_ens = 0.0, cl_ens = 0.0;
    for (int r = 0; r < M_OVER; ++r) {
        ParticleSystem ps1; ps1.upload(init_p);
        FreezeoutConfig fc1{};
        fc1.mode = ClusterMode::FreezeoutThermal;
        fc1.sampling = FreezeoutSampling::GrandCanonical;
        fc1.oversample_factor = 1;
        fc1.n_fr_fm3 = 0.025;
        FreezeoutCellClusterizer clz1(fc1, gcfg);
        clz1.process_step(ps1, 1, 5.0, NT, make_mean_field_config(EOSPreset::MDSoft), 4000ULL + r * 97ULL, true);
        for (const auto& hp : ps1.download()) {
            const SpeciesInfo sp = get_species_info(hp.species);
            b_ens += sp.baryon_num;
            q_ens += sp.charge;
            if (sp.baryon_num >= 2) cl_ens += 1.0;
        }
    }
    b_ens /= M_OVER; q_ens /= M_OVER; cl_ens /= M_OVER;

    ParticleSystem ps; ps.upload(init_p);
    FreezeoutConfig fc{};
    fc.mode = ClusterMode::FreezeoutThermal;
    fc.sampling = FreezeoutSampling::GrandCanonical;
    fc.oversample_factor = M_OVER;
    fc.n_fr_fm3 = 0.025;
    FreezeoutCellClusterizer clz(fc, gcfg);
    clz.process_step(ps, 1, 5.0, NT, make_mean_field_config(EOSPreset::MDSoft), 3030ULL, true);

    ObservableCalculator obs_calc;
    const GlobalObservables obs_ov = obs_calc.compute(ps);
    int raw_cl = 0;
    for (const auto& hp : ps.download()) {
        if (get_species_info(hp.species).baryon_num >= 2) ++raw_cl;
    }
    TEST_ASSERT(std::fabs(raw_cl / static_cast<double>(M_OVER) - cl_ens) / cl_ens < 0.06 &&
                std::fabs(static_cast<double>(obs_ov.baryon_tot) - b_ens) / b_ens < 0.03 &&
                std::fabs(static_cast<double>(obs_ov.charge_tot) - q_ens) / q_ens < 0.04,
                "1x M_OVER=40 pass failed to match 40x M_OVER=1 ensemble mean");

    // 2. Verify oversampled backflow (freezeout_tag = M_OVER) systematically resamples to N_test scale & remelts!
    GridConfig g2{};
    g2.nx = 4; g2.ny = 4; g2.nz = 4;
    g2.xmin = 0.0; g2.xmax = 8.0; g2.ymin = 0.0; g2.ymax = 8.0; g2.zmin = 0.0; g2.zmax = 8.0;
    g2.boundary = BoundaryMode::PeriodicBox;

    constexpr int NT2 = 10;
    std::vector<HostParticle> core_p;
    // Dense bath: 80 test nucleons in cell (1,1,1) [x,y,z in 2..4 fm, dV = 8 fm^3] -> n_B ~ 1.0 / (8) * ... > n_melt
    for (int i = 0; i < 160; ++i) {
        HostParticle hp{};
        hp.species = (i % 2 == 0) ? PROTON : NEUTRON;
        hp.mass = get_species_info(hp.species).pole_mass_gev;
        hp.x = 2.5 + 1.0 * rng.uniform(); hp.y = 2.5 + 1.0 * rng.uniform(); hp.z = 2.5 + 1.0 * rng.uniform();
        const double qx = 0.14 * rng.normal(), qy = 0.14 * rng.normal(), qz = 0.14 * rng.normal();
        hp.E = std::sqrt(hp.mass * hp.mass + qx * qx + qy * qy + qz * qz);
        hp.px = qx; hp.py = qy; hp.pz = qz;
        core_p.push_back(hp);
    }
    // Inject 40 * 12 = 480 oversampled deuterons (= 12 test d -> 24 B, 12 Q) + 40 * 8 = 320 oversampled protons (= 8 B, 8 Q)
    for (int i = 0; i < M_OVER * 12; ++i) {
        HostParticle hd{};
        hd.species = DEUTERON; hd.mass = get_species_info(DEUTERON).pole_mass_gev;
        hd.x = 3.0; hd.y = 3.0; hd.z = 3.0;
        hd.px = 0.12; hd.py = -0.08; hd.pz = 0.05;
        hd.E = std::sqrt(hd.mass * hd.mass + hd.px * hd.px + hd.py * hd.py + hd.pz * hd.pz);
        hd.freezeout_tag = M_OVER;
        core_p.push_back(hd);
    }
    for (int i = 0; i < M_OVER * 8; ++i) {
        HostParticle hp{};
        hp.species = PROTON; hp.mass = phys::MASS_PROTON;
        hp.x = 3.0; hp.y = 3.0; hp.z = 3.0;
        hp.px = -0.10; hp.py = 0.06; hp.pz = -0.04;
        hp.E = std::sqrt(hp.mass * hp.mass + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
        hp.freezeout_tag = M_OVER;
        core_p.push_back(hp);
    }

    ParticleSystem ps2; ps2.upload(core_p);
    FreezeoutConfig fc2{};
    fc2.mode = ClusterMode::FreezeoutThermal;
    fc2.sampling = FreezeoutSampling::GrandCanonical;
    fc2.oversample_factor = M_OVER;
    fc2.n_fr_fm3 = 0.020;  // n_melt = 0.023 fm^-3 << core n_B (~0.20 fm^-3)
    fc2.min_time_fm = 0.0;
    FreezeoutCellClusterizer clz2(fc2, g2);
    clz2.process_step(ps2, 1, 2.0, NT2, make_mean_field_config(EOSPreset::None), 9090ULL, false);

    int active_dense_b = 0, active_dense_q = 0, rem_frozen_cl = 0;
    for (const auto& hp : ps2.download()) {
        const SpeciesInfo sp = get_species_info(hp.species);
        if (hp.freezeout_tag == 0) {
            active_dense_b += sp.baryon_num;
            active_dense_q += sp.charge;
        } else if (sp.baryon_num >= 2) {
            ++rem_frozen_cl;
        }
    }
    TEST_ASSERT(clz2.remelted_clusters() == 12 && clz2.reabsorbed_nucleons() == 8 &&
                clz2.thinned_oversamples() == (480 - 12) + (320 - 8) && rem_frozen_cl == 0,
                "Backflow systematic resampling + Mott remelt did not promote exact 12 d + 8 p");
    TEST_ASSERT(active_dense_b == 160 + 24 + 8 && active_dense_q == 80 + 12 + 8,
                "Backflow resampling + remelt violated N_test-scaled baryon or charge conservation");
    std::printf("[PASS] test_freezeout_oversampling_and_backflow_remelt (M=%d: raw_cl=%d, remelt=%lld/12 d, reabs=%lld/8 p, thinned=%lld, B_dense=%d/192)\n",
                M_OVER, raw_cl, clz2.remelted_clusters(), clz2.reabsorbed_nucleons(),
                clz2.thinned_oversamples(), active_dense_b);
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
    test_hic_and_hydro_binary_output();
    test_in_medium_collision_energy_conservation();
    test_pion_delta_cross_sections_and_detailed_balance();
    test_hic_pion_and_delta_production();
    test_hcgf_tmunu_and_landau_frame();
    test_pure_hadron_and_grid_freezeout_clusters();
    test_a456_isotopes_and_cascade_decays();
    test_hcgf_pratt_torrieri_momentum_anisotropy();
    test_pauli_blocking_and_interacting_hcgf();
    test_poisson_coulomb_potential_and_energy_conservation();
    test_grand_canonical_freezeout_sampling();
    test_freezeout_oversampling_and_backflow_remelt();

    std::printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
