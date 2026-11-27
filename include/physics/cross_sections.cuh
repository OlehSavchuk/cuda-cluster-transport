#pragma once

#include <cmath>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/species.cuh"

namespace transport {

enum class CrossSectionModel : int {
    ConstantIsotropic = 0,   // Fixed sigma_const_mb for all pairs (analytic Boltzmann tests)
    ParametricGlauber = 1    // ERE + Kikuchi-Kawai + Cugnon NN + Gaussian Glauber for clusters
};

enum class AngularMode : int {
    Isotropic = 0,           // Uniform cos(theta*) in [-1, 1]
    ForwardDiffractive = 1   // Cugnon / Glauber exp(B * t) angular distribution
};

struct CrossSectionConfig {
    CrossSectionModel model = CrossSectionModel::ParametricGlauber;
    AngularMode angular_mode = AngularMode::ForwardDiffractive;
    double sigma_const_mb = 40.0;        // Used when model == ConstantIsotropic
    double sigma_nn_max_mb = 250.0;      // Low-energy cap on NN cross section [mb]
    double excited_radius_scale = 1.0;   // Optional extra scaling on R_rms of excited clusters
};

// Smooth step S(x) = 3x^2 - 2x^3 on [0, 1]
HD_INLINE Real smoothstep01(Real x) {
    if (x <= Real(0)) return Real(0);
    if (x >= Real(1)) return Real(1);
    return x * x * (Real(3) - Real(2) * x);
}

// Nucleon-nucleon elastic cross section [mb] as a function of invariant relative gamma_rel:
//   gamma_rel = (p1 . p2) / (m1 * m2) >= 1.
HD_INLINE Real sigma_nn_elastic_mb(bool same_isospin, Real gamma_rel, Real sigma_max_mb) {
    using namespace phys;
    constexpr Real mN = static_cast<Real>(MASS_NUCLEON_AVG);
    constexpr Real hc = static_cast<Real>(HBARC);
    constexpr Real pi_r = static_cast<Real>(PI);
    constexpr Real fm2_mb = static_cast<Real>(FM2_TO_MB);
    if (gamma_rel <= Real(1) + Real(1.0e-6)) {
        return sigma_max_mb;
    }

    const Real p_lab = mN * std::sqrt(gamma_rel * gamma_rel - Real(1));
    const Real e_lab = mN * gamma_rel;
    const Real beta_lab = p_lab / e_lab;

    const Real p_cm_gev = mN * std::sqrt(Real(0.5) * (gamma_rel - Real(1)));
    const Real k_fm = p_cm_gev / hc;

    // 1. S-wave Effective Range Expansion [mb] (1 fm^2 = 10 mb)
    Real sigma_ere_mb = Real(0);
    if (same_isospin) {
        constexpr Real a_s = Real(-18.9);
        constexpr Real r_s = Real(2.75);
        const Real phase_term = Real(-1) / a_s + Real(0.5) * r_s * k_fm * k_fm;
        sigma_ere_mb = fm2_mb * (Real(2) * pi_r) / (k_fm * k_fm + phase_term * phase_term);
    } else {
        constexpr Real a_t = Real(5.424);
        constexpr Real r_t = Real(1.759);
        constexpr Real a_s = Real(-23.748);
        constexpr Real r_s = Real(2.750);
        const Real pt_t = Real(-1) / a_t + Real(0.5) * r_t * k_fm * k_fm;
        const Real pt_s = Real(-1) / a_s + Real(0.5) * r_s * k_fm * k_fm;
        const Real sig_t = (Real(3) * pi_r) / (k_fm * k_fm + pt_t * pt_t);
        const Real sig_s = pi_r / (k_fm * k_fm + pt_s * pt_s);
        sigma_ere_mb = fm2_mb * (sig_t + sig_s);
    }

    // 2. Kikuchi-Kawai / Metropolis nuclear energy parameterization [mb]
    const Real b_clamped = (beta_lab > Real(0.08)) ? beta_lab : Real(0.08);
    Real sigma_kk_mb = Real(0);
    if (same_isospin) {
        sigma_kk_mb = Real(10.63) / (b_clamped * b_clamped) - Real(29.92) / b_clamped + Real(42.9);
    } else {
        sigma_kk_mb = Real(34.10) / (b_clamped * b_clamped) - Real(82.20) / b_clamped + Real(82.2);
    }

    // 3. Relativistic Cugnon / PDG elastic parameterization (p_lab >= 0.8 GeV/c) [mb]
    const Real p_clamped = (p_lab > Real(0.5)) ? p_lab : Real(0.5);
    Real sigma_he_mb = Real(0);
    if (same_isospin) {
        sigma_he_mb = Real(30.9) / std::pow(p_clamped, Real(0.55)) - Real(20.0) / (p_clamped * p_clamped + Real(1.1)) + Real(5.2);
    } else {
        sigma_he_mb = Real(31.5) / std::pow(p_clamped, Real(0.52)) + Real(5.0);
    }

    const Real w_kk = smoothstep01((k_fm - Real(0.25)) * Real(1.0 / (0.42 - 0.25)));
    Real sigma_mb = (Real(1) - w_kk) * sigma_ere_mb + w_kk * sigma_kk_mb;

    const Real w_he = smoothstep01((p_lab - Real(0.70)) * Real(1.0 / (1.00 - 0.70)));
    sigma_mb = (Real(1) - w_he) * sigma_mb + w_he * sigma_he_mb;

    if (sigma_mb > sigma_max_mb) sigma_mb = sigma_max_mb;
    if (sigma_mb < Real(1)) sigma_mb = Real(1);
    return sigma_mb;
}

// Compute total elastic cross section sigma_el [fm^2] for any pair of species (id1, id2)
HD_INLINE Real elastic_cross_section_fm2(int id1, int id2,
                                         const PairKinematics& kin,
                                         const CrossSectionConfig& cfg) {
    using namespace phys;
    constexpr Real mb2fm2 = static_cast<Real>(MB_TO_FM2);
    constexpr Real pi_r = static_cast<Real>(PI);
    if (cfg.model == CrossSectionModel::ConstantIsotropic) {
        return static_cast<Real>(cfg.sigma_const_mb) * mb2fm2;
    }

    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);

    const int A1 = sp1.baryon_num;
    const int A2 = sp2.baryon_num;

    if (A1 == 0 && A2 == 0) {
        return Real(12.0) * mb2fm2;
    }
    if (A1 == 0 || A2 == 0) {
        const SpeciesInfo sp_b = (A1 == 0) ? sp2 : sp1;
        const SpeciesInfo sp_pi = (A1 == 0) ? sp1 : sp2;
        const Real sig_piN_bg_fm2 =
            (Real(10.0) / (Real(1) + Real(0.25) * kin.p_cm * kin.p_cm) + Real(2.0)) * mb2fm2;
        if (sp_b.baryon_num == 1) {
            return sig_piN_bg_fm2;
        }
        const Real Rb = static_cast<Real>(sp_b.rms_radius_fm) *
                        (sp_b.is_excited ? static_cast<Real>(cfg.excited_radius_scale) : Real(1));
        const Real Rpi = static_cast<Real>(sp_pi.rms_radius_fm);
        const Real a_sq = Real(2.0 / 3.0) * (Rb * Rb + Rpi * Rpi);
        const Real geom_area_fm2 = pi_r * a_sq;
        const Real opacity = (static_cast<Real>(sp_b.baryon_num) * sig_piN_bg_fm2) / geom_area_fm2;
        return geom_area_fm2 * std::log1p(opacity);
    }

    const Real sig_max = static_cast<Real>(cfg.sigma_nn_max_mb);
    if (A1 == 1 && A2 == 1) {
        if (is_delta(id1) || is_delta(id2)) {
            const Real s_same = sigma_nn_elastic_mb(true, kin.gamma_rel, sig_max);
            const Real s_diff = sigma_nn_elastic_mb(false, kin.gamma_rel, sig_max);
            return Real(0.5) * (s_same + s_diff) * mb2fm2;
        }
        const bool same_iso = (sp1.charge == sp2.charge);
        return sigma_nn_elastic_mb(same_iso, kin.gamma_rel, sig_max) * mb2fm2;
    }

    const Real sig_same_fm2 = sigma_nn_elastic_mb(true, kin.gamma_rel, sig_max) * mb2fm2;
    const Real sig_diff_fm2 = sigma_nn_elastic_mb(false, kin.gamma_rel, sig_max) * mb2fm2;

    Real Z1 = Real(0), N1 = Real(0), Z2 = Real(0), N2 = Real(0);
    get_effective_pn_weights(id1, Z1, N1);
    get_effective_pn_weights(id2, Z2, N2);
    const Real n_same = Z1 * Z2 + N1 * N2;
    const Real n_diff = Z1 * N2 + N1 * Z2;
    const Real ab_prod = static_cast<Real>(A1 * A2);
    const Real sig_nn_avg_fm2 = (n_same * sig_same_fm2 + n_diff * sig_diff_fm2) / ab_prod;

    const Real R1 = static_cast<Real>(sp1.rms_radius_fm) *
                    (sp1.is_excited ? static_cast<Real>(cfg.excited_radius_scale) : Real(1));
    const Real R2 = static_cast<Real>(sp2.rms_radius_fm) *
                    (sp2.is_excited ? static_cast<Real>(cfg.excited_radius_scale) : Real(1));

    const Real a_sq = Real(2.0 / 3.0) * (R1 * R1 + R2 * R2);
    const Real geom_area_fm2 = pi_r * a_sq;
    const Real opacity = (ab_prod * sig_nn_avg_fm2) / geom_area_fm2;

    return geom_area_fm2 * std::log1p(opacity);
}

// Sample CM scattering polar cosine cos(theta*) and azimuth phi* in [0, 2*pi)
HD_INLINE void sample_elastic_angles(int id1, int id2,
                                     const PairKinematics& kin,
                                     const CrossSectionConfig& cfg,
                                     PhiloxRng& rng,
                                     Real& cos_theta, Real& phi) {
    using namespace phys;
    phi = static_cast<Real>(TWO_PI) * rng.uniform_real();

    if (cfg.model == CrossSectionModel::ConstantIsotropic ||
        cfg.angular_mode == AngularMode::Isotropic) {
        cos_theta = Real(1) - Real(2) * rng.uniform_real();
        return;
    }

    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);

    Real B_gev_inv2 = Real(5.5);
    if (sp1.baryon_num > 1 || sp2.baryon_num > 1) {
        const Real R1 = static_cast<Real>(sp1.rms_radius_fm) *
                        (sp1.is_excited ? static_cast<Real>(cfg.excited_radius_scale) : Real(1));
        const Real R2 = static_cast<Real>(sp2.rms_radius_fm) *
                        (sp2.is_excited ? static_cast<Real>(cfg.excited_radius_scale) : Real(1));
        B_gev_inv2 += (R1 * R1 + R2 * R2) * static_cast<Real>(1.0 / (3.0 * HBARC_SQ));
    }

    const Real a = Real(2) * B_gev_inv2 * kin.p_cm * kin.p_cm;
    const Real u = rng.uniform_real();

    if (a < Real(1.0e-4)) {
        cos_theta = Real(1) - Real(2) * u;
    } else if (a > Real(40.0)) {
        cos_theta = Real(1) + std::log(Real(1) - u) / a;
        if (cos_theta < Real(-1)) cos_theta = Real(-1);
    } else {
        const Real one_minus_exp = -std::expm1(Real(-2) * a);
        cos_theta = Real(1) + std::log(Real(1) - u * one_minus_exp) / a;
    }

    if (id1 == id2 && rng.uniform_real() < Real(0.5)) {
        cos_theta = -cos_theta;
    }
}

}  // namespace transport
