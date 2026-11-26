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
HD_INLINE double smoothstep01(double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    return x * x * (3.0 - 2.0 * x);
}

// Nucleon-nucleon elastic cross section [mb] as a function of invariant relative gamma_rel:
//   gamma_rel = (p1 . p2) / (m1 * m2) >= 1.
// Bridges three physical regimes smoothly:
//   1. Very low energy (T_lab < 12 MeV): S-wave Effective Range Expansion (ERE)
//   2. Nuclear regime (12 MeV <= T_lab <= 320 MeV): Kikuchi-Kawai / Metropolis fit
//   3. Relativistic regime (p_lab > 0.85 GeV/c): PDG / Cugnon high-energy elastic fit
HD_INLINE double sigma_nn_elastic_mb(bool same_isospin, double gamma_rel, double sigma_max_mb) {
    using namespace phys;
    if (gamma_rel <= 1.0 + 1.0e-12) {
        return sigma_max_mb;
    }

    const double p_lab = MASS_NUCLEON_AVG * ::sqrt(gamma_rel * gamma_rel - 1.0);
    const double e_lab = MASS_NUCLEON_AVG * gamma_rel;
    const double beta_lab = p_lab / e_lab;

    // CM wave number k_cm [fm^-1] for two nucleons: s_NN = 2 m_N^2 (1 + gamma_rel)
    // p_cm^2 = m_N^2 * (gamma_rel - 1) / 2
    const double p_cm_gev = MASS_NUCLEON_AVG * ::sqrt(0.5 * (gamma_rel - 1.0));
    const double k_fm = p_cm_gev / HBARC;

    // 1. S-wave Effective Range Expansion [mb] (1 fm^2 = 10 mb)
    double sigma_ere_mb = 0.0;
    if (same_isospin) {
        // Nuclear pp/nn 1S0 scattering length & effective range (Coulomb-subtracted nuclear part)
        constexpr double a_s = -18.9;  // fm
        constexpr double r_s = 2.75;   // fm
        const double phase_term = -1.0 / a_s + 0.5 * r_s * k_fm * k_fm;
        sigma_ere_mb = FM2_TO_MB * (2.0 * PI) / (k_fm * k_fm + phase_term * phase_term);
    } else {
        // np: S=1 triplet (3/4) + S=0 singlet (1/4), distinguishable -> 4*pi per spin channel
        constexpr double a_t = 5.424;   // fm
        constexpr double r_t = 1.759;   // fm
        constexpr double a_s = -23.748; // fm
        constexpr double r_s = 2.750;   // fm
        const double pt_t = -1.0 / a_t + 0.5 * r_t * k_fm * k_fm;
        const double pt_s = -1.0 / a_s + 0.5 * r_s * k_fm * k_fm;
        const double sig_t = (3.0 * PI) / (k_fm * k_fm + pt_t * pt_t);
        const double sig_s = (1.0 * PI) / (k_fm * k_fm + pt_s * pt_s);
        sigma_ere_mb = FM2_TO_MB * (sig_t + sig_s);
    }

    // 2. Kikuchi-Kawai / Metropolis nuclear energy parameterization [mb]
    const double b_clamped = (beta_lab > 0.08) ? beta_lab : 0.08;
    double sigma_kk_mb = 0.0;
    if (same_isospin) {
        sigma_kk_mb = 10.63 / (b_clamped * b_clamped) - 29.92 / b_clamped + 42.9;
    } else {
        sigma_kk_mb = 34.10 / (b_clamped * b_clamped) - 82.20 / b_clamped + 82.2;
    }

    // 3. Relativistic Cugnon / PDG elastic parameterization (p_lab >= 0.8 GeV/c) [mb]
    const double p_clamped = (p_lab > 0.5) ? p_lab : 0.5;
    double sigma_he_mb = 0.0;
    if (same_isospin) {
        sigma_he_mb = 30.9 / ::pow(p_clamped, 0.55) - 20.0 / (p_clamped * p_clamped + 1.1) + 5.2;
    } else {
        sigma_he_mb = 31.5 / ::pow(p_clamped, 0.52) + 5.0;
    }

    // Smoothly blend ERE -> Kikuchi-Kawai around k_cm in [0.25, 0.42] fm^-1 (T_lab ~ 5 - 15 MeV)
    const double w_kk = smoothstep01((k_fm - 0.25) / (0.42 - 0.25));
    double sigma_mb = (1.0 - w_kk) * sigma_ere_mb + w_kk * sigma_kk_mb;

    // Smoothly blend Kikuchi-Kawai -> High-Energy Cugnon around p_lab in [0.70, 1.00] GeV/c
    const double w_he = smoothstep01((p_lab - 0.70) / (1.00 - 0.70));
    sigma_mb = (1.0 - w_he) * sigma_mb + w_he * sigma_he_mb;

    if (sigma_mb > sigma_max_mb) sigma_mb = sigma_max_mb;
    if (sigma_mb < 1.0) sigma_mb = 1.0;
    return sigma_mb;
}

// Compute total elastic cross section sigma_el [fm^2] for any pair of species (id1, id2)
// in our nucleon + light-cluster (ground & excited) catalog.
HD_INLINE double elastic_cross_section_fm2(int id1, int id2,
                                           const PairKinematics& kin,
                                           const CrossSectionConfig& cfg) {
    using namespace phys;
    if (cfg.model == CrossSectionModel::ConstantIsotropic) {
        return cfg.sigma_const_mb * MB_TO_FM2;
    }

    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);

    const int A1 = sp1.baryon_num;
    const int A2 = sp2.baryon_num;

    // Pion-Pion and Pion-Baryon/Cluster non-resonant elastic scattering
    // (Resonant P33 pi + N -> Delta -> pi + N is generated dynamically by 2 <-> 1 channels!)
    if (A1 == 0 && A2 == 0) {
        return 12.0 * MB_TO_FM2;
    }
    if (A1 == 0 || A2 == 0) {
        const SpeciesInfo sp_b = (A1 == 0) ? sp2 : sp1;
        const SpeciesInfo sp_pi = (A1 == 0) ? sp1 : sp2;
        const double sig_piN_bg_fm2 =
            (10.0 / (1.0 + 0.25 * kin.p_cm * kin.p_cm) + 2.0) * MB_TO_FM2;
        if (sp_b.baryon_num == 1) {
            return sig_piN_bg_fm2;
        }
        const double Rb = sp_b.rms_radius_fm * (sp_b.is_excited ? cfg.excited_radius_scale : 1.0);
        const double Rpi = sp_pi.rms_radius_fm;
        const double a_sq = (2.0 / 3.0) * (Rb * Rb + Rpi * Rpi);
        const double geom_area_fm2 = PI * a_sq;
        const double opacity = (static_cast<double>(sp_b.baryon_num) * sig_piN_bg_fm2) / geom_area_fm2;
        return geom_area_fm2 * ::log1p(opacity);
    }

    // Elementary Baryon-Baryon channel (p, n, Delta)
    if (A1 == 1 && A2 == 1) {
        if (is_delta(id1) || is_delta(id2)) {
            const double s_same = sigma_nn_elastic_mb(true, kin.gamma_rel, cfg.sigma_nn_max_mb);
            const double s_diff = sigma_nn_elastic_mb(false, kin.gamma_rel, cfg.sigma_nn_max_mb);
            return 0.5 * (s_same + s_diff) * MB_TO_FM2;
        }
        const bool same_iso = (sp1.charge == sp2.charge);
        return sigma_nn_elastic_mb(same_iso, kin.gamma_rel, cfg.sigma_nn_max_mb) * MB_TO_FM2;
    }

    // Baryon-Cluster or Cluster-Cluster (including Delta, d*, 4He*, 4H*, 4Li*):
    const double sig_same_fm2 =
        sigma_nn_elastic_mb(true, kin.gamma_rel, cfg.sigma_nn_max_mb) * MB_TO_FM2;
    const double sig_diff_fm2 =
        sigma_nn_elastic_mb(false, kin.gamma_rel, cfg.sigma_nn_max_mb) * MB_TO_FM2;

    double Z1 = 0.0, N1 = 0.0, Z2 = 0.0, N2 = 0.0;
    get_effective_pn_weights(id1, Z1, N1);
    get_effective_pn_weights(id2, Z2, N2);
    const double n_same = Z1 * Z2 + N1 * N2;
    const double n_diff = Z1 * N2 + N1 * Z2;
    const double ab_prod = static_cast<double>(A1 * A2);
    const double sig_nn_avg_fm2 = (n_same * sig_same_fm2 + n_diff * sig_diff_fm2) / ab_prod;

    // Effective RMS radii (with optional scaling for excited clusters)
    const double R1 = sp1.rms_radius_fm * (sp1.is_excited ? cfg.excited_radius_scale : 1.0);
    const double R2 = sp2.rms_radius_fm * (sp2.is_excited ? cfg.excited_radius_scale : 1.0);

    // Gaussian Glauber / Karol optical eikonal overlap width:
    //   a^2 = (2/3) * (R_1^2 + R_2^2)
    //   sigma_el(A, B) = pi * a^2 * ln(1 + A * B * sigma_NN_avg / (pi * a^2))
    // Approaches A * B * sigma_NN_avg in the transparent limit and scales geometrically
    // with cluster radii R1, R2 (capturing diffuse d* and breathing-mode 4He* enhancement)
    // in the black-disk / opaque limit.
    const double a_sq = (2.0 / 3.0) * (R1 * R1 + R2 * R2);
    const double geom_area_fm2 = PI * a_sq;
    const double opacity = (ab_prod * sig_nn_avg_fm2) / geom_area_fm2;

    return geom_area_fm2 * ::log1p(opacity);
}

// Sample CM scattering polar cosine cos(theta*) and azimuth phi* in [0, 2*pi)
HD_INLINE void sample_elastic_angles(int id1, int id2,
                                     const PairKinematics& kin,
                                     const CrossSectionConfig& cfg,
                                     PhiloxRng& rng,
                                     double& cos_theta, double& phi) {
    using namespace phys;
    phi = TWO_PI * rng.uniform();

    if (cfg.model == CrossSectionModel::ConstantIsotropic ||
        cfg.angular_mode == AngularMode::Isotropic) {
        cos_theta = 1.0 - 2.0 * rng.uniform();
        return;
    }

    // Diffractive / Cugnon exponential t-slope: dsigma/dt ~ exp(B * t), t = -2 p_cm^2 (1 - cos_theta*)
    // At low CM momentum (s-wave regime), 2 * B * p_cm^2 -> 0 and scattering becomes isotropic automatically.
    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);

    // Nucleon slope B_NN ~ 5.5 GeV^-2 + nuclear cluster form-factor slope (R1^2 + R2^2) / (3 * hbarc^2)
    double B_gev_inv2 = 5.5;
    if (sp1.baryon_num > 1 || sp2.baryon_num > 1) {
        const double R1 = sp1.rms_radius_fm * (sp1.is_excited ? cfg.excited_radius_scale : 1.0);
        const double R2 = sp2.rms_radius_fm * (sp2.is_excited ? cfg.excited_radius_scale : 1.0);
        B_gev_inv2 += (R1 * R1 + R2 * R2) / (3.0 * HBARC_SQ);
    }

    const double a = 2.0 * B_gev_inv2 * kin.p_cm * kin.p_cm;
    const double u = rng.uniform();

    if (a < 1.0e-4) {
        cos_theta = 1.0 - 2.0 * u;
    } else if (a > 50.0) {
        // Prevent exp(-2a) underflow at ultra-high momentum
        cos_theta = 1.0 + ::log(1.0 - u) / a;
        if (cos_theta < -1.0) cos_theta = -1.0;
    } else {
        const double one_minus_exp = -::expm1(-2.0 * a);  // 1 - exp(-2a)
        cos_theta = 1.0 + ::log(1.0 - u * one_minus_exp) / a;
    }

    // Symmetrize forward/backward hemisphere for identical species
    if (id1 == id2 && rng.uniform() < 0.5) {
        cos_theta = -cos_theta;
    }
}

}  // namespace transport
