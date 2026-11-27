#pragma once

#include <cmath>
#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/species.cuh"

namespace transport {

struct DecayBranch {
    int d1 = -1;
    int d2 = -1;
    int L = 0;            // Orbital angular momentum L in {0, 1, 2}
    double br_pole = 0.0; // Branching fraction at pole mass m_0 (sum = 1 over open branches)
};

struct ResonanceDecaySpec {
    int n_branches = 0;
    DecayBranch branches[3]{};
};

// Precomputed spectral normalization constants N_R = int_{m_min}^{m_max} A_raw(m) dm
// plus 6x96 O(1) Delta(1232) phase-space spectral integral tables for NN -> N Delta.
struct SpectralNormTable {
    static constexpr int DELTA_LUT_N = 96;
    static constexpr double DELTA_S_MIN = 2.010;
    static constexpr double DELTA_S_MAX = 3.650;
    double norm[NUM_SPECIES];
    // [0..3]: (Delta-, Delta0, Delta+, Delta++) + p; [4..5]: (Delta0, Delta+) + n; [6]: (Delta++) + n; [7]: (Delta-) + n
    float delta_ps_lut[8][DELTA_LUT_N]{};
};

// Map every unstable species R to its physical 2-body decay branches (TUNL / isospin rules)
HD_INLINE constexpr ResonanceDecaySpec get_decay_spec(int res_id) {
    ResonanceDecaySpec spec{};
    switch (res_id) {
        // A = 2 spin-singlet (0+, I = 1) isospin triplet -> s-wave NN
        case D_STAR_NN:
            spec.n_branches = 1;
            spec.branches[0] = {NEUTRON, NEUTRON, 0, 1.0};
            break;
        case D_STAR_PN:
            spec.n_branches = 1;
            spec.branches[0] = {PROTON, NEUTRON, 0, 1.0};
            break;
        case D_STAR_PP:
            spec.n_branches = 1;
            spec.branches[0] = {PROTON, PROTON, 0, 1.0};
            break;

        // 4He*(20.21, 0+, I=0): sits above p+t (19.81 MeV) and below n+3He (20.58 MeV) -> 100% p+t (L=0)
        case HE4_2021:
            spec.n_branches = 1;
            spec.branches[0] = {PROTON, TRITON, 0, 1.0};
            break;

        // 4He* states between 20.58 MeV and d+d threshold (23.85 MeV): p+t and n+3He open
        case HE4_2101:
        case HE4_2184:
        case HE4_2333:  // I = 1
        case HE4_2364:  // I = 1
            spec.n_branches = 2;
            spec.branches[0] = {PROTON, TRITON, 1, 0.55};
            spec.branches[1] = {NEUTRON, HELION, 1, 0.45};
            break;

        // 4He* I = 1 states above 23.85 MeV: d+d forbidden by isospin (I_d = 0)
        case HE4_2640:
        case HE4_2797:
            spec.n_branches = 2;
            spec.branches[0] = {PROTON, TRITON, 1, 0.50};
            spec.branches[1] = {NEUTRON, HELION, 1, 0.50};
            break;

        // 4He* I = 0 negative-parity states above d+d threshold (23.85 MeV): p+t, n+3He, d+d (L=1)
        case HE4_2425:
        case HE4_2528:
        case HE4_2595:
        case HE4_2837:
        case HE4_2839:
        case HE4_2864:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 1, 0.45};
            spec.branches[1] = {NEUTRON, HELION, 1, 0.45};
            spec.branches[2] = {DEUTERON, DEUTERON, 1, 0.10};
            break;

        // 4He* I = 0 positive-parity 2+ states above d+d threshold: L=2 for N+A3, L=0 (S=2) for d+d
        case HE4_2742:
        case HE4_2867:
        case HE4_2989:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 2, 0.40};
            spec.branches[1] = {NEUTRON, HELION, 2, 0.40};
            spec.branches[2] = {DEUTERON, DEUTERON, 0, 0.20};
            break;

        // 4He*(28.31, 1+, I=0): L=0 for N+A3 (S=1), L=2 for d+d (S=2)
        case HE4_2831:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 0, 0.45};
            spec.branches[1] = {NEUTRON, HELION, 0, 0.45};
            spec.branches[2] = {DEUTERON, DEUTERON, 2, 0.10};
            break;

        // A = 4, I = 1 isobaric resonances 4H*(2-, 1-, 0-) -> n + t, 4Li*(2-, 1-, 0-) -> p + 3He (p-wave)
        case H4_RES:
        case H4_1M_350:
        case H4_0M_527:
        case H4_1M_602:
            spec.n_branches = 1;
            spec.branches[0] = {NEUTRON, TRITON, 1, 1.0};
            break;
        case LI4_RES:
        case LI4_1M_439:
        case LI4_0M_615:
        case LI4_1M_692:
            spec.n_branches = 1;
            spec.branches[0] = {PROTON, HELION, 1, 1.0};
            break;

        // A = 5 TUNL resonances: 5H, 5He, 5Li
        case H5_GS:
            spec.n_branches = 1;
            spec.branches[0] = {TRITON, D_STAR_NN, 0, 1.0};
            break;
        case HE5_GS:
        case HE5_127:
            spec.n_branches = 1;
            spec.branches[0] = {ALPHA, NEUTRON, 1, 1.0};
            break;
        case HE5_1684:
            spec.n_branches = 2;
            spec.branches[0] = {ALPHA,    NEUTRON, 2, 0.50};
            spec.branches[1] = {DEUTERON, TRITON,  0, 0.50};
            break;
        case LI5_GS:
        case LI5_149:
            spec.n_branches = 1;
            spec.branches[0] = {ALPHA, PROTON, 1, 1.0};
            break;
        case LI5_1687:
            spec.n_branches = 2;
            spec.branches[0] = {ALPHA,    PROTON, 2, 0.50};
            spec.branches[1] = {DEUTERON, HELION, 0, 0.50};
            break;

        // A = 6 TUNL resonances: 6H, 6He*(2+), 6Li*, 6Be
        case H6_GS:
            spec.n_branches = 1;
            spec.branches[0] = {H5_GS, NEUTRON, 1, 1.0};
            break;
        case HE6_180:
            spec.n_branches = 1;
            spec.branches[0] = {HE5_GS, NEUTRON, 1, 1.0};
            break;
        case LI6_219:
        case LI6_431:
            spec.n_branches = 1;
            spec.branches[0] = {ALPHA, DEUTERON, 2, 1.0};
            break;
        case LI6_537:
            spec.n_branches = 1;
            spec.branches[0] = {HE5_GS, PROTON, 1, 1.0};
            break;
        case LI6_565:
            spec.n_branches = 1;
            spec.branches[0] = {ALPHA, DEUTERON, 0, 1.0};
            break;
        case BE6_GS:
            spec.n_branches = 1;
            spec.branches[0] = {ALPHA, D_STAR_PP, 0, 1.0};
            break;
        case BE6_167:
            spec.n_branches = 2;
            spec.branches[0] = {LI5_GS, PROTON,    1, 0.50};
            spec.branches[1] = {ALPHA,  D_STAR_PP, 2, 0.50};
            break;

        // Delta(1232) (J^P = 3/2+, I = 3/2) -> N + pi (p-wave, L = 1) with exact Clebsch-Gordan weights
        case DELTA_MINUS:
            spec.n_branches = 1;
            spec.branches[0] = {NEUTRON, PION_MINUS, 1, 1.0};
            break;
        case DELTA_ZERO:
            spec.n_branches = 2;
            spec.branches[0] = {NEUTRON, PION_ZERO,  1, 2.0 / 3.0};
            spec.branches[1] = {PROTON,  PION_MINUS, 1, 1.0 / 3.0};
            break;
        case DELTA_PLUS:
            spec.n_branches = 2;
            spec.branches[0] = {PROTON,  PION_ZERO,  1, 2.0 / 3.0};
            spec.branches[1] = {NEUTRON, PION_PLUS,  1, 1.0 / 3.0};
            break;
        case DELTA_PLUSPLUS:
            spec.n_branches = 1;
            spec.branches[0] = {PROTON, PION_PLUS, 1, 1.0};
            break;

        default:
            spec.n_branches = 0;
            break;
    }
    return spec;
}

// Minimum threshold mass m_min [GeV] across all decay branches of resonance res_id
HD_INLINE double resonance_min_mass(int res_id) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return sp.pole_mass_gev;

    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    double m_min = 1.0e9;
    for (int b = 0; b < dspec.n_branches; ++b) {
        const double mth = get_species_info(dspec.branches[b].d1).pole_mass_gev +
                           get_species_info(dspec.branches[b].d2).pole_mass_gev;
        if (mth < m_min) m_min = mth;
    }
    return m_min;
}

// Upper support cutoff m_max [GeV] of the Breit-Wigner manifold (shared by 2<->1 and 2<->2)
HD_INLINE double resonance_max_mass(int res_id) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return sp.pole_mass_gev;
    const double span = (10.0 * sp.width_gev > 0.120) ? (10.0 * sp.width_gev) : 0.120;
    return sp.pole_mass_gev + span;
}

// Blatt-Weisskopf barrier penetration factor B_L(q) normalized to -> 1 at q >> Lambda
HD_INLINE double blatt_weisskopf_sq(int L, double q_gev) {
    constexpr double LAMBDA = 0.20;  // GeV/c (~1 fm^-1 nuclear interaction radius)
    if (L <= 0) return 1.0;
    const double x2 = (q_gev * q_gev) / (LAMBDA * LAMBDA);
    if (L == 1) {
        return x2 / (1.0 + x2);
    }
    const double x4 = x2 * x2;
    return x4 / (9.0 + 3.0 * x2 + x4);
}

// Nonsingular reduced partial width tilde_gamma_b(m) = Gamma_b(m) / p_cm(m) [dimensionless]
// Eliminates 0/0 threshold division in v_moller * sigma_{ab -> R}(m)!
HD_INLINE double branch_reduced_width_over_q(int res_id, int branch_idx, double m) {
    const SpeciesInfo sp = get_species_info(res_id);
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    const DecayBranch br = dspec.branches[branch_idx];

    const double ma = get_species_info(br.d1).pole_mass_gev;
    const double mb = get_species_info(br.d2).pole_mass_gev;
    if (m <= ma + mb) return 0.0;

    const double m0 = sp.pole_mass_gev;
    const double q0 = two_body_p_cm(m0, ma, mb);
    if (q0 <= 1.0e-12) return 0.0;

    const double q = two_body_p_cm(m, ma, mb);
    const double bl_ratio = blatt_weisskopf_sq(br.L, q) / blatt_weisskopf_sq(br.L, q0);
    const double gamma_b0 = sp.width_gev * br.br_pole;

    return (gamma_b0 / q0) * (m0 / m) * bl_ratio;
}

HD_INLINE double branch_partial_width_gev(int res_id, int branch_idx, double m) {
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    const DecayBranch br = dspec.branches[branch_idx];
    const double ma = get_species_info(br.d1).pole_mass_gev;
    const double mb = get_species_info(br.d2).pole_mass_gev;
    const double q = two_body_p_cm(m, ma, mb);
    if (q <= 0.0) return 0.0;
    return q * branch_reduced_width_over_q(res_id, branch_idx, m);
}

HD_INLINE double total_width_gev(int res_id, double m) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return 0.0;
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    double g_tot = 0.0;
    for (int b = 0; b < dspec.n_branches; ++b) {
        g_tot += branch_partial_width_gev(res_id, b, m);
    }
    return g_tot;
}

// Unnormalized relativistic Breit-Wigner spectral density A_raw(m) [GeV^-1]
HD_INLINE double spectral_function_raw(int res_id, double m) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return 0.0;
    const double m_min = resonance_min_mass(res_id);
    const double m_max = resonance_max_mass(res_id);
    if (m <= m_min || m > m_max) return 0.0;

    const double g_tot = total_width_gev(res_id, m);
    if (g_tot <= 0.0) return 0.0;

    const double m0 = sp.pole_mass_gev;
    const double dm2 = m * m - m0 * m0;
    const double mg = m * g_tot;
    return (2.0 / phys::PI) * (m * mg) / (dm2 * dm2 + mg * mg);
}

// Normalized spectral function A_R(m) satisfying int_{m_min}^{m_max} A_R(m) dm = 1
HD_INLINE double spectral_function(int res_id, double m, const SpectralNormTable& norms) {
    const double nrm = norms.norm[res_id];
    return (nrm > 0.0) ? (spectral_function_raw(res_id, m) / nrm) : 0.0;
}

HD_INLINE double phase_space_spectral_integral_quad(int res_id,
                                                    double sqrt_s,
                                                    double m_spec,
                                                    const SpectralNormTable& norms);

// Build exact SpectralNormTable on host using composite 8-point Gauss-Legendre quadrature
inline SpectralNormTable compute_spectral_norm_table() {
    SpectralNormTable tbl{};
    constexpr double gl_x[4] = {
        0.183434642495649805, 0.525532409916328986,
        0.796666477413626740, 0.960289856497536232
    };
    constexpr double gl_w[4] = {
        0.362683783378361983, 0.313706645877887287,
        0.222381034453374471, 0.101228536290376259
    };

    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        if (!sp.is_excited) {
            tbl.norm[id] = 1.0;
            continue;
        }
        const double m_min = resonance_min_mass(id);
        const double m_max = resonance_max_mass(id);
        // Split integration at pole mass m0 to resolve even sub-MeV widths to <1e-12 accuracy
        const double m0 = sp.pole_mass_gev;
        const double w_core = 6.0 * sp.width_gev;
        const double b1 = ((m0 - w_core) > m_min) ? (m0 - w_core) : (0.5 * (m_min + m0));
        const double b2 = (m0 > m_min && m0 < m_max) ? m0 : (0.5 * (m_min + m_max));
        const double b3 = ((m0 + w_core) < m_max) ? (m0 + w_core) : (0.5 * (m0 + m_max));
        const double bounds[5] = {m_min, b1, b2, b3, m_max};

        double sum = 0.0;
        for (int seg = 0; seg < 4; ++seg) {
            constexpr int N_SUB = 96;
            const double a = bounds[seg];
            const double b = bounds[seg + 1];
            if (b <= a) continue;
            const double h = (b - a) / static_cast<double>(N_SUB);
            for (int k = 0; k < N_SUB; ++k) {
                const double mid = a + (k + 0.5) * h;
                const double half = 0.5 * h;
                for (int q = 0; q < 4; ++q) {
                    sum += half * gl_w[q] * (spectral_function_raw(id, mid - half * gl_x[q]) +
                                             spectral_function_raw(id, mid + half * gl_x[q]));
                }
            }
        }
        tbl.norm[id] = (sum > 0.0) ? sum : 1.0;
    }

    // Precompute O(1) Delta(1232) phase-space spectral integral tables for NN -> N Delta
    constexpr double ds = (SpectralNormTable::DELTA_S_MAX - SpectralNormTable::DELTA_S_MIN) /
                          (SpectralNormTable::DELTA_LUT_N - 1);
    for (int d = 0; d < 4; ++d) {
        const int res_id = DELTA_MINUS + d;
        for (int k = 0; k < SpectralNormTable::DELTA_LUT_N; ++k) {
            const double sqs = SpectralNormTable::DELTA_S_MIN + k * ds;
            tbl.delta_ps_lut[0 + d][k] =
                static_cast<float>(phase_space_spectral_integral_quad(res_id, sqs, phys::MASS_PROTON, tbl));
            tbl.delta_ps_lut[4 + d][k] =
                static_cast<float>(phase_space_spectral_integral_quad(res_id, sqs, phys::MASS_NEUTRON, tbl));
        }
    }
    return tbl;
}

// Exact 16-point Gauss-Legendre integral I_R(s) = int_{m_min}^{min(m_max, sqrt_s - m_spec)} dm A_R(m) p_cm(s; m_spec, m)
HD_INLINE double phase_space_spectral_integral_quad(int res_id,
                                                    double sqrt_s,
                                                    double m_spec,
                                                    const SpectralNormTable& norms) {
    const double m_min = resonance_min_mass(res_id);
    const double m_hi_kin = sqrt_s - m_spec;
    if (m_hi_kin <= m_min + 1.0e-9) return 0.0;

    const double m_max = resonance_max_mass(res_id);
    const double upper = (m_hi_kin < m_max) ? m_hi_kin : m_max;

    constexpr double gx[4] = {
        0.183434642495649805, 0.525532409916328986,
        0.796666477413626740, 0.960289856497536232
    };
    constexpr double gw[4] = {
        0.362683783378361983, 0.313706645877887287,
        0.222381034453374471, 0.101228536290376259
    };

    const double m0 = get_species_info(res_id).pole_mass_gev;
    const double split = (m0 > m_min + 1.0e-5 && m0 < upper - 1.0e-5)
                             ? m0
                             : (0.5 * (m_min + upper));

    double total = 0.0;
    const double segs[3] = {m_min, split, upper};
    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        const double mid = 0.5 * (segs[s_idx] + segs[s_idx + 1]);
        const double half = 0.5 * (segs[s_idx + 1] - segs[s_idx]);
        if (half <= 0.0) continue;
        for (int q = 0; q < 4; ++q) {
            const double m_lo = mid - half * gx[q];
            const double m_up = mid + half * gx[q];
            const double f_lo = spectral_function(res_id, m_lo, norms) *
                                two_body_p_cm(sqrt_s, m_spec, m_lo);
            const double f_up = spectral_function(res_id, m_up, norms) *
                                two_body_p_cm(sqrt_s, m_spec, m_up);
            total += half * gw[q] * (f_lo + f_up);
        }
    }
    return total;
}

// O(1) LUT-accelerated phase_space_spectral_integral for NN -> N Delta, exact quadrature fallback otherwise
HD_INLINE double phase_space_spectral_integral(int res_id,
                                               double sqrt_s,
                                               double m_spec,
                                               const SpectralNormTable& norms) {
    if (is_delta(res_id) && sqrt_s <= SpectralNormTable::DELTA_S_MAX) {
        if (sqrt_s <= SpectralNormTable::DELTA_S_MIN) return 0.0;
        const int d_idx = res_id - DELTA_MINUS;  // 0..3
        const int n_idx = (std::fabs(m_spec - phys::MASS_PROTON) < 1.0e-5)
                              ? 0
                              : ((std::fabs(m_spec - phys::MASS_NEUTRON) < 1.0e-5) ? 4 : -1);
        if (n_idx >= 0 && norms.delta_ps_lut[n_idx + d_idx][SpectralNormTable::DELTA_LUT_N - 1] > 0.0f) {
            constexpr double inv_ds = (SpectralNormTable::DELTA_LUT_N - 1) /
                                      (SpectralNormTable::DELTA_S_MAX - SpectralNormTable::DELTA_S_MIN);
            const double u = (sqrt_s - SpectralNormTable::DELTA_S_MIN) * inv_ds;
            int k = static_cast<int>(u);
            if (k >= SpectralNormTable::DELTA_LUT_N - 1) k = SpectralNormTable::DELTA_LUT_N - 2;
            const float f = static_cast<float>(u - k);
            const float* row = norms.delta_ps_lut[n_idx + d_idx];
            return static_cast<double>((1.0f - f) * row[k] + f * row[k + 1]);
        }
    }
    return phase_space_spectral_integral_quad(res_id, sqrt_s, m_spec, norms);
}

// Sample off-shell resonance mass m_R in [m_min, min(m_max, sqrt_s - m_spec)] from
// target density proportional to A_R(m_R) * p_cm(sqrt_s; m_spec, m_R) via Lorentzian importance sampling.
HD_INLINE double sample_breit_wigner_2to2_mass(int res_id,
                                               double sqrt_s,
                                               double m_spec,
                                               PhiloxRng& rng) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return sp.pole_mass_gev;

    const double m_min = resonance_min_mass(res_id);
    const double m_max_supp = resonance_max_mass(res_id);
    const double upper = ((sqrt_s - m_spec) < m_max_supp) ? (sqrt_s - m_spec) : m_max_supp;
    if (upper <= m_min + 1.0e-8) return m_min + 0.5 * (upper - m_min);

    const double m0 = sp.pole_mass_gev;
    const double w0 = m0 * sp.width_gev;
    const double th_min = ::atan((m_min * m_min - m0 * m0) / w0);
    const double th_max = ::atan((upper * upper - m0 * m0) / w0);
    const double p_max = two_body_p_cm(sqrt_s, m_spec, m_min);
    if (p_max <= 0.0) return 0.5 * (m_min + upper);

    // Safe upper bound on ratio (A_raw(m) dm / Lorentzian_proposal(m^2) dm^2) * (p_cm / p_max)
    // across both narrow nuclear states and broad p-wave Delta(1232) up to m_max
    constexpr double RATIO_BOUND = 8.5;

    for (int iter = 0; iter < 128; ++iter) {
        const double th = th_min + (th_max - th_min) * rng.uniform();
        const double m2 = m0 * m0 + w0 * ::tan(th);
        if (m2 <= m_min * m_min || m2 >= upper * upper) continue;
        const double m = ::sqrt(m2);

        const double g_tot = total_width_gev(res_id, m);
        if (g_tot <= 0.0) continue;

        const double dm2 = m2 - m0 * m0;
        const double mg = m * g_tot;
        // Target / Proposal ratio: (mg / w0) * ((dm2^2 + w0^2) / (dm2^2 + mg^2)) * (p_cm / p_max)
        const double bw_ratio = (mg / w0) * ((dm2 * dm2 + w0 * w0) / (dm2 * dm2 + mg * mg));
        const double pcm_ratio = two_body_p_cm(sqrt_s, m_spec, m) / p_max;
        const double accept = (bw_ratio * pcm_ratio) / RATIO_BOUND;
        if (rng.uniform() < accept) return m;
    }
    return 0.5 * (m_min + upper);
}

// Compute nonsingular v_moller * sigma_{ab -> R}(sqrt_s) [fm^2] for 2 -> 1 resonance formation!
// Obeys exact microscopic detailed balance with R(m) -> a + b decay at every invariant mass m = sqrt_s:
//   v_moller * sigma_{ab -> R}(m) = (m / (E_a E_b)) * S_ab * (g_R / (g_a g_b)) * 2 * pi^2 * (hbar*c)^2
//                                   * A_R(m) * [Gamma_{R -> ab}(m) / p_cm(m)]
HD_INLINE double formation_vmoller_sigma_fm2(int id1, int id2,
                                             const FourVector& p1,
                                             const FourVector& p2,
                                             double sqrt_s,
                                             int res_id,
                                             const SpectralNormTable& norms) {
    const double m_min = resonance_min_mass(res_id);
    const double m_max = resonance_max_mass(res_id);
    if (sqrt_s <= m_min || sqrt_s > m_max) return 0.0;

    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    int match_branch = -1;
    for (int b = 0; b < dspec.n_branches; ++b) {
        const int d1 = dspec.branches[b].d1;
        const int d2 = dspec.branches[b].d2;
        if ((id1 == d1 && id2 == d2) || (id1 == d2 && id2 == d1)) {
            match_branch = b;
            break;
        }
    }
    if (match_branch < 0) return 0.0;

    const double A_val = spectral_function(res_id, sqrt_s, norms);
    if (A_val <= 0.0) return 0.0;

    const double red_gamma = branch_reduced_width_over_q(res_id, match_branch, sqrt_s);
    if (red_gamma <= 0.0) return 0.0;

    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);
    const SpeciesInfo spR = get_species_info(res_id);

    const double S_ab = (id1 == id2) ? 2.0 : 1.0;
    const double spin_factor = static_cast<double>(spR.degeneracy) /
                               static_cast<double>(sp1.degeneracy * sp2.degeneracy);

    return (sqrt_s / (p1.t * p2.t)) * S_ab * spin_factor *
           (2.0 * phys::PI * phys::PI * phys::HBARC_SQ) * A_val * red_gamma;
}

}  // namespace transport
