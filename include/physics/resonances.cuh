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
    Real br_pole = Real(0); // Branching fraction at pole mass m_0 (sum = 1 over open branches)
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
    Real norm[NUM_SPECIES];
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
            spec.branches[0] = {PROTON, TRITON, 1, static_cast<Real>(0.55)};
            spec.branches[1] = {NEUTRON, HELION, 1, static_cast<Real>(0.45)};
            break;

        // 4He* I = 1 states above 23.85 MeV: d+d forbidden by isospin (I_d = 0)
        case HE4_2640:
        case HE4_2797:
            spec.n_branches = 2;
            spec.branches[0] = {PROTON, TRITON, 1, static_cast<Real>(0.50)};
            spec.branches[1] = {NEUTRON, HELION, 1, static_cast<Real>(0.50)};
            break;

        // 4He* I = 0 negative-parity states above d+d threshold (23.85 MeV): p+t, n+3He, d+d (L=1)
        case HE4_2425:
        case HE4_2528:
        case HE4_2595:
        case HE4_2837:
        case HE4_2839:
        case HE4_2864:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 1, static_cast<Real>(0.45)};
            spec.branches[1] = {NEUTRON, HELION, 1, static_cast<Real>(0.45)};
            spec.branches[2] = {DEUTERON, DEUTERON, 1, static_cast<Real>(0.10)};
            break;

        // 4He* I = 0 positive-parity 2+ states above d+d threshold: L=2 for N+A3, L=0 (S=2) for d+d
        case HE4_2742:
        case HE4_2867:
        case HE4_2989:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 2, static_cast<Real>(0.40)};
            spec.branches[1] = {NEUTRON, HELION, 2, static_cast<Real>(0.40)};
            spec.branches[2] = {DEUTERON, DEUTERON, 0, static_cast<Real>(0.20)};
            break;

        // 4He*(28.31, 1+, I=0): L=0 for N+A3 (S=1), L=2 for d+d (S=2)
        case HE4_2831:
            spec.n_branches = 3;
            spec.branches[0] = {PROTON, TRITON, 0, static_cast<Real>(0.45)};
            spec.branches[1] = {NEUTRON, HELION, 0, static_cast<Real>(0.45)};
            spec.branches[2] = {DEUTERON, DEUTERON, 2, static_cast<Real>(0.10)};
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
            spec.branches[0] = {NEUTRON, PION_ZERO,  1, static_cast<Real>(2.0 / 3.0)};
            spec.branches[1] = {PROTON,  PION_MINUS, 1, static_cast<Real>(1.0 / 3.0)};
            break;
        case DELTA_PLUS:
            spec.n_branches = 2;
            spec.branches[0] = {PROTON,  PION_ZERO,  1, static_cast<Real>(2.0 / 3.0)};
            spec.branches[1] = {NEUTRON, PION_PLUS,  1, static_cast<Real>(1.0 / 3.0)};
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
HD_INLINE Real resonance_min_mass(int res_id) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return static_cast<Real>(sp.pole_mass_gev);

    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    Real m_min = Real(1.0e9);
    for (int b = 0; b < dspec.n_branches; ++b) {
        const Real mth = static_cast<Real>(get_species_info(dspec.branches[b].d1).pole_mass_gev +
                                           get_species_info(dspec.branches[b].d2).pole_mass_gev);
        if (mth < m_min) m_min = mth;
    }
    return m_min;
}

// Upper support cutoff m_max [GeV] of the Breit-Wigner manifold (shared by 2<->1 and 2<->2)
HD_INLINE Real resonance_max_mass(int res_id) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return sp.pole_mass_gev;
    const Real span = (Real(10) * sp.width_gev > Real(0.120)) ? (Real(10) * sp.width_gev) : Real(0.120);
    return sp.pole_mass_gev + span;
}

// Blatt-Weisskopf barrier penetration factor B_L(q) normalized to -> 1 at q >> Lambda
HD_INLINE Real blatt_weisskopf_sq(int L, Real q_gev) {
    constexpr Real INV_LAM2 = Real(1.0 / (0.20 * 0.20));
    if (L <= 0) return Real(1);
    const Real x2 = (q_gev * q_gev) * INV_LAM2;
    if (L == 1) {
        return x2 / (Real(1) + x2);
    }
    const Real x4 = x2 * x2;
    return x4 / (Real(9) + Real(3) * x2 + x4);
}

// Nonsingular reduced partial width tilde_gamma_b(m) = Gamma_b(m) / p_cm(m) [dimensionless]
HD_INLINE Real branch_reduced_width_over_q(int res_id, int branch_idx, Real m) {
    const SpeciesInfo sp = get_species_info(res_id);
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    const DecayBranch br = dspec.branches[branch_idx];

    const Real ma = static_cast<Real>(get_species_info(br.d1).pole_mass_gev);
    const Real mb = static_cast<Real>(get_species_info(br.d2).pole_mass_gev);
    if (m <= ma + mb) return Real(0);

    const Real m0 = static_cast<Real>(sp.pole_mass_gev);
    const Real q0 = two_body_p_cm(m0, ma, mb);
    if (q0 <= Real(1.0e-12)) return Real(0);

    const Real q = two_body_p_cm(m, ma, mb);
    const Real bl_ratio = blatt_weisskopf_sq(br.L, q) / blatt_weisskopf_sq(br.L, q0);
    const Real gamma_b0 = static_cast<Real>(sp.width_gev * br.br_pole);

    return (gamma_b0 / q0) * (m0 / m) * bl_ratio;
}

HD_INLINE Real branch_partial_width_gev(int res_id, int branch_idx, Real m) {
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    const DecayBranch br = dspec.branches[branch_idx];
    const Real ma = static_cast<Real>(get_species_info(br.d1).pole_mass_gev);
    const Real mb = static_cast<Real>(get_species_info(br.d2).pole_mass_gev);
    const Real q = two_body_p_cm(m, ma, mb);
    if (q <= Real(0)) return Real(0);
    return q * branch_reduced_width_over_q(res_id, branch_idx, m);
}

HD_INLINE Real total_width_gev(int res_id, Real m) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return Real(0);
    const ResonanceDecaySpec dspec = get_decay_spec(res_id);
    Real g_tot = Real(0);
    for (int b = 0; b < dspec.n_branches; ++b) {
        g_tot += branch_partial_width_gev(res_id, b, m);
    }
    return g_tot;
}

// Unnormalized relativistic Breit-Wigner spectral density A_raw(m) [GeV^-1]
HD_FUNC Real spectral_function_raw(int res_id, Real m) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return Real(0);
    const Real m_min = resonance_min_mass(res_id);
    const Real m_max = resonance_max_mass(res_id);
    if (m <= m_min || m > m_max) return Real(0);

    const Real g_tot = total_width_gev(res_id, m);
    if (g_tot <= Real(0)) return Real(0);

    const Real m0 = static_cast<Real>(sp.pole_mass_gev);
    const Real dm2 = (m - m0) * (m + m0);
    const Real mg = m * g_tot;
    return static_cast<Real>(2.0 / phys::PI) * (m * mg) / (dm2 * dm2 + mg * mg);
}

// Normalized spectral function A_R(m) satisfying int_{m_min}^{m_max} A_R(m) dm = 1
HD_INLINE Real spectral_function(int res_id, Real m, const SpectralNormTable& norms) {
    const Real nrm = static_cast<Real>(norms.norm[res_id]);
    return (nrm > Real(0)) ? (spectral_function_raw(res_id, m) / nrm) : Real(0);
}

template <typename T>
HD_FUNC T phase_space_spectral_integral_quad(int res_id,
                                             T sqrt_s,
                                             T m_spec,
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
            tbl.norm[id] = Real(1);
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
                    sum += half * gl_w[q] * (spectral_function_raw(id, static_cast<Real>(mid - half * gl_x[q])) +
                                             spectral_function_raw(id, static_cast<Real>(mid + half * gl_x[q])));
                }
            }
        }
        tbl.norm[id] = static_cast<Real>((sum > 0.0) ? sum : 1.0);
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
template <typename T>
HD_FUNC T phase_space_spectral_integral_quad(int res_id,
                                             T sqrt_s,
                                             T m_spec,
                                             const SpectralNormTable& norms) {
    const T m_min = static_cast<T>(resonance_min_mass(res_id));
    const T m_hi_kin = sqrt_s - m_spec;
    constexpr T eps = (sizeof(T) == 4) ? T(1.0e-6) : static_cast<T>(1.0e-9);
    if (m_hi_kin <= m_min + eps) return T(0);

    const T m_max = static_cast<T>(resonance_max_mass(res_id));
    const T upper = (m_hi_kin < m_max) ? m_hi_kin : m_max;

    constexpr T gx[4] = {
        static_cast<T>(0.183434642495649805), static_cast<T>(0.525532409916328986),
        static_cast<T>(0.796666477413626740), static_cast<T>(0.960289856497536232)
    };
    constexpr T gw[4] = {
        static_cast<T>(0.362683783378361983), static_cast<T>(0.313706645877887287),
        static_cast<T>(0.222381034453374471), static_cast<T>(0.101228536290376259)
    };

    const T m0 = static_cast<T>(get_species_info(res_id).pole_mass_gev);
    const T split = (m0 > m_min + T(1.0e-5) && m0 < upper - T(1.0e-5))
                        ? m0
                        : (T(0.5) * (m_min + upper));

    T total = T(0);
    const T segs[3] = {m_min, split, upper};
    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        const T mid = T(0.5) * (segs[s_idx] + segs[s_idx + 1]);
        const T half = T(0.5) * (segs[s_idx + 1] - segs[s_idx]);
        if (half <= T(0)) continue;
        for (int q = 0; q < 4; ++q) {
            const T m_lo = mid - half * gx[q];
            const T m_up = mid + half * gx[q];
            const T f_lo = static_cast<T>(spectral_function(res_id, static_cast<Real>(m_lo), norms)) *
                           static_cast<T>(two_body_p_cm(static_cast<Real>(sqrt_s), static_cast<Real>(m_spec), static_cast<Real>(m_lo)));
            const T f_up = static_cast<T>(spectral_function(res_id, static_cast<Real>(m_up), norms)) *
                           static_cast<T>(two_body_p_cm(static_cast<Real>(sqrt_s), static_cast<Real>(m_spec), static_cast<Real>(m_up)));
            total += half * gw[q] * (f_lo + f_up);
        }
    }
    return total;
}

// O(1) LUT-accelerated phase_space_spectral_integral for NN -> N Delta, exact quadrature fallback otherwise
HD_FUNC Real phase_space_spectral_integral(int res_id,
                                           Real sqrt_s,
                                           Real m_spec,
                                           const SpectralNormTable& norms) {
    if (is_delta(res_id) && sqrt_s <= static_cast<Real>(SpectralNormTable::DELTA_S_MAX)) {
        if (sqrt_s <= static_cast<Real>(SpectralNormTable::DELTA_S_MIN)) return Real(0);
        const int d_idx = res_id - DELTA_MINUS;  // 0..3
        const int n_idx = (std::fabs(m_spec - static_cast<Real>(phys::MASS_PROTON)) < Real(1.0e-4))
                              ? 0
                              : ((std::fabs(m_spec - static_cast<Real>(phys::MASS_NEUTRON)) < Real(1.0e-4)) ? 4 : -1);
        if (n_idx >= 0 && norms.delta_ps_lut[n_idx + d_idx][SpectralNormTable::DELTA_LUT_N - 1] > 0.0f) {
            constexpr Real inv_ds = static_cast<Real>((SpectralNormTable::DELTA_LUT_N - 1) /
                                                      (SpectralNormTable::DELTA_S_MAX - SpectralNormTable::DELTA_S_MIN));
            const Real u = (sqrt_s - static_cast<Real>(SpectralNormTable::DELTA_S_MIN)) * inv_ds;
            int k = static_cast<int>(u);
            if (k >= SpectralNormTable::DELTA_LUT_N - 1) k = SpectralNormTable::DELTA_LUT_N - 2;
            const float f = static_cast<float>(u) - static_cast<float>(k);
            const float* row = norms.delta_ps_lut[n_idx + d_idx];
            return static_cast<Real>((1.0f - f) * row[k] + f * row[k + 1]);
        }
    }
    return phase_space_spectral_integral_quad<Real>(res_id, sqrt_s, m_spec, norms);
}

// Sample off-shell resonance mass m_R in [m_min, min(m_max, sqrt_s - m_spec)]
HD_FUNC Real sample_breit_wigner_2to2_mass(int res_id,
                                           Real sqrt_s,
                                           Real m_spec,
                                           PhiloxRng& rng) {
    const SpeciesInfo sp = get_species_info(res_id);
    if (!sp.is_excited) return static_cast<Real>(sp.pole_mass_gev);

    const Real m_min = resonance_min_mass(res_id);
    const Real m_max_supp = resonance_max_mass(res_id);
    const Real upper = ((sqrt_s - m_spec) < m_max_supp) ? (sqrt_s - m_spec) : m_max_supp;
    if (upper <= m_min + Real(1.0e-6)) return m_min + Real(0.5) * (upper - m_min);

    const Real m0 = static_cast<Real>(sp.pole_mass_gev);
    const Real w0 = m0 * static_cast<Real>(sp.width_gev);
    const Real th_min = std::atan(((m_min - m0) * (m_min + m0)) / w0);
    const Real th_max = std::atan(((upper - m0) * (upper + m0)) / w0);
    const Real p_max = two_body_p_cm(sqrt_s, m_spec, m_min);
    if (p_max <= Real(0)) return Real(0.5) * (m_min + upper);

    constexpr Real INV_RATIO_BOUND = Real(1.0 / 8.5);

    for (int iter = 0; iter < 128; ++iter) {
        const Real th = th_min + (th_max - th_min) * rng.uniform_real();
        const Real dm2_cand = w0 * std::tan(th);
        const Real m2 = m0 * m0 + dm2_cand;
        if (m2 <= m_min * m_min || m2 >= upper * upper) continue;
        const Real m = std::sqrt(m2);

        const Real g_tot = total_width_gev(res_id, m);
        if (g_tot <= Real(0)) continue;

        const Real dm2 = (m - m0) * (m + m0);
        const Real mg = m * g_tot;
        const Real bw_ratio = (mg / w0) * ((dm2 * dm2 + w0 * w0) / (dm2 * dm2 + mg * mg));
        const Real pcm_ratio = two_body_p_cm(sqrt_s, m_spec, m) / p_max;
        const Real accept = (bw_ratio * pcm_ratio) * INV_RATIO_BOUND;
        if (rng.uniform_real() < accept) return m;
    }
    return Real(0.5) * (m_min + upper);
}

// Compute nonsingular v_moller * sigma_{ab -> R}(sqrt_s) [fm^2] for 2 -> 1 resonance formation!
HD_FUNC Real formation_vmoller_sigma_fm2(int id1, int id2,
                                           const FourVector& p1,
                                           const FourVector& p2,
                                           Real sqrt_s,
                                           int res_id,
                                           const SpectralNormTable& norms) {
    const Real m_min = resonance_min_mass(res_id);
    const Real m_max = resonance_max_mass(res_id);
    if (sqrt_s <= m_min || sqrt_s > m_max) return Real(0);

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
    if (match_branch < 0) return Real(0);

    const Real A_val = spectral_function(res_id, sqrt_s, norms);
    if (A_val <= Real(0)) return Real(0);

    const Real red_gamma = branch_reduced_width_over_q(res_id, match_branch, sqrt_s);
    if (red_gamma <= Real(0)) return Real(0);

    const SpeciesInfo sp1 = get_species_info(id1);
    const SpeciesInfo sp2 = get_species_info(id2);
    const SpeciesInfo spR = get_species_info(res_id);

    const Real S_ab = (id1 == id2) ? Real(2) : Real(1);
    const Real spin_factor = static_cast<Real>(spR.degeneracy) /
                             static_cast<Real>(sp1.degeneracy * sp2.degeneracy);

    return (sqrt_s / (p1.t * p2.t)) * S_ab * spin_factor *
           static_cast<Real>(2.0 * phys::PI * phys::PI * phys::HBARC_SQ) * A_val * red_gamma;
}

}  // namespace transport
