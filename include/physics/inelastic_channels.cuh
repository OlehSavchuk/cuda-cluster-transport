#pragma once

#include <cmath>
#include "backend/cuda_compat.cuh"
#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/resonances.cuh"
#include "physics/species.cuh"

namespace transport {

// Symmetric reversible 2 <-> 2 binary inelastic channel: a + b <-> c + d
// Supports both:
//   - On-shell nuclear rearrangement / stripping / charge-exchange (all 4 species stable)
//   - Off-shell excitation / de-excitation & deuteron catalysis (one species in (c, d) is
//     a Breit-Wigner resonance R with spectral density A_R(m_R))
// By parameterizing the invariant reduced matrix-element strength Sigma_0(s) [fm^2 * GeV^2]
// symmetrically in s, microscopic detailed balance is 100% exact by construction!
struct BinaryInelasticChannel {
    int a;
    int b;
    int c;
    int d;
    double strength_fm2_gev2;  // Symmetric coupling constant Sigma_scale [fm^2 * GeV^2]
    double form_scale_gev2;    // High-energy form-factor falloff scale Lambda_s^2 [GeV^2]
};

constexpr int NUM_INELASTIC_2TO2_CHANNELS = 62;

HD_INLINE constexpr BinaryInelasticChannel get_inelastic_2to2_channel(int idx) {
    // Channels 0..5:   Ground-state rearrangement, fusion/breakup & charge-exchange
    // Channels 6..9:   Deuteron catalysis via spin-singlet d* triplet (N + d <-> N + d*)
    // Channels 10..43: Proton/neutron inelastic excitation of 4He <-> 4He* (17 TUNL levels)
    // Channels 44..45: Isobaric charge-exchange excitation N + 4He <-> N' + 4H* / 4Li*
    // Channels 46..51: Isobar excitation/absorption N + N <-> N + Delta(1232) (I=1 Clebsch-Gordan 3:1:1:1)
    // Channels 52..54: Two-nucleon pion fusion / deuteron pion absorption N + N <-> d + pi (I=1)
    // Channels 55..56: Non-resonant pion-nucleon charge exchange pi + N <-> pi0 + N'
    // Channels 57..61: A = 6 bound-state transfer (t/3He + 4He <-> N + 6Li/6He) & 6Li*(3+) catalysis
    constexpr BinaryInelasticChannel table[NUM_INELASTIC_2TO2_CHANNELS] = {
        // 1. Ground-state A <= 4 transfer & charge-exchange chemistry
        {PROTON,   TRITON,   NEUTRON,  HELION,   18.0, 1.5},  // p + t <-> n + 3He
        {DEUTERON, DEUTERON, PROTON,   TRITON,   22.0, 1.8},  // d + d <-> p + t
        {DEUTERON, DEUTERON, NEUTRON,  HELION,   22.0, 1.8},  // d + d <-> n + 3He
        {DEUTERON, TRITON,   NEUTRON,  ALPHA,    36.0, 2.0},  // d + t <-> n + 4He
        {DEUTERON, HELION,   PROTON,   ALPHA,    36.0, 2.0},  // d + 3He <-> p + 4He
        {TRITON,   HELION,   DEUTERON, ALPHA,    24.0, 2.2},  // t + 3He <-> d + 4He

        // 2. Deuteron catalysis (Oliinychenko et al.): N + d <-> N + d*(0+)
        {PROTON,   DEUTERON, PROTON,   D_STAR_PN, 48.0, 1.2}, // p + d <-> p + d*(0+)
        {PROTON,   DEUTERON, NEUTRON,  D_STAR_PP, 48.0, 1.2}, // p + d <-> n + pp*(0+)
        {NEUTRON,  DEUTERON, NEUTRON,  D_STAR_PN, 48.0, 1.2}, // n + d <-> n + d*(0+)
        {NEUTRON,  DEUTERON, PROTON,   D_STAR_NN, 48.0, 1.2}, // n + d <-> p + nn*(0+)

        // 3. Proton-induced excitation / de-excitation: p + 4He <-> p + 4He*
        {PROTON, ALPHA, PROTON, HE4_2021, 16.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2101, 14.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2184, 12.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2333, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2364, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2425, 11.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2528, 11.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2595, 11.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2640, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2742, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2797, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2831, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2837, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2839, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2864, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2867, 10.0, 2.0},
        {PROTON, ALPHA, PROTON, HE4_2989, 10.0, 2.0},

        // 4. Neutron-induced excitation / de-excitation: n + 4He <-> n + 4He*
        {NEUTRON, ALPHA, NEUTRON, HE4_2021, 16.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2101, 14.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2184, 12.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2333, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2364, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2425, 11.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2528, 11.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2595, 11.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2640, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2742, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2797, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2831, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2837, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2839, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2864, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2867, 10.0, 2.0},
        {NEUTRON, ALPHA, NEUTRON, HE4_2989, 10.0, 2.0},

        // 5. Charge-exchange A = 4 resonance excitation
        {NEUTRON, ALPHA, PROTON,  H4_RES,   12.0, 2.0},
        {PROTON,  ALPHA, NEUTRON, LI4_RES,  12.0, 2.0},

        // 6. Nucleon-nucleon Delta(1232) excitation & absorption: N + N <-> N + Delta (I_tot = 1)
        //    Clebsch-Gordan ratios 3/4 : 1/4 : 1/4 : 1/4 (VerWest-Arndt / Huber-Aichelin)
        {PROTON,  PROTON,  NEUTRON, DELTA_PLUSPLUS, 6.3, 1.4},  // p + p <-> n + Delta++
        {PROTON,  PROTON,  PROTON,  DELTA_PLUS,     2.1, 1.4},  // p + p <-> p + Delta+
        {PROTON,  NEUTRON, PROTON,  DELTA_ZERO,     2.1, 1.4},  // p + n <-> p + Delta0
        {PROTON,  NEUTRON, NEUTRON, DELTA_PLUS,     2.1, 1.4},  // p + n <-> n + Delta+
        {NEUTRON, NEUTRON, PROTON,  DELTA_MINUS,    6.3, 1.4},  // n + n <-> p + Delta-
        {NEUTRON, NEUTRON, NEUTRON, DELTA_ZERO,     2.1, 1.4},  // n + n <-> n + Delta0

        // 7. Two-nucleon pion fusion / deuteron pion absorption: N + N <-> d + pi (I_tot = 1)
        {PROTON,  PROTON,  DEUTERON, PION_PLUS,     1.8, 0.6},  // p + p <-> d + pi+
        {PROTON,  NEUTRON, DEUTERON, PION_ZERO,     0.9, 0.6},  // p + n <-> d + pi0
        {NEUTRON, NEUTRON, DEUTERON, PION_MINUS,    1.8, 0.6},  // n + n <-> d + pi-

        // 8. Non-resonant pion-nucleon charge-exchange background
        {PION_MINUS, PROTON,  PION_ZERO, NEUTRON,   1.2, 0.8},  // pi- + p <-> pi0 + n
        {PION_PLUS,  NEUTRON, PION_ZERO, PROTON,    1.2, 0.8},  // pi+ + n <-> pi0 + p

        // 9. A = 6 ground-state synthesis & 6Li*(3+) catalysis
        {TRITON,  ALPHA,  NEUTRON, LI6_GS,  24.0, 2.0},         // t + 4He <-> n + 6Li
        {HELION,  ALPHA,  PROTON,  LI6_GS,  24.0, 2.0},         // 3He + 4He <-> p + 6Li
        {TRITON,  ALPHA,  PROTON,  HE6_GS,  18.0, 2.0},         // t + 4He <-> p + 6He
        {PROTON,  LI6_GS, PROTON,  LI6_219, 32.0, 1.5},         // p + 6Li <-> p + 6Li*(3+)
        {NEUTRON, LI6_GS, NEUTRON, LI6_219, 32.0, 1.5}          // n + 6Li <-> n + 6Li*(3+)
    };
    return table[idx];
}

// True if channel involves any A >= 2 nuclear cluster (false for pure N + Delta + pi channels)
HD_INLINE constexpr bool channel_involves_nuclear_cluster(int ch_idx) {
    return (ch_idx <= 45) || (ch_idx >= 52 && ch_idx <= 54) || (ch_idx >= 57);
}

// Evaluate nonsingular v_moller * sigma_{in -> out}(s) [fm^2] for a given channel & direction.
// Direction +1: (id1, id2) matches (a, b) -> outgoing (c, d)
// Direction -1: (id1, id2) matches (c, d) -> outgoing (a, b)
// Note: In all our channels, (a, b) are stable ground-state species, while d may be an
// unstable Breit-Wigner resonance! Thus:
//   Forward (+1): uses exact phase_space_spectral_integral over m_d if d is excited
//   Reverse (-1): uses on-shell p_cm(sqrt_s; m_a, m_b) directly (exact off-shell detailed balance!)
HD_INLINE double inelastic_2to2_vmoller_sigma_fm2(int ch_idx,
                                                  int id1, int id2,
                                                  const FourVector& p1,
                                                  const FourVector& p2,
                                                  double sqrt_s,
                                                  const SpectralNormTable& norms,
                                                  double scale_factor,
                                                  int& out_id3,
                                                  int& out_id4) {
    const BinaryInelasticChannel ch = get_inelastic_2to2_channel(ch_idx);

    bool forward = false;
    if ((id1 == ch.a && id2 == ch.b) || (id1 == ch.b && id2 == ch.a)) {
        forward = true;
    } else if (!((id1 == ch.c && id2 == ch.d) || (id1 == ch.d && id2 == ch.c))) {
        return 0.0;
    }

    const int out1 = forward ? ch.c : ch.a;
    const int out2 = forward ? ch.d : ch.b;
    out_id3 = out1;
    out_id4 = out2;

    const SpeciesInfo sp_out1 = get_species_info(out1);
    const SpeciesInfo sp_out2 = get_species_info(out2);

    // Symmetric threshold s_th = max((m_a + m_b)^2, (m_c_min + m_d_min)^2)
    const double m_ab = get_species_info(ch.a).pole_mass_gev + get_species_info(ch.b).pole_mass_gev;
    const double m_cd_min = resonance_min_mass(ch.c) + resonance_min_mass(ch.d);
    const double m_th_max = (m_ab > m_cd_min) ? m_ab : m_cd_min;
    const double s = sqrt_s * sqrt_s;
    const double s_th = m_th_max * m_th_max;
    if (s <= s_th) return 0.0;

    // Outgoing phase-space factor: p_cm_out (if both stable) or spectral integral I_R(s)
    double p_out_eff = 0.0;
    if (!sp_out1.is_excited && !sp_out2.is_excited) {
        p_out_eff = two_body_p_cm(sqrt_s, sp_out1.pole_mass_gev, sp_out2.pole_mass_gev);
    } else if (!sp_out1.is_excited && sp_out2.is_excited) {
        p_out_eff = phase_space_spectral_integral(out2, sqrt_s, sp_out1.pole_mass_gev, norms);
    } else if (sp_out1.is_excited && !sp_out2.is_excited) {
        p_out_eff = phase_space_spectral_integral(out1, sqrt_s, sp_out2.pole_mass_gev, norms);
    }
    if (p_out_eff <= 0.0) return 0.0;

    // Symmetric invariant coupling Sigma_0(s) [fm^2 * GeV^2]
    const double sigma_sym =
        scale_factor * ch.strength_fm2_gev2 / (1.0 + (s - s_th) / ch.form_scale_gev2);

    const double S_out = (out1 == out2) ? 2.0 : 1.0;
    const double g_out = static_cast<double>(sp_out1.degeneracy * sp_out2.degeneracy);

    // v_moller * sigma_{in -> out} = (g_out / S_out) * p_out_eff * Sigma_0(s) / (sqrt_s * E_1 * E_2)
    return (g_out / S_out) * p_out_eff * sigma_sym / (sqrt_s * p1.t * p2.t);
}

}  // namespace transport
