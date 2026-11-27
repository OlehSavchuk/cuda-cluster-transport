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
    Real strength_fm2_gev2;      // Symmetric coupling constant Sigma_scale [fm^2 * GeV^2]
    Real inv_form_scale_gev2;    // Reciprocal form-factor scale 1 / Lambda_s^2 [GeV^-2]
};

constexpr int NUM_INELASTIC_2TO2_CHANNELS = 62;

#define TR_CH_ROW(A, B, C, D, S, F) \
    {A, B, C, D, static_cast<Real>(S), static_cast<Real>(1.0 / (F))}

#define TR_INEL_TABLE_INIT { \
    TR_CH_ROW(PROTON,   TRITON,   NEUTRON,  HELION,   18.0, 1.5), \
    TR_CH_ROW(DEUTERON, DEUTERON, PROTON,   TRITON,   22.0, 1.8), \
    TR_CH_ROW(DEUTERON, DEUTERON, NEUTRON,  HELION,   22.0, 1.8), \
    TR_CH_ROW(DEUTERON, TRITON,   NEUTRON,  ALPHA,    36.0, 2.0), \
    TR_CH_ROW(DEUTERON, HELION,   PROTON,   ALPHA,    36.0, 2.0), \
    TR_CH_ROW(TRITON,   HELION,   DEUTERON, ALPHA,    24.0, 2.2), \
    TR_CH_ROW(PROTON,   DEUTERON, PROTON,   D_STAR_PN, 48.0, 1.2), \
    TR_CH_ROW(PROTON,   DEUTERON, NEUTRON,  D_STAR_PP, 48.0, 1.2), \
    TR_CH_ROW(NEUTRON,  DEUTERON, NEUTRON,  D_STAR_PN, 48.0, 1.2), \
    TR_CH_ROW(NEUTRON,  DEUTERON, PROTON,   D_STAR_NN, 48.0, 1.2), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2021, 16.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2101, 14.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2184, 12.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2333, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2364, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2425, 11.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2528, 11.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2595, 11.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2640, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2742, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2797, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2831, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2837, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2839, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2864, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2867, 10.0, 2.0), \
    TR_CH_ROW(PROTON, ALPHA, PROTON, HE4_2989, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2021, 16.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2101, 14.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2184, 12.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2333, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2364, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2425, 11.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2528, 11.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2595, 11.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2640, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2742, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2797, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2831, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2837, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2839, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2864, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2867, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, NEUTRON, HE4_2989, 10.0, 2.0), \
    TR_CH_ROW(NEUTRON, ALPHA, PROTON,  H4_RES,   12.0, 2.0), \
    TR_CH_ROW(PROTON,  ALPHA, NEUTRON, LI4_RES,  12.0, 2.0), \
    TR_CH_ROW(PROTON,  PROTON,  NEUTRON, DELTA_PLUSPLUS, 6.3, 1.4), \
    TR_CH_ROW(PROTON,  PROTON,  PROTON,  DELTA_PLUS,     2.1, 1.4), \
    TR_CH_ROW(PROTON,  NEUTRON, PROTON,  DELTA_ZERO,     2.1, 1.4), \
    TR_CH_ROW(PROTON,  NEUTRON, NEUTRON, DELTA_PLUS,     2.1, 1.4), \
    TR_CH_ROW(NEUTRON, NEUTRON, PROTON,  DELTA_MINUS,    6.3, 1.4), \
    TR_CH_ROW(NEUTRON, NEUTRON, NEUTRON, DELTA_ZERO,     2.1, 1.4), \
    TR_CH_ROW(PROTON,  PROTON,  DEUTERON, PION_PLUS,     1.8, 0.6), \
    TR_CH_ROW(PROTON,  NEUTRON, DEUTERON, PION_ZERO,     0.9, 0.6), \
    TR_CH_ROW(NEUTRON, NEUTRON, DEUTERON, PION_MINUS,    1.8, 0.6), \
    TR_CH_ROW(PION_MINUS, PROTON,  PION_ZERO, NEUTRON,   1.2, 0.8), \
    TR_CH_ROW(PION_PLUS,  NEUTRON, PION_ZERO, PROTON,    1.2, 0.8), \
    TR_CH_ROW(TRITON,  ALPHA,  NEUTRON, LI6_GS,  24.0, 2.0), \
    TR_CH_ROW(HELION,  ALPHA,  PROTON,  LI6_GS,  24.0, 2.0), \
    TR_CH_ROW(TRITON,  ALPHA,  PROTON,  HE6_GS,  18.0, 2.0), \
    TR_CH_ROW(PROTON,  LI6_GS, PROTON,  LI6_219, 32.0, 1.5), \
    TR_CH_ROW(NEUTRON, LI6_GS, NEUTRON, LI6_219, 32.0, 1.5)  \
}

inline constexpr BinaryInelasticChannel kHostInelasticTable[NUM_INELASTIC_2TO2_CHANNELS] = TR_INEL_TABLE_INIT;
#if TRANSPORT_HAS_CUDA
static __device__ __constant__ const BinaryInelasticChannel kDevInelasticTable[NUM_INELASTIC_2TO2_CHANNELS] = TR_INEL_TABLE_INIT;
#endif
#undef TR_CH_ROW
#undef TR_INEL_TABLE_INIT

HD_INLINE constexpr BinaryInelasticChannel get_inelastic_2to2_channel(int idx) {
#if defined(__CUDA_ARCH__)
    return kDevInelasticTable[idx];
#else
    return kHostInelasticTable[idx];
#endif
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
HD_FUNC Real inelastic_2to2_vmoller_sigma_fm2(int ch_idx,
                                                int id1, int id2,
                                                const FourVector& p1,
                                                const FourVector& p2,
                                                Real sqrt_s,
                                                const SpectralNormTable& norms,
                                                Real scale_factor,
                                                int& out_id3,
                                                int& out_id4) {
    const BinaryInelasticChannel ch = get_inelastic_2to2_channel(ch_idx);

    bool forward = false;
    if ((id1 == ch.a && id2 == ch.b) || (id1 == ch.b && id2 == ch.a)) {
        forward = true;
    } else if (!((id1 == ch.c && id2 == ch.d) || (id1 == ch.d && id2 == ch.c))) {
        return Real(0);
    }

    const int out1 = forward ? ch.c : ch.a;
    const int out2 = forward ? ch.d : ch.b;
    out_id3 = out1;
    out_id4 = out2;

    const SpeciesInfo sp_out1 = get_species_info(out1);
    const SpeciesInfo sp_out2 = get_species_info(out2);

    const Real m_ab = get_species_info(ch.a).pole_mass_gev + get_species_info(ch.b).pole_mass_gev;
    const Real m_cd_min = resonance_min_mass(ch.c) + resonance_min_mass(ch.d);
    const Real m_th_max = (m_ab > m_cd_min) ? m_ab : m_cd_min;
    const Real s = sqrt_s * sqrt_s;
    const Real s_th = m_th_max * m_th_max;
    if (s <= s_th) return Real(0);

    Real p_out_eff = Real(0);
    if (!sp_out1.is_excited && !sp_out2.is_excited) {
        p_out_eff = two_body_p_cm(sqrt_s, sp_out1.pole_mass_gev, sp_out2.pole_mass_gev);
    } else if (!sp_out1.is_excited && sp_out2.is_excited) {
        p_out_eff = phase_space_spectral_integral(out2, sqrt_s, sp_out1.pole_mass_gev, norms);
    } else if (sp_out1.is_excited && !sp_out2.is_excited) {
        p_out_eff = phase_space_spectral_integral(out1, sqrt_s, sp_out2.pole_mass_gev, norms);
    }
    if (p_out_eff <= Real(0)) return Real(0);

    const Real sigma_sym =
        scale_factor * ch.strength_fm2_gev2 /
        (Real(1) + (s - s_th) * ch.inv_form_scale_gev2);

    const Real S_out = (out1 == out2) ? Real(2) : Real(1);
    const Real g_out = static_cast<Real>(sp_out1.degeneracy * sp_out2.degeneracy);

    return (g_out / S_out) * p_out_eff * sigma_sym / (sqrt_s * p1.t * p2.t);
}

}  // namespace transport
