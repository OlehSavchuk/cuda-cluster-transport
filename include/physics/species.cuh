#pragma once

#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"

namespace transport {

// Complete nucleon + light-cluster (A <= 4) species catalog, including
// spin-singlet dinucleon virtual/resonant states (d*) and the TUNL discrete
// excited spectrum of 4He* (Tilley et al., Nucl. Phys. A541, 1 (1992)) plus
// isobaric A = 4 resonant partners 4H and 4Li.
enum SpeciesId : int {
    // A = 1 nucleons
    PROTON = 0,
    NEUTRON = 1,

    // A = 2 ground state & spin-singlet (J^P = 0+, I = 1) excited/virtual triplet
    DEUTERON = 2,       // d (1+, I=0)
    D_STAR_NN = 3,      // nn* (0+, I=1, Iz=-1)
    D_STAR_PN = 4,      // pn* / d* (0+, I=1, Iz=0)
    D_STAR_PP = 5,      // pp* (0+, I=1, Iz=+1)

    // A = 3 ground states
    TRITON = 6,         // t / 3H (1/2+, I=1/2)
    HELION = 7,         // 3He (1/2+, I=1/2)

    // A = 4 ground state
    ALPHA = 8,          // 4He g.s. (0+, I=0)

    // A = 4 excited 4He* states (TUNL evaluation up to Ex < 30 MeV)
    HE4_2021 = 9,       // Ex = 20.21 MeV, J^P = 0+, I = 0, Gamma = 0.50 MeV (breathing mode)
    HE4_2101 = 10,      // Ex = 21.01 MeV, J^P = 0-, I = 0, Gamma = 0.84 MeV
    HE4_2184 = 11,      // Ex = 21.84 MeV, J^P = 2-, I = 0, Gamma = 2.01 MeV
    HE4_2333 = 12,      // Ex = 23.33 MeV, J^P = 2-, I = 1, Gamma = 5.01 MeV
    HE4_2364 = 13,      // Ex = 23.64 MeV, J^P = 1-, I = 1, Gamma = 6.20 MeV
    HE4_2425 = 14,      // Ex = 24.25 MeV, J^P = 1-, I = 0, Gamma = 6.10 MeV
    HE4_2528 = 15,      // Ex = 25.28 MeV, J^P = 0-, I = 0, Gamma = 7.97 MeV
    HE4_2595 = 16,      // Ex = 25.95 MeV, J^P = 1-, I = 0, Gamma = 12.66 MeV
    HE4_2640 = 17,      // Ex = 26.40 MeV, J^P = 2-, I = 1, Gamma = 10.70 MeV
    HE4_2742 = 18,      // Ex = 27.42 MeV, J^P = 2+, I = 0, Gamma = 8.69 MeV
    HE4_2797 = 19,      // Ex = 27.97 MeV, J^P = 1-, I = 1, Gamma = 9.00 MeV
    HE4_2831 = 20,      // Ex = 28.31 MeV, J^P = 1+, I = 0, Gamma = 9.89 MeV
    HE4_2837 = 21,      // Ex = 28.37 MeV, J^P = 1-, I = 0, Gamma = 3.92 MeV
    HE4_2839 = 22,      // Ex = 28.39 MeV, J^P = 2-, I = 0, Gamma = 8.75 MeV
    HE4_2864 = 23,      // Ex = 28.64 MeV, J^P = 0-, I = 0, Gamma = 4.89 MeV
    HE4_2867 = 24,      // Ex = 28.67 MeV, J^P = 2+, I = 0, Gamma = 3.78 MeV
    HE4_2989 = 25,      // Ex = 29.89 MeV, J^P = 2+, I = 0, Gamma = 9.72 MeV

    // A = 4, I = 1 ground resonances (isobaric partners)
    H4_RES = 26,        // 4H (2-, I=1, Iz=-1), ~3.19 MeV above t + n
    LI4_RES = 27,       // 4Li (2-, I=1, Iz=+1), ~4.07 MeV above 3He + p

    // Mesons: Pion isospin triplet (B = 0, J^P = 0-, I = 1)
    PION_MINUS = 28,    // pi- (0-, I=1, Iz=-1, Z=-1)
    PION_ZERO = 29,     // pi0 (0-, I=1, Iz=0,  Z=0)
    PION_PLUS = 30,     // pi+ (0-, I=1, Iz=+1, Z=+1)

    // Baryon resonances: Delta(1232) P33 isospin quartet (B = 1, J^P = 3/2+, I = 3/2)
    DELTA_MINUS = 31,    // Delta-  (3/2+, I=3/2, Iz=-3/2, Z=-1)
    DELTA_ZERO = 32,     // Delta0  (3/2+, I=3/2, Iz=-1/2, Z=0)
    DELTA_PLUS = 33,     // Delta+  (3/2+, I=3/2, Iz=+1/2, Z=+1)
    DELTA_PLUSPLUS = 34, // Delta++ (3/2+, I=3/2, Iz=+3/2, Z=+2)

    // Remaining A = 4, I = 1 TUNL excited multiplets of 4H and 4Li (Tilley et al., NPA 541, 1992)
    H4_1M_350 = 35,      // 4H*(1-_1), Ex = 0.31 MeV (3.50 MeV above t+n), Gamma = 6.73 MeV
    H4_0M_527 = 36,      // 4H*(0-),   Ex = 2.08 MeV (5.27 MeV above t+n), Gamma = 8.92 MeV
    H4_1M_602 = 37,      // 4H*(1-_2), Ex = 2.83 MeV (6.02 MeV above t+n), Gamma = 12.99 MeV
    LI4_1M_439 = 38,     // 4Li*(1-_1), Ex = 0.32 MeV (4.39 MeV above 3He+p), Gamma = 7.35 MeV
    LI4_0M_615 = 39,     // 4Li*(0-),   Ex = 2.08 MeV (6.15 MeV above 3He+p), Gamma = 9.35 MeV
    LI4_1M_692 = 40,     // 4Li*(1-_2), Ex = 2.85 MeV (6.92 MeV above 3He+p), Gamma = 13.51 MeV

    // A = 5 isotopes: 5H, 5He, 5Li (TUNL, Tilley et al., NPA 708, 2002; all particle-unstable!)
    H5_GS = 41,          // 5H g.s. (1/2+, I=3/2), 1.80 MeV above t+2n -> t + nn*
    HE5_GS = 42,         // 5He g.s. (3/2-, I=1/2), 0.798 MeV above 4He+n, Gamma = 0.648 MeV
    HE5_127 = 43,        // 5He*(1.27, 1/2-, I=1/2), 2.068 MeV above 4He+n, Gamma = 5.57 MeV
    HE5_1684 = 44,       // 5He*(16.84, 3/2+, I=1/2), d+t fusion resonance, Gamma = 0.075 MeV
    LI5_GS = 45,         // 5Li g.s. (3/2-, I=1/2), 1.965 MeV above 4He+p, Gamma = 1.23 MeV
    LI5_149 = 46,        // 5Li*(1.49, 1/2-, I=1/2), 3.455 MeV above 4He+p, Gamma = 6.60 MeV
    LI5_1687 = 47,       // 5Li*(16.87, 3/2+, I=1/2), d+3He mirror resonance, Gamma = 0.267 MeV

    // A = 6 isotopes: 6H, 6He (bound g.s. + 2+), 6Li (bound g.s. + 4 resonances), 6Be (g.s. + 2+)
    H6_GS = 48,          // 6H g.s. (2-, I=2), 2.70 MeV above t+3n -> 5H + n
    HE6_GS = 49,         // 6He g.s. (0+, I=1), BOUND (B = 29.271 MeV)
    HE6_180 = 50,        // 6He*(1.80, 2+, I=1), Gamma = 0.113 MeV -> 5He + n
    LI6_GS = 51,         // 6Li g.s. (1+, I=0), BOUND (B = 31.994 MeV)
    LI6_219 = 52,        // 6Li*(2.186, 3+, I=0), Gamma = 0.024 MeV -> 4He + d
    LI6_431 = 53,        // 6Li*(4.312, 2+, I=0), Gamma = 1.30 MeV  -> 4He + d
    LI6_537 = 54,        // 6Li*(5.366, 2+, I=1), Gamma = 0.54 MeV  -> 5He + p
    LI6_565 = 55,        // 6Li*(5.650, 1+, I=0), Gamma = 1.50 MeV  -> 4He + d
    BE6_GS = 56,         // 6Be g.s. (0+, I=1), 1.371 MeV above 4He+2p -> 4He + pp*
    BE6_167 = 57,        // 6Be*(1.67, 2+, I=1), Gamma = 1.16 MeV -> 5Li + p / 4He + pp*

    NUM_SPECIES = 58
};

HD_INLINE constexpr bool is_pion(int id) {
    return id >= PION_MINUS && id <= PION_PLUS;
}

HD_INLINE constexpr bool is_delta(int id) {
    return id >= DELTA_MINUS && id <= DELTA_PLUSPLUS;
}

HD_INLINE constexpr bool is_a4_excited(int id) {
    return (id >= HE4_2021 && id <= LI4_RES) || (id >= H4_1M_350 && id <= LI4_1M_692);
}

HD_INLINE constexpr bool is_a56_excited(int id) {
    return (id >= H5_GS && id <= H6_GS) || id == HE6_180 || (id >= LI6_219 && id <= BE6_167);
}

// Pure numeric POD (pointer-free so nvcc places it in GPU constant/register memory cleanly)
struct SpeciesInfo {
    int id;
    int baryon_num;         // A (baryon number)
    int charge;             // Z (electric charge)
    int two_J;              // 2 * spin J
    int parity;             // +1 or -1
    int two_I;              // 2 * total isospin I
    int degeneracy;         // g = 2*J + 1
    Real pole_mass_gev;     // On-shell / Breit-Wigner pole mass [GeV]
    Real excitation_mev;    // Excitation energy above ground state (or threshold) [MeV]
    Real width_gev;         // Total decay width Gamma [GeV] (0 for stable states)
    Real rms_radius_fm;     // Matter RMS radius [fm] used in Glauber/optical cross sections
    bool is_excited;        // True for excited / resonant states
};

#define TR_SP_ROW(ID, A, Z, J2, P, I2, G, M, EX, W, R, EXC) \
    {ID, A, Z, J2, P, I2, G, static_cast<Real>(M), static_cast<Real>(EX), static_cast<Real>(W), static_cast<Real>(R), EXC}

#define TR_SPECIES_TABLE_INIT { \
    TR_SP_ROW(PROTON,    1, 1, 1, +1, 1, 2, phys::MASS_PROTON,                          0.00,  0.0,      0.84, false), \
    TR_SP_ROW(NEUTRON,   1, 0, 1, +1, 1, 2, phys::MASS_NEUTRON,                         0.00,  0.0,      0.84, false), \
    TR_SP_ROW(DEUTERON,  2, 1, 2, +1, 0, 3, phys::MASS_DEUTERON,                        0.00,  0.0,      2.14, false), \
    TR_SP_ROW(D_STAR_NN, 2, 0, 0, +1, 2, 1, 2.0 * phys::MASS_NEUTRON + 0.0010,          3.22,  0.0100,   3.80, true),  \
    TR_SP_ROW(D_STAR_PN, 2, 1, 0, +1, 2, 1, phys::MASS_PROTON + phys::MASS_NEUTRON + 0.0010, 3.22, 0.0100, 3.80, true),\
    TR_SP_ROW(D_STAR_PP, 2, 2, 0, +1, 2, 1, 2.0 * phys::MASS_PROTON + 0.0010,           3.22,  0.0100,   3.80, true),  \
    TR_SP_ROW(TRITON,    3, 1, 1, +1, 1, 2, phys::MASS_TRITON,                          0.00,  0.0,      1.76, false), \
    TR_SP_ROW(HELION,    3, 2, 1, +1, 1, 2, phys::MASS_HELION,                          0.00,  0.0,      1.97, false), \
    TR_SP_ROW(ALPHA,     4, 2, 0, +1, 0, 1, phys::MASS_ALPHA,                           0.00,  0.0,      1.68, false), \
    TR_SP_ROW(HE4_2021,  4, 2, 0, +1, 0, 1, phys::MASS_ALPHA + 0.02021,                20.21,  0.00050,  2.52, true),  \
    TR_SP_ROW(HE4_2101,  4, 2, 0, -1, 0, 1, phys::MASS_ALPHA + 0.02101,                21.01,  0.00084,  2.35, true),  \
    TR_SP_ROW(HE4_2184,  4, 2, 4, -1, 0, 5, phys::MASS_ALPHA + 0.02184,                21.84,  0.00201,  2.35, true),  \
    TR_SP_ROW(HE4_2333,  4, 2, 4, -1, 2, 5, phys::MASS_ALPHA + 0.02333,                23.33,  0.00501,  2.40, true),  \
    TR_SP_ROW(HE4_2364,  4, 2, 2, -1, 2, 3, phys::MASS_ALPHA + 0.02364,                23.64,  0.00620,  2.40, true),  \
    TR_SP_ROW(HE4_2425,  4, 2, 2, -1, 0, 3, phys::MASS_ALPHA + 0.02425,                24.25,  0.00610,  2.42, true),  \
    TR_SP_ROW(HE4_2528,  4, 2, 0, -1, 0, 1, phys::MASS_ALPHA + 0.02528,                25.28,  0.00797,  2.45, true),  \
    TR_SP_ROW(HE4_2595,  4, 2, 2, -1, 0, 3, phys::MASS_ALPHA + 0.02595,                25.95,  0.01266,  2.48, true),  \
    TR_SP_ROW(HE4_2640,  4, 2, 4, -1, 2, 5, phys::MASS_ALPHA + 0.02640,                26.40,  0.01070,  2.48, true),  \
    TR_SP_ROW(HE4_2742,  4, 2, 4, +1, 0, 5, phys::MASS_ALPHA + 0.02742,                27.42,  0.00869,  2.50, true),  \
    TR_SP_ROW(HE4_2797,  4, 2, 2, -1, 2, 3, phys::MASS_ALPHA + 0.02797,                27.97,  0.00900,  2.50, true),  \
    TR_SP_ROW(HE4_2831,  4, 2, 2, +1, 0, 3, phys::MASS_ALPHA + 0.02831,                28.31,  0.00989,  2.52, true),  \
    TR_SP_ROW(HE4_2837,  4, 2, 2, -1, 0, 3, phys::MASS_ALPHA + 0.02837,                28.37,  0.00392,  2.52, true),  \
    TR_SP_ROW(HE4_2839,  4, 2, 4, -1, 0, 5, phys::MASS_ALPHA + 0.02839,                28.39,  0.00875,  2.52, true),  \
    TR_SP_ROW(HE4_2864,  4, 2, 0, -1, 0, 1, phys::MASS_ALPHA + 0.02864,                28.64,  0.00489,  2.54, true),  \
    TR_SP_ROW(HE4_2867,  4, 2, 4, +1, 0, 5, phys::MASS_ALPHA + 0.02867,                28.67,  0.00378,  2.54, true),  \
    TR_SP_ROW(HE4_2989,  4, 2, 4, +1, 0, 5, phys::MASS_ALPHA + 0.02989,                29.89,  0.00972,  2.58, true),  \
    TR_SP_ROW(H4_RES,    4, 1, 4, -1, 2, 5, phys::MASS_TRITON + phys::MASS_NEUTRON + 0.00319,  3.19, 0.00542, 2.42, true), \
    TR_SP_ROW(LI4_RES,   4, 3, 4, -1, 2, 5, phys::MASS_HELION + phys::MASS_PROTON  + 0.00407,  4.07, 0.00603, 2.45, true), \
    TR_SP_ROW(PION_MINUS,     0, -1, 0, -1, 2, 1, phys::MASS_PION_CHARGED,  0.00,   0.0,     0.66, false), \
    TR_SP_ROW(PION_ZERO,      0,  0, 0, -1, 2, 1, phys::MASS_PION_NEUTRAL,  0.00,   0.0,     0.66, false), \
    TR_SP_ROW(PION_PLUS,      0, +1, 0, -1, 2, 1, phys::MASS_PION_CHARGED,  0.00,   0.0,     0.66, false), \
    TR_SP_ROW(DELTA_MINUS,    1, -1, 3, +1, 3, 4, phys::MASS_DELTA_1232,  293.00,   0.11800, 0.95, true),  \
    TR_SP_ROW(DELTA_ZERO,     1,  0, 3, +1, 3, 4, phys::MASS_DELTA_1232,  293.00,   0.11800, 0.95, true),  \
    TR_SP_ROW(DELTA_PLUS,     1, +1, 3, +1, 3, 4, phys::MASS_DELTA_1232,  293.00,   0.11800, 0.95, true),  \
    TR_SP_ROW(DELTA_PLUSPLUS, 1, +2, 3, +1, 3, 4, phys::MASS_DELTA_1232,  293.00,   0.11800, 0.95, true),  \
    TR_SP_ROW(H4_1M_350,      4, 1, 2, -1, 2, 3, phys::MASS_TRITON + phys::MASS_NEUTRON + 0.00350,   3.50, 0.00673, 2.44, true), \
    TR_SP_ROW(H4_0M_527,      4, 1, 0, -1, 2, 1, phys::MASS_TRITON + phys::MASS_NEUTRON + 0.00527,   5.27, 0.00892, 2.46, true), \
    TR_SP_ROW(H4_1M_602,      4, 1, 2, -1, 2, 3, phys::MASS_TRITON + phys::MASS_NEUTRON + 0.00602,   6.02, 0.01299, 2.48, true), \
    TR_SP_ROW(LI4_1M_439,     4, 3, 2, -1, 2, 3, phys::MASS_HELION + phys::MASS_PROTON  + 0.00439,   4.39, 0.00735, 2.46, true), \
    TR_SP_ROW(LI4_0M_615,     4, 3, 0, -1, 2, 1, phys::MASS_HELION + phys::MASS_PROTON  + 0.00615,   6.15, 0.00935, 2.48, true), \
    TR_SP_ROW(LI4_1M_692,     4, 3, 2, -1, 2, 3, phys::MASS_HELION + phys::MASS_PROTON  + 0.00692,   6.92, 0.01351, 2.50, true), \
    TR_SP_ROW(H5_GS,          5, 1, 1, +1, 3, 2, phys::MASS_TRITON + 2.0 * phys::MASS_NEUTRON + 0.00180, 1.80, 0.00190,  2.60, true), \
    TR_SP_ROW(HE5_GS,         5, 2, 3, -1, 1, 4, phys::MASS_ALPHA  + phys::MASS_NEUTRON + 0.000798,      0.80, 0.000648, 2.45, true), \
    TR_SP_ROW(HE5_127,        5, 2, 1, -1, 1, 2, phys::MASS_ALPHA  + phys::MASS_NEUTRON + 0.002068,      2.07, 0.00557,  2.52, true), \
    TR_SP_ROW(HE5_1684,       5, 2, 3, +1, 1, 4, phys::MASS_DEUTERON + phys::MASS_TRITON + 0.000060,    17.65, 0.000075, 2.65, true), \
    TR_SP_ROW(LI5_GS,         5, 3, 3, -1, 1, 4, phys::MASS_ALPHA  + phys::MASS_PROTON  + 0.001965,      1.97, 0.00123,  2.48, true), \
    TR_SP_ROW(LI5_149,        5, 3, 1, -1, 1, 2, phys::MASS_ALPHA  + phys::MASS_PROTON  + 0.003455,      3.46, 0.00660,  2.55, true), \
    TR_SP_ROW(LI5_1687,       5, 3, 3, +1, 1, 4, phys::MASS_DEUTERON + phys::MASS_HELION + 0.000480,    18.83, 0.000267, 2.68, true), \
    TR_SP_ROW(H6_GS,          6, 1, 4, -1, 4, 5, phys::MASS_TRITON + 3.0 * phys::MASS_NEUTRON + 0.00270, 2.70, 0.00160,  2.75, true), \
    TR_SP_ROW(HE6_GS,         6, 2, 0, +1, 2, 1, phys::MASS_HE6,                                   0.00, 0.0,      2.54, false),\
    TR_SP_ROW(HE6_180,        6, 2, 4, +1, 2, 5, phys::MASS_HE6 + 0.001850,                        1.80, 0.000113, 2.65, true), \
    TR_SP_ROW(LI6_GS,         6, 3, 2, +1, 0, 3, phys::MASS_LI6,                                   0.00, 0.0,      2.55, false),\
    TR_SP_ROW(LI6_219,        6, 3, 6, +1, 0, 7, phys::MASS_LI6 + 0.002186,                        2.19, 0.000024, 2.62, true), \
    TR_SP_ROW(LI6_431,        6, 3, 4, +1, 0, 5, phys::MASS_LI6 + 0.004312,                        4.31, 0.00130,  2.66, true), \
    TR_SP_ROW(LI6_537,        6, 3, 4, +1, 2, 5, phys::MASS_LI6 + 0.005366,                        5.37, 0.00054,  2.68, true), \
    TR_SP_ROW(LI6_565,        6, 3, 2, +1, 0, 3, phys::MASS_LI6 + 0.005650,                        5.65, 0.00150,  2.70, true), \
    TR_SP_ROW(BE6_GS,         6, 4, 0, +1, 2, 1, phys::MASS_ALPHA + 2.0 * phys::MASS_PROTON + 0.001371,  1.37, 0.000092, 2.65, true), \
    TR_SP_ROW(BE6_167,        6, 4, 4, +1, 2, 5, phys::MASS_ALPHA + 2.0 * phys::MASS_PROTON + 0.003041,  3.04, 0.00116,  2.72, true)  \
}

inline constexpr SpeciesInfo kHostSpeciesTable[NUM_SPECIES] = TR_SPECIES_TABLE_INIT;
#if TRANSPORT_HAS_CUDA
static __device__ __constant__ const SpeciesInfo kDevSpeciesTable[NUM_SPECIES] = TR_SPECIES_TABLE_INIT;
#endif
#undef TR_SP_ROW
#undef TR_SPECIES_TABLE_INIT

HD_INLINE constexpr SpeciesInfo get_species_info(int id) {
#if defined(__CUDA_ARCH__)
    return kDevSpeciesTable[id];
#else
    return kHostSpeciesTable[id];
#endif
}

// Effective non-negative proton/neutron constituent weights (wp >= 0, wn >= 0, wp + wn == A)
// for Skyrme symmetry potentials and Glauber isospin-weighted cross sections.
// Uses Clebsch-Gordan isospin projection for the I = 3/2 Delta(1232) quartet.
template <typename T>
HD_INLINE constexpr void get_effective_pn_weights(int id, T& wp, T& wn) {
    if (is_pion(id)) {
        wp = T(0); wn = T(0);
    } else if (id == DELTA_PLUSPLUS) {
        wp = T(1); wn = T(0);
    } else if (id == DELTA_PLUS) {
        wp = T(2) / T(3); wn = T(1) / T(3);
    } else if (id == DELTA_ZERO) {
        wp = T(1) / T(3); wn = T(2) / T(3);
    } else if (id == DELTA_MINUS) {
        wp = T(0); wn = T(1);
    } else {
        const SpeciesInfo sp = get_species_info(id);
        wp = static_cast<T>(sp.charge);
        wn = static_cast<T>(sp.baryon_num - sp.charge);
    }
}

inline const char* get_species_name(int id) {
    static const char* const names[NUM_SPECIES] = {
        "p", "n",
        "d", "nn*(0+)", "d*(0+)", "pp*(0+)",
        "t", "3He",
        "4He",
        "4He*(20.21)", "4He*(21.01)", "4He*(21.84)", "4He*(23.33)", "4He*(23.64)",
        "4He*(24.25)", "4He*(25.28)", "4He*(25.95)", "4He*(26.40)", "4He*(27.42)",
        "4He*(27.97)", "4He*(28.31)", "4He*(28.37)", "4He*(28.39)", "4He*(28.64)",
        "4He*(28.67)", "4He*(29.89)",
        "4H*(2-)", "4Li*(2-)",
        "pi-", "pi0", "pi+",
        "Delta-", "Delta0", "Delta+", "Delta++",
        "4H*(1-_1)", "4H*(0-)", "4H*(1-_2)",
        "4Li*(1-_1)", "4Li*(0-)", "4Li*(1-_2)",
        "5H(1/2+)", "5He(3/2-)", "5He*(1/2-)", "5He*(16.84)",
        "5Li(3/2-)", "5Li*(1/2-)", "5Li*(16.87)",
        "6H(2-)", "6He", "6He*(2+)",
        "6Li", "6Li*(3+)", "6Li*(2+_0)", "6Li*(2+_1)", "6Li*(1+)",
        "6Be(0+)", "6Be*(2+)"
    };
    return (id >= 0 && id < NUM_SPECIES) ? names[id] : "unknown";
}

}  // namespace transport
