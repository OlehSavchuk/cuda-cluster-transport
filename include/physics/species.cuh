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

    NUM_SPECIES = 28
};

// Pure numeric POD (pointer-free so nvcc places it in GPU constant/register memory cleanly)
struct SpeciesInfo {
    int id;
    int baryon_num;         // A (baryon number)
    int charge;             // Z (electric charge)
    int two_J;              // 2 * spin J
    int parity;             // +1 or -1
    int two_I;              // 2 * total isospin I
    int degeneracy;         // g = 2*J + 1
    double pole_mass_gev;   // On-shell / Breit-Wigner pole mass [GeV]
    double excitation_mev;  // Excitation energy above ground state (or threshold) [MeV]
    double width_gev;       // Total decay width Gamma [GeV] (0 for stable states)
    double rms_radius_fm;   // Matter RMS radius [fm] used in Glauber/optical cross sections
    bool is_excited;        // True for excited / resonant states
};

HD_INLINE constexpr SpeciesInfo get_species_info(int id) {
    using namespace phys;
    constexpr SpeciesInfo table[NUM_SPECIES] = {
        // id, A, Z, 2J, P, 2I, g, mass [GeV], Ex [MeV], Gamma [GeV], R_rms [fm], excited
        {PROTON,    1, 1, 1, +1, 1, 2, MASS_PROTON,                          0.00,  0.0,       0.84, false},
        {NEUTRON,   1, 0, 1, +1, 1, 2, MASS_NEUTRON,                         0.00,  0.0,       0.84, false},

        // A = 2: deuteron + singlet d* isospin triplet (modeled ~1 MeV above NN threshold)
        {DEUTERON,  2, 1, 2, +1, 0, 3, MASS_DEUTERON,                        0.00,  0.0,       2.14, false},
        {D_STAR_NN, 2, 0, 0, +1, 2, 1, 2.0 * MASS_NEUTRON + 0.0010,          3.22,  0.0100,    3.80, true},
        {D_STAR_PN, 2, 1, 0, +1, 2, 1, MASS_PROTON + MASS_NEUTRON + 0.0010,  3.22,  0.0100,    3.80, true},
        {D_STAR_PP, 2, 2, 0, +1, 2, 1, 2.0 * MASS_PROTON + 0.0010,           3.22,  0.0100,    3.80, true},

        // A = 3: triton & helion
        {TRITON,    3, 1, 1, +1, 1, 2, MASS_TRITON,                          0.00,  0.0,       1.76, false},
        {HELION,    3, 2, 1, +1, 1, 2, MASS_HELION,                          0.00,  0.0,       1.97, false},

        // A = 4: alpha ground state
        {ALPHA,     4, 2, 0, +1, 0, 1, MASS_ALPHA,                           0.00,  0.0,       1.68, false},

        // A = 4: discrete 4He* excited states (R_rms expanded relative to g.s. 4He = 1.68 fm;
        // 20.21 MeV 0+ monopole state has documented diffuse breathing-mode radius ~2.5 fm)
        {HE4_2021,  4, 2, 0, +1, 0, 1, MASS_ALPHA + 0.02021,                20.21,  0.00050,   2.52, true},
        {HE4_2101,  4, 2, 0, -1, 0, 1, MASS_ALPHA + 0.02101,                21.01,  0.00084,   2.35, true},
        {HE4_2184,  4, 2, 4, -1, 0, 5, MASS_ALPHA + 0.02184,                21.84,  0.00201,   2.35, true},
        {HE4_2333,  4, 2, 4, -1, 2, 5, MASS_ALPHA + 0.02333,                23.33,  0.00501,   2.40, true},
        {HE4_2364,  4, 2, 2, -1, 2, 3, MASS_ALPHA + 0.02364,                23.64,  0.00620,   2.40, true},
        {HE4_2425,  4, 2, 2, -1, 0, 3, MASS_ALPHA + 0.02425,                24.25,  0.00610,   2.42, true},
        {HE4_2528,  4, 2, 0, -1, 0, 1, MASS_ALPHA + 0.02528,                25.28,  0.00797,   2.45, true},
        {HE4_2595,  4, 2, 2, -1, 0, 3, MASS_ALPHA + 0.02595,                25.95,  0.01266,   2.48, true},
        {HE4_2640,  4, 2, 4, -1, 2, 5, MASS_ALPHA + 0.02640,                26.40,  0.01070,   2.48, true},
        {HE4_2742,  4, 2, 4, +1, 0, 5, MASS_ALPHA + 0.02742,                27.42,  0.00869,   2.50, true},
        {HE4_2797,  4, 2, 2, -1, 2, 3, MASS_ALPHA + 0.02797,                27.97,  0.00900,   2.50, true},
        {HE4_2831,  4, 2, 2, +1, 0, 3, MASS_ALPHA + 0.02831,                28.31,  0.00989,   2.52, true},
        {HE4_2837,  4, 2, 2, -1, 0, 3, MASS_ALPHA + 0.02837,                28.37,  0.00392,   2.52, true},
        {HE4_2839,  4, 2, 4, -1, 0, 5, MASS_ALPHA + 0.02839,                28.39,  0.00875,   2.52, true},
        {HE4_2864,  4, 2, 0, -1, 0, 1, MASS_ALPHA + 0.02864,                28.64,  0.00489,   2.54, true},
        {HE4_2867,  4, 2, 4, +1, 0, 5, MASS_ALPHA + 0.02867,                28.67,  0.00378,   2.54, true},
        {HE4_2989,  4, 2, 4, +1, 0, 5, MASS_ALPHA + 0.02989,                29.89,  0.00972,   2.58, true},

        // A = 4, I = 1 resonant nuclei 4H and 4Li
        {H4_RES,    4, 1, 4, -1, 2, 5, MASS_TRITON + MASS_NEUTRON + 0.00319,  3.19,  0.00542,   2.42, true},
        {LI4_RES,   4, 3, 4, -1, 2, 5, MASS_HELION + MASS_PROTON + 0.00407,   4.07,  0.00603,   2.45, true}
    };
    return table[id];
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
        "4H*(2-)", "4Li*(2-)"
    };
    return (id >= 0 && id < NUM_SPECIES) ? names[id] : "unknown";
}

}  // namespace transport
