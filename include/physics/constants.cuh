#pragma once

#include "backend/cuda_compat.cuh"

namespace transport {
namespace phys {

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 6.28318530717958647692;
constexpr double HBARC = 0.1973269804;          // GeV * fm
constexpr double HBARC_SQ = HBARC * HBARC;      // GeV^2 * fm^2
constexpr double ALPHA_EM = 1.0 / 137.035999084;// Fine-structure constant e^2 / (4 pi eps_0 hbar c)
constexpr double E2_COULOMB_GEV_FM = ALPHA_EM * HBARC;  // e^2 = alpha * hbar*c [GeV * fm] (~1.4400e-3)
constexpr double MB_TO_FM2 = 0.1;               // 1 mb = 0.1 fm^2
constexpr double FM2_TO_MB = 10.0;              // 1 fm^2 = 10 mb
constexpr double MEV_TO_GEV = 1.0e-3;
constexpr double GEV_TO_MEV = 1.0e3;

// Fundamental nucleon masses (PDG / CODATA, in GeV)
constexpr double MASS_PROTON = 0.93827208816;
constexpr double MASS_NEUTRON = 0.93956542052;
constexpr double MASS_NUCLEON_AVG = 0.5 * (MASS_PROTON + MASS_NEUTRON);

// Ground-state nuclear masses (AME2020 nuclear masses, electrons subtracted, in GeV)
constexpr double MASS_DEUTERON = 1.87561294257;
constexpr double MASS_TRITON = 2.80892113298;
constexpr double MASS_HELION = 2.80839160743;
constexpr double MASS_ALPHA = 3.7273794066;
constexpr double MASS_HE6 = 5.60553516;   // 6He g.s. (0+, B = 29.271 MeV, 2n halo)
constexpr double MASS_LI6 = 5.60151816;   // 6Li g.s. (1+, B = 31.994 MeV)

// Pion and Delta(1232) resonance pole masses (PDG, in GeV)
constexpr double MASS_PION_CHARGED = 0.13957039;
constexpr double MASS_PION_NEUTRAL = 0.13497680;
constexpr double MASS_DELTA_1232 = 1.23200000;

}  // namespace phys
}  // namespace transport
