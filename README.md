# CUDA Relativistic Nucleon & Light-Cluster Transport

[![Build & Physics Verification Suite](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen)](#verification-suite)
[![CUDA C++17](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?logo=nvidia&logoColor=white)](#build--quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A GPU-native relativistic Boltzmann transport code written in **CUDA C++17** for **protons, neutrons, and light nuclear clusters ($A \le 4$) including their discrete excited states** ($d^*$ spin-singlet isospin triplet and 19 excited $A=4$ states ${}^4\text{He}^*$, ${}^4\text{H}^*$, ${}^4\text{Li}^*$), featuring **off-shell relativistic Breit–Wigner spectral functions, $2 \leftrightarrow 1$ resonance formation/decay, 46 reversible $2 \leftrightarrow 2$ inelastic nuclear transfer/catalysis/excitation channels with strict detailed balance, and Glauber elastic scattering**.

When `nvcc` is present, all particle propagation, dynamic slot allocation/compaction, 3D spatial grid binning, multi-level prefix-sum cell sorting, counter-based Philox RNG, and stochastic reaction/decay kernels execute natively on the GPU. On machines without `nvcc` (such as macOS laptops or CI runners), CMake seamlessly compiles the exact same physics kernels via a zero-dependency C++17 host backend.

---

## Visual Demonstrations

### 1. Sub-Saturation Cluster Nucleation at $\rho_B = n_0 / 8$, $T_{\text{start}} = 20\;\text{MeV}$
Starting from **100% pure protons and neutrons** at nuclear sub-saturation density $\rho_B = n_0 / 8 = 0.02\;\text{fm}^{-3}$ ($L = 30\;\text{fm}, N_{\text{test}} = 60$, zero probability overflows) and $T_{\text{start}} = 20.0\;\text{MeV}$, nearly **45% of all baryons** assemble into bound and resonant clusters ($d, d^*, t, {}^3\text{He}, {}^4\text{He}, {}^4\text{He}^*$)—exhibiting a BBN-like transient deuteron overshoot before converging onto the dashed analytical Saha / SHM equilibrium targets while self-heating the isolated box via cluster formation:

<p align="center">
  <img src="assets/nucleation_n0_over_8_T20MeV.gif" alt="Cluster Nucleation at n0/8 and T=20 MeV" width="100%">
</p>

### 2. High-Density Nucleosynthesis & Off-Shell Breit–Wigner Spectrum (`--mode nucleosynthesis`)
Starting from **100% pure protons and neutrons ($N_{\text{clusters}}(0) = 0$)** in a dense periodic box ($T_0 = 62\;\text{MeV}$), the coupled $2 \leftrightarrow 1$ and $2 \leftrightarrow 2$ network synthesizes the $A \le 4$ hierarchy while populating the thermal-folded Breit–Wigner spectral distribution:

<p align="center">
  <img src="assets/cluster_nucleosynthesis.gif" alt="Dynamic Light-Cluster Nucleosynthesis" width="100%">
</p>

### 2. Relativistic Breit–Wigner Spectral Functions & Threshold Widths (`--dump-bw`)
Unit-normalized spectral functions $\mathcal{A}_R(m)$ and Blatt–Weisskopf momentum-dependent widths $\Gamma_{\text{tot}}(m) \propto q^{2L+1}$ across $s$-, $p$-, and $d$-wave resonances (`include/physics/resonances.cuh`):

<p align="center">
  <img src="assets/breit_wigner_spectra.png" alt="Relativistic Breit-Wigner Spectral Functions" width="100%">
</p>

### 3. Counter-Streaming Nuclear & Excited-Cluster Slabs (`--mode slabs`)
Two opposing beams ($\beta_z = \pm 0.48c$) composed of nucleons ($p, n$), ground-state clusters ($d, t, {}^3\text{He}, {}^4\text{He}$), singlet dinucleons ($d^*$), and excited ${}^4\text{He}^*$ states collide head-on inside a 3D box. Bright yellow-white flashes mark active $2 \to 2$ elastic collision vertices in each step as longitudinal beam momentum splashes into transverse $4\pi$ thermal motion:

<p align="center">
  <img src="assets/colliding_slabs.gif" alt="Counter-Streaming Nuclear and Cluster Slabs" width="100%">
</p>

### 2. 3D Expanding Heavy-Ion Hadronic & Cluster Fireball (`--mode fireball`)
A hot baryon-rich fireball ($R_{\text{rms}}(0) = 5\;\text{fm}$) initialized with radial Hubble flow and Maxwell–Jüttner local thermal motion expands into open vacuum, undergoing intense core elastic rescattering followed by kinetic freeze-out:

<p align="center">
  <img src="assets/fireball_expansion.gif" alt="3D Expanding Heavy-Ion Cluster Fireball" width="100%">
</p>

### 3. Relativistic Boltzmann Pressure Isotropization (`--mode aniso`)
In a 3D periodic torus $[0, L)^3$, an initially momentum-anisotropic multi-species plasma ($P_z / P_\perp > 3$) relaxes exponentially to the isotropic Maxwell–Jüttner equilibrium circle while conserving total $P^\mu = (E, \vec{P})$ to IEEE-754 double precision ($|\Delta E|/E \sim 10^{-15}$) and $\Delta B = \Delta Q = 0$:

<p align="center">
  <img src="assets/box_isotropization.gif" alt="3D Periodic Box Boltzmann Isotropization" width="100%">
</p>

---

## Physics Model

### Complete 28-Species Catalog ($A \le 4$, Ground + Excited)
Defined in [`include/physics/species.cuh`](include/physics/species.cuh):

| Family | Species | $A$ | $Z$ | $J^\pi$ | $I$ | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Nucleons** | $p,\; n$ | 1 | 1, 0 | $1/2^+$ | $1/2$ | AME2020 / CODATA masses |
| **Dinucleons** | $d$ | 2 | 1 | $1^+$ | $0$ | Ground-state deuteron ($R_{\text{rms}} = 2.14\;\text{fm}$) |
| **Excited $A=2$** | $nn^*(0^+),\; d^*(0^+),\; pp^*(0^+)$ | 2 | 0, 1, 2 | $0^+$ | $1$ | Spin-singlet virtual/resonant isospin triplet ($R_{\text{rms}} = 3.80\;\text{fm}$) |
| **Trinucleons** | $t,\; {}^3\text{He}$ | 3 | 1, 2 | $1/2^+$ | $1/2$ | Ground-state $A=3$ mirror doublet |
| **Alpha (g.s.)** | ${}^4\text{He}$ | 4 | 2 | $0^+$ | $0$ | Compact ground state ($R_{\text{rms}} = 1.68\;\text{fm}$) |
| **Excited ${}^4\text{He}^*$** | 17 discrete TUNL levels | 4 | 2 | $0^\pm, 1^\pm, 2^\pm$ | $0, 1$ | $E_x = 20.21\;\text{MeV}$ ($0^+$ breathing mode) to $29.89\;\text{MeV}$ ($2^+$) |
| **Isobaric $A=4^*$** | ${}^4\text{H}^*(2^-),\; {}^4\text{Li}^*(2^-)$ | 4 | 1, 3 | $2^-$ | $1$ | $I=1$ resonant partners above $t+n$ and ${}^3\text{He}+p$ |

### Cancellation-Free Relativistic Kinematics
In [`include/physics/kinematics.cuh`](include/physics/kinematics.cuh), invariant pair quantities are evaluated via the algebraic identity

$$Q^2 \equiv (p_1 \cdot p_2)^2 - m_1^2 m_2^2 = |E_2 \vec{p}_1 - E_1 \vec{p}_2|^2 - |\vec{p}_1 \times \vec{p}_2|^2$$

eliminating catastrophic floating-point cancellation near threshold $s \to (m_1 + m_2)^2$ even when $\sim 3.75\;\text{GeV}$ excited $A=4$ clusters collide at keV relative kinetic energies under large Lorentz boosts.

### Elastic Cross Sections & Excited-State Spatial Swelling
Implemented in [`include/physics/cross_sections.cuh`](include/physics/cross_sections.cuh):
- **Elementary $NN$ ($pp, nn, np$)**: Smoothly bridges low-energy $S$-wave Effective Range Expansion (ERE), nuclear-regime Kikuchi–Kawai / Metropolis parameterizations, and high-energy Cugnon / PDG elastic fits.
- **Nucleon–Cluster & Cluster–Cluster**: Evaluates isospin-weighted $\bar{\sigma}_{NN}(\gamma_{\text{rel}})$ at the Lorentz-invariant relative factor $\gamma_{\text{rel}} = (p_1 \cdot p_2)/(m_1 m_2)$ and combines matter radii via the analytic Gaussian Glauber / Karol optical eikonal overlap:

$$\sigma_{AB}^{\text{el}}(\gamma_{\text{rel}}) = \pi a_{AB}^2 \ln\!\left(1 + \frac{A B \, \bar{\sigma}_{NN}(\gamma_{\text{rel}})}{\pi a_{AB}^2}\right), \qquad a_{AB}^2 = \frac{2}{3}\left(R_A^2 + R_B^2\right)$$

Diffuse excited states such as $d^*(0^+)$ and the ${}^4\text{He}^*(20.21, 0^+)$ monopole breathing mode naturally exhibit enhanced elastic cross sections relative to their compact ground states:

<p align="center">
  <img src="assets/cross_sections_overview.png" alt="Elastic Cross Sections Overview" width="100%">
</p>

---

## GPU Architecture & Repository Layout

```text
├── CMakeLists.txt                  # Auto-detects CUDA nvcc (SM 70–90) or C++17 host fallback
├── include/
│   ├── backend/
│   │   ├── cuda_compat.cuh         # DeviceBuffer<T>, atomic wrappers, CUDA/CPU portability macros
│   │   └── philox_rng.cuh          # Stateless counter-based Philox4x32-10 PRNG (zero VRAM state)
│   ├── physics/
│   │   ├── constants.cuh           # Physical constants & AME2020 nuclear masses (GeV, fm)
│   │   ├── species.cuh             # 28-species nucleon + cluster + excited-state table
│   │   ├── kinematics.cuh          # Cancellation-free Moller flux, CM boosts, 2->2 & 1->2 decays
│   │   ├── cross_sections.cuh      # ERE+KK+Cugnon NN & Gaussian Glauber cluster cross sections
│   │   ├── resonances.cuh          # Relativistic Breit-Wigner spectral functions & BW samplers
│   │   └── inelastic_channels.cuh  # 46 reversible 2<->2 transfer, catalysis & excitation rules
│   └── sim/
│       ├── particles.cuh           # Dynamic SoA layout with GPU atomic appends & compaction
│       ├── grid.cuh                # GPU multi-level prefix sum + counting-sort 3D cell builder
│       ├── stepper.cuh             # Strang-split drift + decay + DisjointPairs/AllPairs kernels
│       ├── init_conditions.cuh     # Off-shell grand-canonical sampler, nucleosynthesis & fireball
│       └── observables.cuh         # Parallel GPU reductions for P^mu, B, Q, T^ij & mass moments
├── src/
│   ├── main.cu                     # CLI simulation driver, CSV logger & binary/BW exporters
│   └── tests.cu                    # Automated physics, detailed-balance & nucleosynthesis suite
├── scripts/
│   ├── generate_inelastic_animations.py
│   └── generate_animations.py      # Reproduces all GIF/MP4 animations and physics figures
└── assets/                         # Rendered animations and spectral/cross-section plots
```

### Stochastic Collision Kernel (`DisjointPairs` vs `AllPairs`)
Inside each spatial cell of volume $\Delta V$ containing $N_c$ test particles (oversampling factor $N_{\text{test}}$):
- **`DisjointPairs` (default, $O(N_p)$ lock-free GPU parallel)**: 1 CUDA thread block per cell performs an in-cache Fisher–Yates shuffle and evaluates all $M_c = \lfloor N_c / 2 \rfloor$ disjoint pairs $(2k, 2k+1)$ simultaneously across threads with exact odd/even pair-multiplicity weighting:
  $$P_{\text{pair}} = \frac{N_c(N_c - 1)/2}{\lfloor N_c / 2 \rfloor} \, \frac{v_{\text{Møller}} \, \sigma_{\text{el}}(\sqrt{s}) \, \Delta t}{N_{\text{test}} \, \Delta V}$$
- **`AllPairs`**: Traverses all $N_c(N_c-1)/2$ intra-cell pairs in randomized order with single-scatter-per-substep guards.

---

## Build & Quickstart

### 1. Build (CUDA GPU or CPU Host Fallback)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
On a CUDA workstation/cluster, CMake automatically invokes `nvcc` (pass `-DCMAKE_CUDA_ARCHITECTURES=native` to target your exact GPU).

### 2. Run Verification Suite (`ctest`)
```bash
ctest --test-dir build --output-on-failure -V
```

### 3. Run Simulations
```bash
# Print all 28 species (ground & excited), thresholds, spectral norms, and cross sections
./build/cluster_transport --print-species

# Dynamic nucleosynthesis of d, d*, t, 3He, 4He, and 4He* from pure p + n
./build/cluster_transport --mode nucleosynthesis --npart 30000 --ntest 15 --steps 80 --dt 0.15

# Run in pure elastic baseline mode (--reactions elastic) or full inelastic+BW mode (--reactions all)
./build/cluster_transport --mode aniso --reactions all --npart 100000 --ntest 25 --steps 100 --dt 0.20
./build/cluster_transport --mode fireball --reactions all --npart 100000 --ntest 25 --steps 80 --dt 0.25
```

### 4. Regenerate Animations & Figures
Requires `python3` (`numpy`, `matplotlib`) and `ffmpeg`:
```bash
python3 scripts/generate_inelastic_animations.py
python3 scripts/generate_animations.py
```

---

## Verification Suite

[`src/tests.cu`](src/tests.cu) validates all physics and numerical invariants on every build:
1. **`test_species_table`**: Quantum numbers ($B, Z, J^\pi, I, g=2J+1$), pole masses, widths, and radii across all 6 ground and 22 excited states.
2. **`test_relativistic_kinematics`**: Boosted sub-keV threshold elastic, exothermic $2 \to 2$ ($d+t \to n+{}^4\text{He}$), and $1 \to 2$ resonance decay kinematics conserving $P^\mu$ to $< 10^{-13}$.
3. **`test_cross_section_hierarchy`**: Verifies $\sigma_{pn} > \sigma_{pp}$, $\sigma_{p d^*} > \sigma_{p d}$, and $\sigma_{p\,{}^4\text{He}^*(20.21)} > \sigma_{p\,{}^4\text{He}}$.
4. **`test_breit_wigner_spectral_properties`**: Verifies unit spectral normalization $\int \mathcal{A}_R(m)\,dm = 1$, exact threshold vanishing $\Gamma(m \le m_{\text{th}}) = 0$, and $(B, Q)$ conservation across all 22 resonances and 46 inelastic $2 \leftrightarrow 2$ channels.
5. **`test_stochastic_boltzmann_rate`**: Benchmarks both `DisjointPairs` and `AllPairs` at odd cell occupancy ($N_c = 251$) against the exact analytical pair sum.
6. **`test_detailed_balance_2to1_resonances`**: Verifies microscopic $2 \leftrightarrow 1$ detailed balance ($p+n \leftrightarrow d^*$: formations match decays within $<1\%$) and stationary off-shell $\langle m_{d^*} \rangle$ in a grand-canonical box.
7. **`test_detailed_balance_full_chemical_network`**: Verifies chemical stationarity across all 28 species with all $2 \leftrightarrow 1$ and 46 $2 \leftrightarrow 2$ reactions simultaneously active.
8. **`test_dynamic_cluster_nucleosynthesis`**: Verifies spontaneous nucleation of $d, d^*, t, {}^3\text{He}, {}^4\text{He}$, and ${}^4\text{He}^*$ from pure $p+n$ with $|\Delta E|/E \sim 10^{-15}$.
9. **`test_reactive_expanding_fireball`**: Radial Hubble expansion with coupled elastic, inelastic, and resonance decay dynamics.

## Roadmap
- [x] Covariant $2 \to 2$ elastic scattering for $p, n, d, d^*, t, {}^3\text{He}, {}^4\text{He}$, and 19 excited $A=4^*$ states on CUDA
- [x] Relativistic Breit–Wigner spectral functions and $2 \leftrightarrow 1$ resonance formation/decay ($d^* \leftrightarrow N+N$, ${}^4\text{He}^* \leftrightarrow p+t / n+{}^3\text{He} / d+d$, ${}^4\text{H}^* \leftrightarrow n+t$, ${}^4\text{Li}^* \leftrightarrow p+{}^3\text{He}$)
- [x] Reversible $2 \leftrightarrow 2$ nuclear transfer ($dd \leftrightarrow pt/n\,{}^3\text{He}$, $dt \leftrightarrow n\alpha$, $d\,{}^3\text{He} \leftrightarrow p\alpha$), deuteron catalysis ($N+d \leftrightarrow N+d^*$), and ${}^4\text{He} \leftrightarrow {}^4\text{He}^*$ excitation with off-shell detailed balance
- [ ] Direct multi-pion & 3-body $2 \leftrightarrow 3$ channels ($\pi + d \leftrightarrow \pi + p + n$, $N + N + N \leftrightarrow N + d$) and nuclear mean-field potentials
