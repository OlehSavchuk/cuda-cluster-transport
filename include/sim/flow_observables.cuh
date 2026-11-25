#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "backend/philox_rng.cuh"
#include "physics/constants.cuh"
#include "physics/kinematics.cuh"
#include "physics/resonances.cuh"
#include "physics/species.cuh"
#include "sim/particles.cuh"

namespace transport {

// Force all remaining unstable resonances (d*, 4He*, 4H*, 4Li*) at freeze-out to decay
// into asymptotic ground-state daughters (p, n, d, t, 3He) prior to FOPI flow analysis.
inline std::vector<HostParticle> decay_all_unstable_for_detection(
    const std::vector<HostParticle>& in_particles, std::uint64_t seed) {
    std::vector<HostParticle> out;
    out.reserve(in_particles.size() + 2048);

    for (std::size_t i = 0; i < in_particles.size(); ++i) {
        const HostParticle& hp = in_particles[i];
        const SpeciesInfo sp = get_species_info(hp.species);
        if (!sp.is_excited) {
            out.push_back(hp);
            continue;
        }

        PhiloxRng rng(seed ^ 0xFACE1234ULL, 999ULL, static_cast<std::uint64_t>(i));
        const ResonanceDecaySpec dspec = get_decay_spec(hp.species);
        double br_w[3] = {0.0, 0.0, 0.0}, br_sum = 0.0;
        for (int b = 0; b < dspec.n_branches; ++b) {
            br_w[b] = branch_partial_width_gev(hp.species, b, hp.mass);
            br_sum += br_w[b];
        }
        int chosen = 0;
        if (br_sum > 0.0) {
            const double tgt = rng.uniform() * br_sum;
            double acc = 0.0;
            for (int b = 0; b < dspec.n_branches; ++b) {
                acc += br_w[b];
                if (tgt <= acc) { chosen = b; break; }
            }
        }
        const int d1 = dspec.branches[chosen].d1;
        const int d2 = dspec.branches[chosen].d2;
        const double ma = get_species_info(d1).pole_mass_gev;
        const double mb = get_species_info(d2).pole_mass_gev;

        FourVector pa{}, pb{};
        const double cth = 1.0 - 2.0 * rng.uniform();
        const double phi = phys::TWO_PI * rng.uniform();
        const double m_eff = (hp.mass > ma + mb + 1.0e-6) ? hp.mass : (ma + mb + 1.0e-4);
        const double E_eff = std::sqrt(m_eff * m_eff + hp.px * hp.px + hp.py * hp.py + hp.pz * hp.pz);
        decay_isotropic_1to2(FourVector{E_eff, hp.px, hp.py, hp.pz}, m_eff, ma, mb, cth, phi, pa, pb);

        HostParticle ha = hp, hb = hp;
        ha.species = d1; ha.mass = ma; ha.E = pa.t; ha.px = pa.x; ha.py = pa.y; ha.pz = pa.z;
        hb.species = d2; hb.mass = mb; hb.E = pb.t; hb.px = pb.x; hb.py = pb.y; hb.pz = pb.z;
        out.push_back(ha);
        out.push_back(hb);
    }
    return out;
}

inline void compute_and_write_fopi_flow(const std::vector<HostParticle>& raw_particles,
                                        int n_test,
                                        double y_cm_beam,
                                        const std::string& flow_csv_path,
                                        std::uint64_t seed) {
    const auto det = decay_all_unstable_for_detection(raw_particles, seed);
    const double y0_norm = (y_cm_beam > 1.0e-6) ? y_cm_beam : 0.3464;

    // 7 categories: 0=p, 1=n, 2=d, 3=t, 4=3He, 5=A3(t+3He), 6=4He
    constexpr int N_CAT = 7;
    const char* cat_names[N_CAT] = {"p", "n", "d", "t", "3He", "A3", "4He"};

    constexpr int N_YBINS = 18;
    constexpr double Y0_MIN = -1.44, Y0_MAX = 1.44;
    const double dy0 = (Y0_MAX - Y0_MIN) / N_YBINS;
    const double dy_phys = dy0 * y0_norm;

    std::vector<double> cnt(N_CAT * N_YBINS, 0.0);
    std::vector<double> v1_sum(N_CAT * N_YBINS, 0.0), v1_sq(N_CAT * N_YBINS, 0.0);
    std::vector<double> v2_sum(N_CAT * N_YBINS, 0.0), v2_sq(N_CAT * N_YBINS, 0.0);
    std::vector<double> pt_cnt(N_CAT * N_YBINS, 0.0);

    double mid_v2_sum[N_CAT] = {}, mid_v2_cnt[N_CAT] = {};
    double tot_yield[N_CAT] = {};

    auto add_to_cat = [&](int cat, int yb, double y0, double pt, double px, double py) {
        tot_yield[cat] += 1.0 / n_test;
        if (yb >= 0 && yb < N_YBINS) {
            const int k = cat * N_YBINS + yb;
            cnt[k] += 1.0;
            if (pt > 1.0e-5) {
                const double c1 = px / pt;
                const double c2 = (px * px - py * py) / (pt * pt);
                v1_sum[k] += c1; v1_sq[k] += c1 * c1;
                v2_sum[k] += c2; v2_sq[k] += c2 * c2;
                pt_cnt[k] += 1.0;
            }
        }
        if (std::fabs(y0) < 0.45 && pt > 0.05) {
            mid_v2_sum[cat] += (px * px - py * py) / (pt * pt);
            mid_v2_cnt[cat] += 1.0;
        }
    };

    for (const auto& hp : det) {
        if (hp.E <= std::fabs(hp.pz) + 1.0e-9) continue;
        const double y = 0.5 * std::log((hp.E + hp.pz) / (hp.E - hp.pz));
        const double y0 = y / y0_norm;
        const int yb = static_cast<int>(std::floor((y0 - Y0_MIN) / dy0));
        const double pt = std::sqrt(hp.px * hp.px + hp.py * hp.py);

        if (hp.species == PROTON)       add_to_cat(0, yb, y0, pt, hp.px, hp.py);
        else if (hp.species == NEUTRON) add_to_cat(1, yb, y0, pt, hp.px, hp.py);
        else if (hp.species == DEUTERON)add_to_cat(2, yb, y0, pt, hp.px, hp.py);
        else if (hp.species == TRITON) {
            add_to_cat(3, yb, y0, pt, hp.px, hp.py);
            add_to_cat(5, yb, y0, pt, hp.px, hp.py);
        } else if (hp.species == HELION) {
            add_to_cat(4, yb, y0, pt, hp.px, hp.py);
            add_to_cat(5, yb, y0, pt, hp.px, hp.py);
        } else if (hp.species == ALPHA) add_to_cat(6, yb, y0, pt, hp.px, hp.py);
    }

    std::printf("\n=== FOPI Freeze-Out Yields & Midrapidity Squeeze-Out v2 (|y/y_beam| < 0.45) ===\n");
    for (int c = 0; c < N_CAT; ++c) {
        const double mv2 = (mid_v2_cnt[c] > 0) ? (mid_v2_sum[c] / mid_v2_cnt[c]) : 0.0;
        const double ev2 = (mid_v2_cnt[c] > 1) ? std::sqrt((0.5) / mid_v2_cnt[c]) : 0.0;
        std::printf("  %-4s : Yield/ev = %7.2f | v2(|y0|<0.45) = %+7.4f +- %.4f (N_test_mid = %.0f)\n",
                    cat_names[c], tot_yield[c], mv2, ev2, mid_v2_cnt[c]);
    }

    if (!flow_csv_path.empty()) {
        std::ofstream f(flow_csv_path);
        f << "species,bin,y,y0,dN_dy,dN_dy0,v1,v1_err,v2,v2_err,n_stat\n";
        for (int c = 0; c < N_CAT; ++c) {
            for (int b = 0; b < N_YBINS; ++b) {
                const int k = c * N_YBINS + b;
                const double y0_c = Y0_MIN + (b + 0.5) * dy0;
                const double y_c = y0_c * y0_norm;
                const double dn_dy = (cnt[k] / n_test) / dy_phys;
                const double dn_dy0 = (cnt[k] / n_test) / dy0;
                const double n_st = pt_cnt[k];
                const double m_v1 = (n_st > 0) ? (v1_sum[k] / n_st) : 0.0;
                const double m_v2 = (n_st > 0) ? (v2_sum[k] / n_st) : 0.0;
                const double e_v1 = (n_st > 1) ? std::sqrt(std::max(0.0, v1_sq[k] / n_st - m_v1 * m_v1) / n_st) : 0.0;
                const double e_v2 = (n_st > 1) ? std::sqrt(std::max(0.0, v2_sq[k] / n_st - m_v2 * m_v2) / n_st) : 0.0;
                f << cat_names[c] << "," << b << "," << y_c << "," << y0_c << ","
                  << dn_dy << "," << dn_dy0 << ","
                  << m_v1 << "," << e_v1 << "," << m_v2 << "," << e_v2 << "," << n_st << "\n";
            }
        }
    }
}

}  // namespace transport
