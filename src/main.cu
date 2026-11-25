#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"
#include "physics/cross_sections.cuh"
#include "physics/kinematics.cuh"
#include "physics/species.cuh"
#include "sim/grid.cuh"
#include "sim/init_conditions.cuh"
#include "sim/observables.cuh"
#include "sim/particles.cuh"
#include "sim/stepper.cuh"

namespace {

using namespace transport;

#pragma pack(push, 1)
struct FrameHeader {
    std::uint32_t magic = 0x43545250U;  // "CTRP"
    std::int32_t step = 0;
    std::int32_t n_saved = 0;
    std::int32_t n_total = 0;
    double t_fm = 0.0;
    double E_tot_gev = 0.0;
    double aniso_pz_pt = 0.0;
    double rms_r_fm = 0.0;
    std::uint64_t collisions_cum = 0;
    double box_xmin = 0.0, box_xmax = 0.0;
    double box_ymin = 0.0, box_ymax = 0.0;
    double box_zmin = 0.0, box_zmax = 0.0;
};

struct FrameParticleRecord {
    float x, y, z;
    float px, py, pz;
    float E, mass;
    std::int16_t species;
    std::int16_t collided;
};
#pragma pack(pop)

void write_binary_frame(const std::string& dump_dir,
                        int frame_idx,
                        int step,
                        double t_fm,
                        const GridConfig& gcfg,
                        const GlobalObservables& obs,
                        unsigned long long collisions_cum,
                        const std::vector<HostParticle>& host_p,
                        int max_save) {
    std::filesystem::create_directories(dump_dir);
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/frame_%04d.bin", dump_dir.c_str(), frame_idx);

    const int n_total = static_cast<int>(host_p.size());
    const int stride = (max_save > 0 && n_total > max_save) ? (n_total / max_save) : 1;

    std::vector<FrameParticleRecord> recs;
    recs.reserve((max_save > 0 && max_save < n_total) ? max_save : n_total);
    for (int i = 0; i < n_total; i += stride) {
        if (max_save > 0 && static_cast<int>(recs.size()) >= max_save) break;
        const HostParticle& hp = host_p[i];
        FrameParticleRecord r{};
        r.x = static_cast<float>(hp.x);
        r.y = static_cast<float>(hp.y);
        r.z = static_cast<float>(hp.z);
        r.px = static_cast<float>(hp.px);
        r.py = static_cast<float>(hp.py);
        r.pz = static_cast<float>(hp.pz);
        r.E = static_cast<float>(hp.E);
        r.mass = static_cast<float>(hp.mass);
        r.species = static_cast<std::int16_t>(hp.species);
        r.collided = static_cast<std::int16_t>(hp.collided);
        recs.push_back(r);
    }

    FrameHeader hdr{};
    hdr.step = step;
    hdr.n_saved = static_cast<std::int32_t>(recs.size());
    hdr.n_total = n_total;
    hdr.t_fm = t_fm;
    hdr.E_tot_gev = obs.E_tot;
    hdr.aniso_pz_pt = obs.pressure_anisotropy();
    hdr.rms_r_fm = obs.rms_radius(n_total);
    hdr.collisions_cum = collisions_cum;
    hdr.box_xmin = gcfg.xmin; hdr.box_xmax = gcfg.xmax;
    hdr.box_ymin = gcfg.ymin; hdr.box_ymax = gcfg.ymax;
    hdr.box_zmin = gcfg.zmin; hdr.box_zmax = gcfg.zmax;

    std::ofstream out(fname, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(FrameHeader));
    if (!recs.empty()) {
        out.write(reinterpret_cast<const char*>(recs.data()),
                  static_cast<std::streamsize>(recs.size() * sizeof(FrameParticleRecord)));
    }
}

void dump_cross_section_curves(const std::string& path, const CrossSectionConfig& xs_cfg) {
    std::ofstream f(path);
    f << "p_lab_gev,pp_mb,pn_mb,pd_mb,pdstar_mb,pt_mb,p4He_mb,p4He2021_mb,4He4He_mb\n";

    auto eval_pair = [&](int id_proj, int id_targ, double p_lab) {
        const double m1 = get_species_info(id_proj).pole_mass_gev;
        const double m2 = get_species_info(id_targ).pole_mass_gev;
        const FourVector p1{std::sqrt(m1 * m1 + p_lab * p_lab), 0.0, 0.0, p_lab};
        const FourVector p2{m2, 0.0, 0.0, 0.0};
        const PairKinematics kin = compute_pair_kinematics(p1, m1, p2, m2);
        return elastic_cross_section_fm2(id_proj, id_targ, kin, xs_cfg) * phys::FM2_TO_MB;
    };

    constexpr int N_PTS = 200;
    for (int i = 0; i < N_PTS; ++i) {
        const double frac = static_cast<double>(i) / static_cast<double>(N_PTS - 1);
        const double p_lab = 0.04 * std::pow(75.0, frac);  // 0.04 to 3.0 GeV/c log-spaced
        f << p_lab << ","
          << eval_pair(PROTON, PROTON, p_lab) << ","
          << eval_pair(PROTON, NEUTRON, p_lab) << ","
          << eval_pair(PROTON, DEUTERON, p_lab) << ","
          << eval_pair(PROTON, D_STAR_PN, p_lab) << ","
          << eval_pair(PROTON, TRITON, p_lab) << ","
          << eval_pair(PROTON, ALPHA, p_lab) << ","
          << eval_pair(PROTON, HE4_2021, p_lab) << ","
          << eval_pair(ALPHA, ALPHA, 4.0 * p_lab) << "\n";
    }
}

void print_species_catalog(const CrossSectionConfig& xs_cfg) {
    std::printf("\n=== Nucleon & Light-Cluster Species Catalog (A <= 4, Ground & Excited) ===\n");
    std::printf("%-4s %-13s %2s %2s %5s %3s %2s %10s %8s %9s %7s %11s\n",
                "ID", "Name", "A", "Z", "J^P", "2I", "g", "Mass[GeV]",
                "Ex[MeV]", "Gam[MeV]", "R[fm]", "sig_pX[mb]");
    std::printf("-------------------------------------------------------------------------------------\n");

    const double mp = phys::MASS_PROTON;
    const double p_beam = 0.50;
    const FourVector p_proj{std::sqrt(mp * mp + p_beam * p_beam), 0.0, 0.0, p_beam};

    for (int id = 0; id < NUM_SPECIES; ++id) {
        const SpeciesInfo sp = get_species_info(id);
        const FourVector p_targ{sp.pole_mass_gev, 0.0, 0.0, 0.0};
        const PairKinematics kin =
            compute_pair_kinematics(p_proj, mp, p_targ, sp.pole_mass_gev);
        const double sig_mb =
            elastic_cross_section_fm2(PROTON, id, kin, xs_cfg) * phys::FM2_TO_MB;

        char jp_buf[16];
        if (sp.two_J % 2 == 0) {
            std::snprintf(jp_buf, sizeof(jp_buf), "%d%c",
                          sp.two_J / 2, (sp.parity > 0) ? '+' : '-');
        } else {
            std::snprintf(jp_buf, sizeof(jp_buf), "%d/2%c",
                          sp.two_J, (sp.parity > 0) ? '+' : '-');
        }

        std::printf("%-4d %-13s %2d %2d %5s %3d %2d %10.6f %8.2f %9.2f %7.2f %11.2f\n",
                    sp.id, get_species_name(id), sp.baryon_num, sp.charge, jp_buf, sp.two_I,
                    sp.degeneracy, sp.pole_mass_gev, sp.excitation_mev,
                    sp.width_gev * 1.0e3, sp.rms_radius_fm, sig_mb);
    }
    std::printf("  * sig_pX[mb] evaluated for p (p_lab = 0.50 GeV/c) + X(at rest)\n\n");
}

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --mode <aniso|box|fireball|slabs>  Simulation scenario (default: aniso)\n"
        "  --npart <int>                      Total test particles (default: 32000)\n"
        "  --ntest <int>                      Test-particle oversampling N_test (default: 25)\n"
        "  --steps <int>                      Number of time steps (default: 100)\n"
        "  --dt <float>                       Time step [fm/c] (default: 0.20)\n"
        "  --substeps <int>                   Collision substeps per dt (default: 2)\n"
        "  --box <float>                      Box length L [fm] (default: 24.0)\n"
        "  --grid <int>                       Grid cells per axis Nx=Ny=Nz (default: 12)\n"
        "  --temp <float>                     Temperature T [MeV] (default: 50.0)\n"
        "  --cluster-boost <float>            Statistical cluster boost (default: 2.5)\n"
        "  --excited-boost <float>            Statistical excited-state boost (default: 4.0)\n"
        "  --xs <glauber|const>               Cross-section model (default: glauber)\n"
        "  --sigma-mb <float>                 Constant cross section [mb] (default: 40.0)\n"
        "  --ang <forward|iso>                Angular distribution (default: forward)\n"
        "  --scheme <disjoint|allpairs>       Cell collision kernel (default: disjoint)\n"
        "  --csv <path>                       Write time-series observables to CSV\n"
        "  --dump-dir <dir>                   Write binary particle trajectory frames\n"
        "  --dump-every <int>                 Steps between trajectory frames (default: 1)\n"
        "  --dump-limit <int>                 Max particles saved per frame (default: 4000)\n"
        "  --dump-xs <csv_path>               Export cross-section curves vs p_lab and exit\n"
        "  --print-species                    Print full 28-species catalog and exit\n"
        "  --help                             Show this help message\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace transport;

    std::string mode = "aniso";
    int n_particles = 32000;
    int n_test = 25;
    int n_steps = 100;
    double dt_fm = 0.20;
    int substeps = 2;
    double box_L = 24.0;
    int grid_n = 12;
    double temp_mev = 50.0;
    double cluster_boost = 2.5;
    double excited_boost = 4.0;
    std::string csv_path;
    std::string dump_dir;
    int dump_every = 1;
    int dump_limit = 4000;
    std::string dump_xs_path;
    bool only_print_species = false;

    CrossSectionConfig xs_cfg{};
    StepperConfig step_cfg{};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_str = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--mode") {
            mode = next_str();
        } else if (arg == "--npart") {
            n_particles = std::stoi(next_str());
        } else if (arg == "--ntest") {
            n_test = std::stoi(next_str());
        } else if (arg == "--steps") {
            n_steps = std::stoi(next_str());
        } else if (arg == "--dt") {
            dt_fm = std::stod(next_str());
        } else if (arg == "--substeps") {
            substeps = std::stoi(next_str());
        } else if (arg == "--box") {
            box_L = std::stod(next_str());
        } else if (arg == "--grid") {
            grid_n = std::stoi(next_str());
        } else if (arg == "--temp") {
            temp_mev = std::stod(next_str());
        } else if (arg == "--cluster-boost") {
            cluster_boost = std::stod(next_str());
        } else if (arg == "--excited-boost") {
            excited_boost = std::stod(next_str());
        } else if (arg == "--xs") {
            const std::string v = next_str();
            xs_cfg.model = (v == "const") ? CrossSectionModel::ConstantIsotropic
                                          : CrossSectionModel::ParametricGlauber;
        } else if (arg == "--sigma-mb") {
            xs_cfg.sigma_const_mb = std::stod(next_str());
        } else if (arg == "--ang") {
            const std::string v = next_str();
            xs_cfg.angular_mode = (v == "iso") ? AngularMode::Isotropic
                                               : AngularMode::ForwardDiffractive;
        } else if (arg == "--scheme") {
            const std::string v = next_str();
            step_cfg.scheme = (v == "allpairs") ? CollisionScheme::AllPairs
                                                : CollisionScheme::DisjointPairs;
        } else if (arg == "--csv") {
            csv_path = next_str();
        } else if (arg == "--dump-dir") {
            dump_dir = next_str();
        } else if (arg == "--dump-every") {
            dump_every = std::stoi(next_str());
        } else if (arg == "--dump-limit") {
            dump_limit = std::stoi(next_str());
        } else if (arg == "--dump-xs") {
            dump_xs_path = next_str();
        } else if (arg == "--print-species") {
            only_print_species = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (only_print_species) {
        print_species_catalog(xs_cfg);
        return 0;
    }
    if (!dump_xs_path.empty()) {
        dump_cross_section_curves(dump_xs_path, xs_cfg);
        return 0;
    }

    step_cfg.dt_fm = dt_fm;
    step_cfg.n_test = n_test;
    step_cfg.collision_substeps = substeps;

    GridConfig gcfg{};
    gcfg.nx = grid_n;
    gcfg.ny = grid_n;
    gcfg.nz = grid_n;

    MixtureConfig mix{};
    mix.cluster_yield_boost = cluster_boost;
    mix.excited_yield_boost = excited_boost;

    const double T_gev = temp_mev * phys::MEV_TO_GEV;
    const double mu_B_gev = 0.720;
    const double mu_Q_gev = -0.008;

    std::vector<HostParticle> host_init;
    if (mode == "fireball") {
        gcfg.boundary = BoundaryMode::OpenExpanding;
        const double half_L = 1.5 * box_L;
        gcfg.xmin = -half_L; gcfg.xmax = half_L;
        gcfg.ymin = -half_L; gcfg.ymax = half_L;
        gcfg.zmin = -half_L; gcfg.zmax = half_L;
        host_init = init_expanding_fireball(
            n_particles, /*radius_rms_fm=*/5.0, /*beta_surface=*/0.38,
            T_gev, mu_B_gev, mu_Q_gev, mix, step_cfg.seed);
    } else if (mode == "slabs") {
        gcfg.boundary = BoundaryMode::PeriodicBox;
        gcfg.xmin = 0.0; gcfg.xmax = box_L;
        gcfg.ymin = 0.0; gcfg.ymax = box_L;
        gcfg.zmin = 0.0; gcfg.zmax = box_L;
        host_init = init_colliding_slabs(
            n_particles, gcfg, /*beta_beam=*/0.48, /*T_internal=*/0.35 * T_gev,
            mu_B_gev, mu_Q_gev, mix, step_cfg.seed);
    } else if (mode == "box") {
        gcfg.boundary = BoundaryMode::PeriodicBox;
        gcfg.xmin = 0.0; gcfg.xmax = box_L;
        gcfg.ymin = 0.0; gcfg.ymax = box_L;
        gcfg.zmin = 0.0; gcfg.zmax = box_L;
        host_init = init_box_ensemble(
            n_particles, gcfg, T_gev, T_gev, mu_B_gev, mu_Q_gev, mix, step_cfg.seed);
    } else {
        gcfg.boundary = BoundaryMode::PeriodicBox;
        gcfg.xmin = 0.0; gcfg.xmax = box_L;
        gcfg.ymin = 0.0; gcfg.ymax = box_L;
        gcfg.zmin = 0.0; gcfg.zmax = box_L;
        host_init = init_box_ensemble(
            n_particles, gcfg, /*T_perp=*/0.55 * T_gev, /*T_z=*/1.85 * T_gev,
            mu_B_gev, mu_Q_gev, mix, step_cfg.seed);
    }

    std::printf("===========================================================================\n");
    std::printf(" CUDA Relativistic Nucleon & Light-Cluster Transport (Elastic 2 -> 2)\n");
    std::printf("===========================================================================\n");
    std::printf(" Backend        : %s\n", backend_description().c_str());
    std::printf(" Scenario       : %s (%s)\n", mode.c_str(),
                (gcfg.boundary == BoundaryMode::PeriodicBox) ? "3D Periodic Box" : "Open Expanding");
    std::printf(" Particles      : %d test particles (N_test = %d -> %.1f physical particles)\n",
                n_particles, n_test, static_cast<double>(n_particles) / n_test);
    std::printf(" Grid           : %dx%dx%d = %d cells, dV = %.3f fm^3\n",
                gcfg.nx, gcfg.ny, gcfg.nz, gcfg.num_cells(), gcfg.cell_volume_fm3());
    std::printf(" Time Stepping  : %d steps, dt = %.3f fm/c (%d substeps/step), scheme = %s\n",
                n_steps, dt_fm, substeps,
                (step_cfg.scheme == CollisionScheme::DisjointPairs) ? "DisjointPairs" : "AllPairs");
    std::printf(" Cross Sections : %s (%s)\n",
                (xs_cfg.model == CrossSectionModel::ParametricGlauber) ? "ERE+KK+Cugnon+Glauber" : "Constant",
                (xs_cfg.angular_mode == AngularMode::ForwardDiffractive) ? "ForwardDiffractive" : "Isotropic");

    ParticleSystem particles;
    particles.upload(host_init);

    SpatialGrid grid(gcfg);
    TransportStepper stepper(step_cfg, xs_cfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);

    int n_nuc = obs0.counts[PROTON] + obs0.counts[NEUTRON];
    int n_d = obs0.counts[DEUTERON];
    int n_dstar = obs0.counts[D_STAR_NN] + obs0.counts[D_STAR_PN] + obs0.counts[D_STAR_PP];
    int n_a3 = obs0.counts[TRITON] + obs0.counts[HELION];
    int n_he4_gs = obs0.counts[ALPHA];
    int n_a4_exc = 0;
    for (int id = HE4_2021; id < NUM_SPECIES; ++id) n_a4_exc += obs0.counts[id];

    std::printf(" Initial Yields : N(p,n)=%d | d=%d, d*=%d | (t,3He)=%d | 4He=%d, A=4*(19 states)=%d\n",
                n_nuc, n_d, n_dstar, n_a3, n_he4_gs, n_a4_exc);
    std::printf("===========================================================================\n");
    std::printf("%6s %8s %13s %10s %10s %12s %9s\n",
                "Step", "t[fm/c]", "E_tot[GeV]", "Pz/P_perp", "R_rms[fm]", "Collisions", "Max(P22)");
    std::printf("---------------------------------------------------------------------------\n");

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "step,t_fm,E_tot_gev,Px_tot,Py_tot,Pz_tot,Pz_over_Pperp,R_rms_fm,collisions_cum,max_P22\n";
    }

    const int report_every = (n_steps >= 10) ? (n_steps / 10) : 1;
    int frame_idx = 0;
    const auto t_wall_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= n_steps; ++step) {
        const bool need_obs = (step % report_every == 0) || (step == n_steps) ||
                              (!dump_dir.empty() && (step % dump_every == 0));

        if (need_obs) {
            const GlobalObservables obs = obs_calc.compute(particles);
            const StepStats st = stepper.read_stats();
            const double t_now = step * dt_fm;

            if (step % report_every == 0 || step == n_steps) {
                std::printf("%6d %8.2f %13.6f %10.4f %10.4f %12llu %9.4f\n",
                            step, t_now, obs.E_tot, obs.pressure_anisotropy(),
                            obs.rms_radius(n_particles), st.collisions_accepted, st.max_prob_seen);

                if (csv.is_open()) {
                    csv << step << "," << t_now << "," << obs.E_tot << ","
                        << obs.Px_tot << "," << obs.Py_tot << "," << obs.Pz_tot << ","
                        << obs.pressure_anisotropy() << "," << obs.rms_radius(n_particles) << ","
                        << st.collisions_accepted << "," << st.max_prob_seen << "\n";
                }
            }

            if (!dump_dir.empty() && (step % dump_every == 0)) {
                const auto host_snap = particles.download();
                write_binary_frame(dump_dir, frame_idx++, step, t_now, gcfg,
                                   obs, st.collisions_accepted, host_snap, dump_limit);
            }
        }

        if (step < n_steps) {
            stepper.step(particles, grid);
        }
    }

    device_synchronize();
    const auto t_wall_end = std::chrono::high_resolution_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    const GlobalObservables obs_final = obs_calc.compute(particles);
    const StepStats final_stats = stepper.read_stats();
    const double rel_dE = std::fabs(obs_final.E_tot - obs0.E_tot) / obs0.E_tot;
    const double dP_mag = std::sqrt(
        (obs_final.Px_tot - obs0.Px_tot) * (obs_final.Px_tot - obs0.Px_tot) +
        (obs_final.Py_tot - obs0.Py_tot) * (obs_final.Py_tot - obs0.Py_tot) +
        (obs_final.Pz_tot - obs0.Pz_tot) * (obs_final.Pz_tot - obs0.Pz_tot));

    std::printf("---------------------------------------------------------------------------\n");
    std::printf(" Conservation   : |dE|/E = %.3e | |dP| = %.3e GeV/c | dB = %lld | dQ = %lld\n",
                rel_dE, dP_mag,
                obs_final.baryon_tot - obs0.baryon_tot,
                obs_final.charge_tot - obs0.charge_tot);
    std::printf(" Collision Stat : %llu elastic scatters out of %llu candidate pairs (overflows: %llu)\n",
                final_stats.collisions_accepted, final_stats.pairs_tested, final_stats.prob_overflows);
    std::printf(" Wall Time      : %.3f s (%.2f ms/step)\n",
                elapsed_s, 1000.0 * elapsed_s / (n_steps > 0 ? n_steps : 1));
    std::printf("===========================================================================\n");

    return 0;
}
