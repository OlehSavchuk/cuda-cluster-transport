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
#include "physics/inelastic_channels.cuh"
#include "physics/kinematics.cuh"
#include "physics/mean_field.cuh"
#include "physics/resonances.cuh"
#include "physics/species.cuh"
#include "sim/flow_observables.cuh"
#include "sim/freezeout_clusters.cuh"
#include "sim/grid.cuh"
#include "sim/hcgf_io.cuh"
#include "sim/hydro_io.cuh"
#include "sim/init_conditions.cuh"
#include "sim/observables.cuh"
#include "sim/particles.cuh"
#include "sim/stepper.cuh"

namespace {

using namespace transport;

#pragma pack(push, 1)
struct FrameHeader {
    std::uint32_t magic = 0x43545232U;  // "CTR2"
    std::int32_t step = 0;
    std::int32_t n_saved = 0;
    std::int32_t n_total = 0;
    double t_fm = 0.0;
    double E_tot_gev = 0.0;
    double aniso_pz_pt = 0.0;
    double rms_r_fm = 0.0;
    std::uint64_t el_cum = 0;
    std::uint64_t inel_cum = 0;
    std::uint64_t form_cum = 0;
    std::uint64_t decay_cum = 0;
    std::int32_t count_nuc = 0;
    std::int32_t count_d = 0;
    std::int32_t count_dstar = 0;
    std::int32_t count_a3 = 0;
    std::int32_t count_he4 = 0;
    std::int32_t count_a4star = 0;
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
                        const StepStats& st,
                        const std::vector<HostParticle>& host_p,
                        int max_save) {
    std::filesystem::create_directories(dump_dir);
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/frame_%04d.bin", dump_dir.c_str(), frame_idx);

    const int n_total = static_cast<int>(host_p.size());
    const int stride = (max_save > 0 && n_total > max_save) ? (n_total / max_save) : 1;

    // Preserve rare clusters in visual subsample! Always save clusters/excited states first
    // if room permits, then fill remaining quota with nucleons at uniform stride.
    std::vector<FrameParticleRecord> recs;
    const int cap = (max_save > 0 && max_save < n_total) ? max_save : n_total;
    recs.reserve(cap);

    auto push_rec = [&](const HostParticle& hp) {
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
    };

    if (stride == 1) {
        for (int i = 0; i < n_total; ++i) push_rec(host_p[i]);
    } else {
        for (int i = 0; i < n_total; i += stride) {
            if (static_cast<int>(recs.size()) >= cap) break;
            push_rec(host_p[i]);
        }
    }

    int n_a4_exc = 0;
    for (int id = 0; id < NUM_SPECIES; ++id) {
        if (is_a4_excited(id)) n_a4_exc += obs.counts[id];
    }

    FrameHeader hdr{};
    hdr.step = step;
    hdr.n_saved = static_cast<std::int32_t>(recs.size());
    hdr.n_total = obs.n_active;
    hdr.t_fm = t_fm;
    hdr.E_tot_gev = obs.E_tot;
    hdr.aniso_pz_pt = obs.pressure_anisotropy();
    hdr.rms_r_fm = obs.rms_radius();
    hdr.el_cum = st.collisions_elastic;
    hdr.inel_cum = st.collisions_inelastic_2to2;
    hdr.form_cum = st.formations_2to1;
    hdr.decay_cum = st.decays_1to2;
    hdr.count_nuc = obs.counts[PROTON] + obs.counts[NEUTRON];
    hdr.count_d = obs.counts[DEUTERON];
    hdr.count_dstar = obs.counts[D_STAR_NN] + obs.counts[D_STAR_PN] + obs.counts[D_STAR_PP];
    hdr.count_a3 = obs.counts[TRITON] + obs.counts[HELION];
    hdr.count_he4 = obs.counts[ALPHA];
    hdr.count_a4star = n_a4_exc;
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
        const double p_lab = 0.04 * std::pow(75.0, frac);
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

void dump_breit_wigner_curves(const std::string& path) {
    const SpectralNormTable norms = compute_spectral_norm_table();
    std::ofstream f(path);
    f << "species_id,name,m_gev,dm_above_th_mev,A_gev_inv,gamma_tot_mev\n";

    const int showcase_ids[] = {D_STAR_PN, HE4_2021, HE4_2184, HE4_2425, HE4_2595, HE4_2742, H4_RES, DELTA_PLUS};
    for (int id : showcase_ids) {
        const SpeciesInfo sp = get_species_info(id);
        const double m_min = resonance_min_mass(id);
        const double m_0 = sp.pole_mass_gev;
        const double span = (3.0 * sp.width_gev > 0.045) ? (3.0 * sp.width_gev) : 0.045;
        const double m_hi = m_0 + span;
        constexpr int N_PTS = 300;
        for (int k = 0; k < N_PTS; ++k) {
            const double m = m_min + (m_hi - m_min) * (static_cast<double>(k) + 0.5) / N_PTS;
            const double A = spectral_function(id, m, norms);
            const double gam_mev = total_width_gev(id, m) * 1.0e3;
            f << id << "," << get_species_name(id) << ","
              << m << "," << (m - m_min) * 1.0e3 << ","
              << A << "," << gam_mev << "\n";
        }
    }
}

void dump_eos_curves(const std::string& path) {
    const MeanFieldConfig sk_soft = make_mean_field_config(EOSPreset::SkyrmeSoft);
    const MeanFieldConfig md_soft = make_mean_field_config(EOSPreset::MDSoft);
    const MeanFieldConfig md_stiff = make_mean_field_config(EOSPreset::MDStiff);

    std::ofstream f(path);
    f << "u_rho_over_rho0,ea_skyrme_soft_mev,ea_md_soft_mev,ea_md_stiff_mev,"
         "p_gev,uopt_skyrme_soft_mev,uopt_md_soft_mev,uopt_md_stiff_mev\n";

    double alpha1 = 1.0, da = 0.0, d2a = 0.0;
    cold_fermi_md_factors(1.0, md_soft.rho0_fm3, md_soft.Lambda_md_gev, alpha1, da, d2a);

    auto u_opt_rho0 = [&](const MeanFieldConfig& c, double p_gev) {
        const double u_sk = c.A_sk_gev + c.B_sk_gev;
        const double u_md = 0.5 * c.C_md_gev * (alpha1 + 1.0 / (1.0 + (p_gev * p_gev) / (c.Lambda_md_gev * c.Lambda_md_gev)));
        return (u_sk + u_md) * 1.0e3;
    };

    constexpr int N_PTS = 200;
    for (int i = 0; i < N_PTS; ++i) {
        const double u = 0.02 + 3.48 * static_cast<double>(i) / (N_PTS - 1);
        const double p_gev = 1.20 * static_cast<double>(i) / (N_PTS - 1);
        f << u << ","
          << cold_eos_energy_per_nucleon_gev(u, sk_soft) * 1.0e3 << ","
          << cold_eos_energy_per_nucleon_gev(u, md_soft) * 1.0e3 << ","
          << cold_eos_energy_per_nucleon_gev(u, md_stiff) * 1.0e3 << ","
          << p_gev << ","
          << u_opt_rho0(sk_soft, p_gev) << ","
          << u_opt_rho0(md_soft, p_gev) << ","
          << u_opt_rho0(md_stiff, p_gev) << "\n";
    }
}

void print_species_catalog(const CrossSectionConfig& xs_cfg) {
    const SpectralNormTable norms = compute_spectral_norm_table();
    std::printf("\n=== Nucleon & Light-Cluster Species Catalog (A <= 4, Breit-Wigner Spectral Enabled) ===\n");
    std::printf("%-3s %-13s %2s %2s %5s %2s %10s %10s %8s %7s %9s\n",
                "ID", "Name", "A", "Z", "J^P", "g", "Pole[GeV]", "m_th[GeV]",
                "Gam[MeV]", "N_raw", "sig_pX[mb]");
    std::printf("---------------------------------------------------------------------------------------\n");

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

        std::printf("%-3d %-13s %2d %2d %5s %2d %10.6f %10.6f %8.2f %7.4f %9.2f\n",
                    sp.id, get_species_name(id), sp.baryon_num, sp.charge, jp_buf,
                    sp.degeneracy, sp.pole_mass_gev, resonance_min_mass(id),
                    sp.width_gev * 1.0e3, norms.norm[id], sig_mb);
    }
    std::printf("\n");
}

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --mode <aniso|box|fireball|slabs|nucleosynthesis|hic>\n"
        "                                     Simulation scenario (default: nucleosynthesis)\n"
        "  --A-nuc <int>                      HIC nucleus mass number A (default: 197 for Au)\n"
        "  --Z-nuc <int>                      HIC nucleus proton number Z (default: 79 for Au)\n"
        "  --elab-amev <float>                HIC lab kinetic energy [A MeV] (default: 250.0)\n"
        "  --impact-b <float>                 HIC impact parameter b [fm] (default: 5.5)\n"
        "  --flow-csv <path>                  Write FOPI rapidity dN/dy, v1, v2 table to CSV\n"
        "  --dump-hydro-dir <dir>             Write binary 3D energy & baryon density grids\n"
        "  --dump-hydro-every <int>           Steps between binary hydro grid dumps (default: 5)\n"
        "  --dump-tmunu-dir <dir>             Write 24-field HCGF T^munu + J_B^mu + J_Q^mu binary grids\n"
        "  --dump-tmunu-every <int>           Steps between HCGF T^munu dumps (default: 5)\n"
        "  --cluster-mode <microscopic|none|freezeout>\n"
        "                                     Cluster treatment: dynamical, pure N+Delta+pi, or n<nfr thermal\n"
        "  --backflow <remelt|outward|ghost>  Back-emitted freeze-out particle handling (default: remelt)\n"
        "  --max-cluster-A <int>              Max nuclear mass A in freeze-out EoS/sampling (default: 6)\n"
        "  --nfr <float>                      Freeze-out Eckart baryon density n_fr [fm^-3] (default: 0.04)\n"
        "  --freezeout-every <int>            Steps between n<nfr cell checks (default: 2)\n"
        "  --freezeout-tmin <float>           Min time before n<nfr check [fm/c] (<0: auto overlap time)\n"
        "  --freezeout-cells-csv <path>       Write HCGF n<nfr freeze-out hypersurface cells to CSV\n"
        "  --reactions <all|elastic>          Enable full inelastic+BW network or elastic only\n"
        "  --eos <none|skyrme-soft|md-soft|md-stiff>\n"
        "                                     Mean-field EOS + MD potential (default: none)\n"
        "  --in-medium-ec <off|on|exact>      MD energy conservation: per-vertex (on) or +projection (exact)\n"
        "  --inelastic-scale <float>          Scale factor on 2<->2 inelastic couplings (default: 1.0)\n"
        "  --npart <int>                      Initial test particles (default: 32000)\n"
        "  --ntest <int>                      Test-particle oversampling N_test (default: 20)\n"
        "  --steps <int>                      Number of time steps (default: 100)\n"
        "  --dt <float>                       Time step [fm/c] (default: 0.20)\n"
        "  --substeps <int>                   Collision substeps per dt (default: 2)\n"
        "  --box <float>                      Box length L [fm] (default: 20.0)\n"
        "  --grid <int>                       Grid cells per axis Nx=Ny=Nz (default: 10)\n"
        "  --temp <float>                     Temperature T [MeV] (default: 55.0)\n"
        "  --include-pions                    Include thermal pi & Delta(1232) in initial box/fireball\n"
        "  --cluster-boost <float>            Initial statistical cluster boost (default: 1.0)\n"
        "  --excited-boost <float>            Initial statistical excited boost (default: 1.0)\n"
        "  --xs <glauber|const>               Elastic cross-section model (default: glauber)\n"
        "  --sigma-mb <float>                 Constant elastic cross section [mb] (default: 40.0)\n"
        "  --ang <forward|iso>                Elastic angular distribution (default: forward)\n"
        "  --scheme <disjoint|allpairs>       Cell collision kernel (default: disjoint)\n"
        "  --csv <path>                       Write time-series observables to CSV\n"
        "  --dump-dir <dir>                   Write binary particle trajectory frames\n"
        "  --dump-every <int>                 Steps between trajectory frames (default: 1)\n"
        "  --dump-limit <int>                 Max particles saved per frame (default: 4000)\n"
        "  --dump-xs <csv_path>               Export elastic cross-section curves and exit\n"
        "  --dump-bw <csv_path>               Export Breit-Wigner spectral curves and exit\n"
        "  --dump-eos <csv_path>              Export Skyrme + MD EOS & optical potential curves\n"
        "  --print-species                    Print full 28-species catalog and exit\n"
        "  --help                             Show this help message\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace transport;

    std::string mode = "nucleosynthesis";
    std::string rxn_mode = "all";
    std::string eos_name = "none";
    double inelastic_scale = 1.0;
    int n_particles = 32000;
    int n_test = 20;
    int n_steps = 100;
    double dt_fm = 0.20;
    int substeps = 2;
    double box_L = 20.0;
    int grid_n = 10;
    double temp_mev = 55.0;
    bool init_thermal_pions = false;
    double cluster_boost = 1.0;
    double excited_boost = 1.0;
    int A_nuc = 197;
    int Z_nuc = 79;
    double elab_amev = 250.0;
    double impact_b = 5.5;
    std::string flow_csv_path;
    std::string dump_hydro_dir;
    int dump_hydro_every = 5;
    std::string dump_tmunu_dir;
    int dump_tmunu_every = 5;
    std::string cluster_mode_str = "freezeout";
    std::string backflow_str = "remelt";
    bool local_anisotropy = true;
    int pauli_override = -1;  // -1: auto (ON in hic mode, OFF in classical box modes); 0: OFF; 1: ON
    int max_cluster_A = 6;
    double nfr_fm3 = 0.040;
    int freezeout_every = 2;
    double freezeout_tmin = -1.0;
    std::string freezeout_cells_csv;
    std::string csv_path;
    std::string dump_dir;
    int dump_every = 1;
    int dump_limit = 4000;
    std::string dump_xs_path;
    std::string dump_bw_path;
    std::string dump_eos_path;
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
        } else if (arg == "--A-nuc") {
            A_nuc = std::stoi(next_str());
        } else if (arg == "--Z-nuc") {
            Z_nuc = std::stoi(next_str());
        } else if (arg == "--elab-amev") {
            elab_amev = std::stod(next_str());
        } else if (arg == "--impact-b") {
            impact_b = std::stod(next_str());
        } else if (arg == "--flow-csv") {
            flow_csv_path = next_str();
        } else if (arg == "--dump-hydro-dir") {
            dump_hydro_dir = next_str();
        } else if (arg == "--dump-hydro-every") {
            dump_hydro_every = std::stoi(next_str());
        } else if (arg == "--dump-tmunu-dir") {
            dump_tmunu_dir = next_str();
        } else if (arg == "--dump-tmunu-every") {
            dump_tmunu_every = std::stoi(next_str());
        } else if (arg == "--cluster-mode") {
            cluster_mode_str = next_str();
        } else if (arg == "--backflow") {
            backflow_str = next_str();
        } else if (arg == "--anisotropy" || arg == "--local-anisotropy") {
            const std::string v = next_str();
            local_anisotropy = (v == "on" || v == "1" || v == "true" || v == "pt");
        } else if (arg == "--pauli" || arg == "--pauli-blocking") {
            const std::string v = next_str();
            pauli_override = (v == "on" || v == "1" || v == "true") ? 1 : 0;
        } else if (arg == "--max-cluster-A") {
            max_cluster_A = std::stoi(next_str());
        } else if (arg == "--nfr") {
            nfr_fm3 = std::stod(next_str());
        } else if (arg == "--freezeout-every") {
            freezeout_every = std::stoi(next_str());
        } else if (arg == "--freezeout-tmin") {
            freezeout_tmin = std::stod(next_str());
        } else if (arg == "--freezeout-cells-csv") {
            freezeout_cells_csv = next_str();
        } else if (arg == "--reactions") {
            rxn_mode = next_str();
        } else if (arg == "--eos") {
            eos_name = next_str();
        } else if (arg == "--in-medium-ec" || arg == "--md-ec") {
            const std::string v = next_str();
            step_cfg.in_medium_energy_conservation = (v == "on" || v == "1" || v == "true" || v == "exact");
            step_cfg.exact_hamiltonian_projection = (v == "exact");
        } else if (arg == "--inelastic-scale") {
            inelastic_scale = std::stod(next_str());
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
        } else if (arg == "--include-pions") {
            init_thermal_pions = true;
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
        } else if (arg == "--seed") {
            step_cfg.seed = std::stoull(next_str());
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
        } else if (arg == "--dump-bw") {
            dump_bw_path = next_str();
        } else if (arg == "--dump-eos") {
            dump_eos_path = next_str();
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
    if (!dump_bw_path.empty()) {
        dump_breit_wigner_curves(dump_bw_path);
        return 0;
    }
    if (!dump_eos_path.empty()) {
        dump_eos_curves(dump_eos_path);
        return 0;
    }

    EOSPreset eos_preset = EOSPreset::None;
    if (eos_name == "skyrme-soft") eos_preset = EOSPreset::SkyrmeSoft;
    else if (eos_name == "md-soft") eos_preset = EOSPreset::MDSoft;
    else if (eos_name == "md-stiff") eos_preset = EOSPreset::MDStiff;
    const MeanFieldConfig mf_cfg = make_mean_field_config(eos_preset);

    step_cfg.dt_fm = dt_fm;
    step_cfg.n_test = n_test;
    step_cfg.collision_substeps = substeps;
    step_cfg.pauli_blocking = (pauli_override >= 0) ? (pauli_override == 1) : (mode == "hic");

    ClusterMode cluster_mode = ClusterMode::FreezeoutThermal;
    if (cluster_mode_str == "microscopic" || cluster_mode_str == "micro") {
        cluster_mode = ClusterMode::Microscopic;
    } else if (cluster_mode_str == "none" || cluster_mode_str == "hcgf") {
        cluster_mode = ClusterMode::None;
    }

    ReactionConfig rxn_cfg{};
    rxn_cfg.enable_elastic = true;
    rxn_cfg.enable_inelastic_2to2 = (rxn_mode != "elastic");
    rxn_cfg.enable_resonance_2to1_and_decays = (rxn_mode != "elastic");
    rxn_cfg.enable_nuclear_clusters = (cluster_mode == ClusterMode::Microscopic);
    rxn_cfg.inelastic_scale = inelastic_scale;

    GridConfig gcfg{};
    gcfg.nx = grid_n;
    gcfg.ny = grid_n;
    gcfg.nz = grid_n;

    MixtureConfig mix{};
    mix.include_pions = init_thermal_pions;
    mix.include_deltas = init_thermal_pions;
    if (cluster_mode != ClusterMode::Microscopic) {
        mix.include_ground_clusters = false;
        mix.include_excited_dstar = false;
        mix.include_excited_he4 = false;
    }
    mix.use_breit_wigner = (rxn_mode != "elastic");
    mix.cluster_yield_boost = cluster_boost;
    mix.excited_yield_boost = excited_boost;

    const double T_gev = temp_mev * phys::MEV_TO_GEV;
    const double mu_B_gev = 0.780;
    const double mu_Q_gev = -0.006;

    double y_cm_beam = 0.3464;
    std::vector<HostParticle> host_init;
    if (mode == "hic") {
        gcfg.boundary = BoundaryMode::OpenExpanding;
        const double half_L = 0.5 * box_L;
        gcfg.xmin = -half_L; gcfg.xmax = half_L;
        gcfg.ymin = -half_L; gcfg.ymax = half_L;
        gcfg.zmin = -half_L; gcfg.zmax = half_L;
        host_init = init_heavy_ion_collision(
            A_nuc, Z_nuc, n_test, elab_amev, impact_b, step_cfg.seed, y_cm_beam);
        n_particles = static_cast<int>(host_init.size());
    } else if (mode == "nucleosynthesis") {
        // Start from 100% pure protons & neutrons and nucleate d, d*, t, 3He, 4He, 4He* dynamically!
        gcfg.boundary = BoundaryMode::PeriodicBox;
        gcfg.xmin = 0.0; gcfg.xmax = box_L;
        gcfg.ymin = 0.0; gcfg.ymax = box_L;
        gcfg.zmin = 0.0; gcfg.zmax = box_L;
        mix.include_ground_clusters = false;
        mix.include_excited_dstar = false;
        mix.include_excited_he4 = false;
        host_init = init_box_ensemble(
            n_particles, gcfg, T_gev, T_gev, mu_B_gev, mu_Q_gev, mix, step_cfg.seed);
    } else if (mode == "fireball") {
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

    FreezeoutConfig fo_cfg{};
    fo_cfg.mode = cluster_mode;
    if (backflow_str == "outward") fo_cfg.backflow = BackflowMode::OutwardOnly;
    else if (backflow_str == "ghost") fo_cfg.backflow = BackflowMode::GhostFreeStream;
    else fo_cfg.backflow = BackflowMode::RemeltInDense;
    fo_cfg.n_fr_fm3 = nfr_fm3;
    fo_cfg.max_cluster_A = max_cluster_A;
    fo_cfg.local_anisotropy = local_anisotropy;
    fo_cfg.cells_csv_path = freezeout_cells_csv;
    if (freezeout_tmin >= 0.0) {
        fo_cfg.min_time_fm = freezeout_tmin;
        fo_cfg.peak_track_time_fm = std::max(0.0, 0.75 * freezeout_tmin);
    } else if (mode == "hic") {
        const double R_ws = 1.12 * std::cbrt(static_cast<double>(A_nuc));
        const double beta_cm = std::max(0.05, std::tanh(y_cm_beam));
        const double t_overlap = (R_ws + 2.5) / beta_cm;
        fo_cfg.peak_track_time_fm = 0.80 * t_overlap;
        fo_cfg.min_time_fm = t_overlap + 3.0;
    } else {
        fo_cfg.peak_track_time_fm = 0.0;
        fo_cfg.min_time_fm = 0.0;
    }

    std::printf("=======================================================================================\n");
    std::printf(" CUDA Relativistic Hadron & Cluster Transport (HCGF T^munu + Freeze-Out / Microscopic)\n");
    std::printf("=======================================================================================\n");
    std::printf(" Backend        : %s\n", backend_description().c_str());
    std::printf(" Scenario       : %s (%s) | EOS: %s | Cluster Mode: %s (n_fr=%.3f fm^-3)\n",
                mode.c_str(),
                (gcfg.boundary == BoundaryMode::PeriodicBox) ? "3D Periodic Box" : "Open Expanding",
                eos_name.c_str(), cluster_mode_str.c_str(), fo_cfg.n_fr_fm3);
    std::printf(" Initial N_part : %d test particles (N_test = %d)\n", n_particles, n_test);
    std::printf(" Grid           : %dx%dx%d = %d cells, dV = %.3f fm^3\n",
                gcfg.nx, gcfg.ny, gcfg.nz, gcfg.num_cells(), gcfg.cell_volume_fm3());
    std::printf(" Time Stepping  : %d steps, dt = %.3f fm/c (%d substeps/step), scheme = %s | In-Medium EC: %s\n",
                n_steps, dt_fm, substeps,
                (step_cfg.scheme == CollisionScheme::DisjointPairs) ? "DisjointPairs" : "AllPairs",
                !mf_cfg.enabled ? "OFF" : (step_cfg.exact_hamiltonian_projection ? "EXACT"
                                          : (step_cfg.in_medium_energy_conservation ? "ON" : "OFF")));
    std::printf("=======================================================================================\n");
    std::printf("%5s %7s %7s %6s %6s %6s %5s %6s %6s %6s %9s %8s %8s %8s\n",
                "Step", "t[fm/c]", "N(p,n)", "d", "d*", "t+3He", "4He", "4He*", "Delta", "pi",
                "Elast", "Inel22", "Form21", "Decay12");
    std::printf("-----------------------------------------------------------------------------------------------------\n");

    ParticleSystem particles;
    particles.upload(host_init);

    SpatialGrid grid(gcfg);
    MeanFieldGrid mean_field(mf_cfg, gcfg);
    mean_field.rebuild_potentials(particles, n_test);
    TransportStepper stepper(step_cfg, xs_cfg, rxn_cfg);
    FreezeoutCellClusterizer fo_clusterizer(fo_cfg, gcfg);
    ObservableCalculator obs_calc;

    const GlobalObservables obs0 = obs_calc.compute(particles);
    const double E_pot0 = mean_field.compute_potential_energy_test_gev(n_test);
    const double H_tot0 = obs0.E_tot + E_pot0;

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "step,t_fm,n_active,E_kin_gev,E_pot_gev,H_tot_gev,rho_contrast,Pz_over_Pperp,R_rms_fm,"
               "n_nuc,n_d,n_dstar,n_a3,n_he4,n_a4star,n_delta,n_pi,"
               "el_cum,inel22_cum,form21_cum,decay12_cum,max_P\n";
    }

    const int report_every = (n_steps >= 10) ? (n_steps / 10) : 1;
    int frame_idx = 0;
    int hydro_idx = 0;
    int tmunu_idx = 0;
    const auto t_wall_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= n_steps; ++step) {
        const double t_now = step * dt_fm;
        const bool do_tmunu = (!dump_tmunu_dir.empty() && (step % dump_tmunu_every == 0));
        const double fo_warmup_t = fo_cfg.peak_track_time_fm;
        const bool do_fo_check = (cluster_mode != ClusterMode::Microscopic) &&
                                 ((step % freezeout_every == 0 && t_now >= fo_warmup_t) || step == n_steps);
        if (do_fo_check || do_tmunu) {
            const HcgfGridSnapshot tmunu_snap = fo_clusterizer.process_step(
                particles, step, t_now, n_test, mf_cfg, step_cfg.seed, /*force_finalize=*/(step == n_steps));
            if (do_tmunu) {
                write_hcgf_tmunu_binary(dump_tmunu_dir, tmunu_idx++, step, t_now, gcfg, tmunu_snap);
            }
        }

        const bool do_frame = (!dump_dir.empty() && (step % dump_every == 0));
        const bool do_hydro = (!dump_hydro_dir.empty() && (step % dump_hydro_every == 0));
        const bool need_obs = (step % report_every == 0) || (step == n_steps) || do_frame || do_hydro;

        if (need_obs) {
            const GlobalObservables obs = obs_calc.compute(particles);
            const StepStats st = stepper.read_stats();

            const int n_nuc = obs.counts[PROTON] + obs.counts[NEUTRON];
            const int n_d = obs.counts[DEUTERON];
            const int n_dstar = obs.counts[D_STAR_NN] + obs.counts[D_STAR_PN] + obs.counts[D_STAR_PP];
            const int n_a3 = obs.counts[TRITON] + obs.counts[HELION];
            const int n_he4 = obs.counts[ALPHA];
            int n_a4_exc = 0;
            for (int id = 0; id < NUM_SPECIES; ++id) {
                if (is_a4_excited(id)) n_a4_exc += obs.counts[id];
            }
            const int n_delta = obs.counts[DELTA_MINUS] + obs.counts[DELTA_ZERO] +
                                obs.counts[DELTA_PLUS] + obs.counts[DELTA_PLUSPLUS];
            const int n_pi = obs.counts[PION_MINUS] + obs.counts[PION_ZERO] + obs.counts[PION_PLUS];

            if (step % report_every == 0 || step == n_steps) {
                std::printf("%5d %7.2f %7d %6d %6d %6d %5d %6d %6d %6d %9llu %8llu %8llu %8llu\n",
                            step, t_now, n_nuc, n_d, n_dstar, n_a3, n_he4, n_a4_exc, n_delta, n_pi,
                            st.collisions_elastic, st.collisions_inelastic_2to2,
                            st.formations_2to1, st.decays_1to2);

                if (csv.is_open()) {
                    mean_field.rebuild_potentials(particles, n_test);
                    const double epot = mean_field.compute_potential_energy_test_gev(n_test);
                    const double r_con = mean_field.density_contrast_rms();
                    csv << step << "," << t_now << "," << obs.n_active << ","
                        << obs.E_tot << "," << epot << "," << (obs.E_tot + epot) << "," << r_con << ","
                        << obs.pressure_anisotropy() << "," << obs.rms_radius() << ","
                        << n_nuc << "," << n_d << "," << n_dstar << ","
                        << n_a3 << "," << n_he4 << "," << n_a4_exc << ","
                        << n_delta << "," << n_pi << ","
                        << st.collisions_elastic << "," << st.collisions_inelastic_2to2 << ","
                        << st.formations_2to1 << "," << st.decays_1to2 << "," << st.max_prob_seen << "\n";
                }
            }

            if (do_frame || do_hydro) {
                const auto host_snap = particles.download();
                if (do_frame) {
                    write_binary_frame(dump_dir, frame_idx++, step, t_now, gcfg,
                                       obs, st, host_snap, dump_limit);
                }
                if (do_hydro) {
                    write_hydro_grid_binary(dump_hydro_dir, hydro_idx++, step, t_now, n_test,
                                            gcfg, mf_cfg, host_snap);
                }
            }
        }

        if (step < n_steps) {
            stepper.step(particles, grid, &mean_field);
        }
    }

    device_synchronize();
    const auto t_wall_end = std::chrono::high_resolution_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    mean_field.rebuild_potentials(particles, n_test);
    const GlobalObservables obs_final = obs_calc.compute(particles);
    const double E_pot_final = mean_field.compute_potential_energy_test_gev(n_test);
    const double H_tot_final = obs_final.E_tot + E_pot_final;
    const StepStats final_stats = stepper.read_stats();
    const double rel_dH = std::fabs(H_tot_final - H_tot0) / std::fabs(H_tot0);
    const double dP_mag = std::sqrt(
        (obs_final.Px_tot - obs0.Px_tot) * (obs_final.Px_tot - obs0.Px_tot) +
        (obs_final.Py_tot - obs0.Py_tot) * (obs_final.Py_tot - obs0.Py_tot) +
        (obs_final.Pz_tot - obs0.Pz_tot) * (obs_final.Pz_tot - obs0.Pz_tot));

    std::printf("---------------------------------------------------------------------------------------\n");
    std::printf(" Conservation   : |dH|/H = %.3e | |dP| = %.3e GeV/c | dB = %lld | dQ = %lld\n",
                rel_dH, dP_mag,
                obs_final.baryon_tot - obs0.baryon_tot,
                obs_final.charge_tot - obs0.charge_tot);
    std::printf(" Max Prob Seen  : %.4f (overflows: %llu) | Active Particles: %d -> %d\n",
                final_stats.max_prob_seen, final_stats.prob_overflows,
                obs0.n_active, obs_final.n_active);
    fo_clusterizer.write_freezeout_cells_csv();
    if (cluster_mode != ClusterMode::Microscopic) {
        std::printf(" Freeze-Out Grid: %d cells (n_B < %.3f fm^-3) | %lld B -> %lld cl | back-inward: %lld (reabs N: %lld, remelt cl: %lld)\n",
                    fo_clusterizer.total_cells_frozen(), fo_cfg.n_fr_fm3,
                    fo_clusterizer.total_baryons_converted(), fo_clusterizer.total_clusters_formed(),
                    fo_clusterizer.back_emitted_total(),
                    fo_clusterizer.reabsorbed_nucleons(), fo_clusterizer.remelted_clusters());
    }
    std::printf(" Wall Time      : %.3f s (%.2f ms/step)\n",
                elapsed_s, 1000.0 * elapsed_s / (n_steps > 0 ? n_steps : 1));
    std::printf("=======================================================================================\n");

    if (mode == "hic" || !flow_csv_path.empty()) {
        const auto final_snap = particles.download();
        compute_and_write_fopi_flow(final_snap, n_test, y_cm_beam, flow_csv_path, step_cfg.seed);
    }

    return 0;
}
