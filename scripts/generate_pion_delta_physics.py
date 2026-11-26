#!/usr/bin/env python3
"""
Generate 6-panel validation & heavy-ion physics figure for the coupled
Nucleon + Light-Cluster + Pion + Delta(1232) Vlasov-BUU transport model:
  (a) Delta(1232) p-wave Breit-Wigner spectral function A_Delta(m) & Gamma(m)
  (b) Elementary pi+p -> Delta++, pi-p -> Delta0, and NN -> N Delta / d pi cross sections
  (c) Time evolution of N(Delta) and N(pi-, pi0, pi+) in Au+Au @ 800 AMeV
  (d) FOPI Au+Au beam-energy excitation of M(pi)/<A_part> and pi-/pi+ ratio (0.25 - 1.2 AGeV)
  (e) Freeze-out rapidity spectra dN/dy0 of pi-, pi0, pi+ vs p, d in Au+Au @ 800 AMeV
  (f) Directed flow v1(y0) and elliptic flow v2(y0) of pions vs protons & deuterons
"""

import csv
import math
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_FP32 = os.path.join(ROOT, "build", "cluster_transport_fp32")
BIN_FP64 = os.path.join(ROOT, "build", "cluster_transport")
OUT_DIR = os.path.join(ROOT, "output", "pion_delta")
ASSET_PNG = os.path.join(ROOT, "assets", "pion_delta_physics.png")

HBARC = 0.1973269804
MP = 0.93827208816
MN = 0.93956542052
MD = 1.87561294257
MPI_C = 0.13957039
MPI_0 = 0.13497680
MDELTA = 1.2320
GDELTA = 0.1180


def two_body_pcm(sqrt_s, ma, mb):
    if sqrt_s <= ma + mb:
        return 0.0
    s = sqrt_s * sqrt_s
    t1 = s - (ma + mb) ** 2
    t2 = s - (ma - mb) ** 2
    return 0.5 * math.sqrt(max(0.0, t1 * t2)) / sqrt_s


def bw_pwave_sq(q):
    x2 = (q / 0.20) ** 2
    return x2 / (1.0 + x2)


def delta_width_and_spectral():
    m_min = MP + MPI_C
    m_max = MDELTA + 10.0 * GDELTA
    q0 = two_body_pcm(MDELTA, MP, MPI_C)
    b0 = bw_pwave_sq(q0)

    ms = np.linspace(m_min + 1e-4, m_max, 4000)
    gam = np.zeros_like(ms)
    araw = np.zeros_like(ms)
    for i, m in enumerate(ms):
        q = two_body_pcm(m, MP, MPI_C)
        g = GDELTA * (q / q0) * (MDELTA / m) * (bw_pwave_sq(q) / b0)
        gam[i] = g
        dm2 = m * m - MDELTA * MDELTA
        mg = m * g
        araw[i] = (2.0 / math.pi) * (m * mg) / (dm2 * dm2 + mg * mg)
    norm = np.trapezoid(araw, ms) if hasattr(np, "trapezoid") else np.trapz(araw, ms)
    return ms, gam, araw / norm, norm


def elementary_cross_sections():
    ms, gam, spec_a, norm = delta_width_and_spectral()
    q0 = two_body_pcm(MDELTA, MP, MPI_C)
    b0 = bw_pwave_sq(q0)

    # 1. pi+ p -> Delta++ and pi- p -> Delta0 vs sqrt(s)
    s_piN = np.linspace(MP + MPI_C + 1e-3, 1.60, 260)
    sig_dpp, sig_d0 = [], []
    for m in s_piN:
        q = two_body_pcm(m, MP, MPI_C)
        g = GDELTA * (q / q0) * (MDELTA / m) * (bw_pwave_sq(q) / b0)
        dm2 = m * m - MDELTA * MDELTA
        mg = m * g
        a_val = ((2.0 / math.pi) * (m * mg) / (dm2 * dm2 + mg * mg)) / norm
        red_g = (GDELTA / q0) * (MDELTA / m) * (bw_pwave_sq(q) / b0)
        # sigma = (g_R / (g_pi g_N)) * 2 pi^2 hbarc^2 * A(m) * red_g / q * 10 mb/fm^2
        sig_fm2 = (4.0 / 2.0) * (2.0 * math.pi * math.pi * HBARC * HBARC) * a_val * red_g / q
        sig_dpp.append(10.0 * sig_fm2)
        sig_d0.append(10.0 * sig_fm2 / 3.0)

    # 2. p+p -> n+Delta++, p+p -> p+Delta+, p+p -> d+pi+ vs sqrt(s)
    s_NN = np.linspace(2.0 * MP + MPI_0 + 1e-3, 2.85, 220)
    sig_ndpp, sig_pdp, sig_dpi = [], [], []
    s_th_del = (MP + MN + MPI_C) ** 2
    s_th_dpi = (MD + MPI_C) ** 2
    for sqs in s_NN:
        s = sqs * sqs
        pin = two_body_pcm(sqs, MP, MP)
        m_hi = min(MDELTA + 10.0 * GDELTA, sqs - MN)
        if m_hi > MP + MPI_C + 1e-4 and pin > 1e-9:
            grid_m = np.linspace(MP + MPI_C + 1e-4, m_hi, 160)
            a_interp = np.interp(grid_m, ms, spec_a)
            p_out = np.array([two_body_pcm(sqs, MN, mr) for mr in grid_m])
            integ = np.trapezoid(a_interp * p_out, grid_m) if hasattr(np, "trapezoid") else np.trapz(a_interp * p_out, grid_m)
            sig0_34 = 6.3 / (1.0 + (s - s_th_del) / 1.4)
            sig_ndpp.append(10.0 * 8.0 * integ * sig0_34 / (s * pin))
            sig_pdp.append(10.0 * 8.0 * integ * (sig0_34 / 3.0) / (s * pin))
        else:
            sig_ndpp.append(0.0)
            sig_pdp.append(0.0)
        p_dpi = two_body_pcm(sqs, MD, MPI_C)
        sig_dpi.append(10.0 * 3.0 * p_dpi * (1.8 / (1.0 + max(0.0, s - s_th_dpi) / 0.6)) / (s * pin) if p_dpi > 0 else 0.0)

    return s_piN, np.array(sig_dpp), np.array(sig_d0), s_NN, np.array(sig_ndpp), np.array(sig_pdp), np.array(sig_dpi)


def run_auau_simulations():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 1. Central Au+Au (b = 2.0 fm, A_part ~ 345) beam-energy scan: 250, 400, 800, 1200 AMeV
    energies = [250.0, 400.0, 800.0, 1200.0]
    ntests   = [250,   160,   120,   100]
    steps_l  = [220,   190,   160,   140]
    dts      = [0.25,  0.22,  0.20,  0.18]
    scan_results = {}

    for elab, nt, nst, dt in zip(energies, ntests, steps_l, dts):
        tag = f"auau_central_{int(elab)}"
        ts_csv = os.path.join(OUT_DIR, f"{tag}_ts.csv")
        fl_csv = os.path.join(OUT_DIR, f"{tag}_flow.csv")
        cmd = [
            BIN_FP32, "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
            "--elab-amev", str(elab), "--impact-b", "2.0",
            "--ntest", str(nt), "--steps", str(nst), "--dt", str(dt), "--substeps", "2",
            "--box", "48.0", "--grid", "24", "--eos", "md-soft", "--in-medium-ec", "on",
            "--seed", str(5000 + int(elab)), "--csv", ts_csv, "--flow-csv", fl_csv,
        ]
        print(f"Running central Au+Au @ {elab:.0f} AMeV (N_test={nt})...", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        yields = {}
        with open(fl_csv + ".yields.csv", "r") as f:
            for row in csv.DictReader(f):
                yields[row["species"]] = float(row["yield_per_event"])
        scan_results[elab] = {"ts_csv": ts_csv, "fl_csv": fl_csv, "yields": yields, "ntest": nt}

    # 2. Semicentral Au+Au @ 800 AMeV (b = 5.5 fm, N_test = 240) for high-stat pion v1(y0) & v2(y0)
    semi_ts = os.path.join(OUT_DIR, "auau800_b55_ts.csv")
    semi_fl = os.path.join(OUT_DIR, "auau800_b55_flow.csv")
    cmd_semi = [
        BIN_FP32, "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "800.0", "--impact-b", "5.5",
        "--ntest", "240", "--steps", "160", "--dt", "0.20", "--substeps", "2",
        "--box", "48.0", "--grid", "24", "--eos", "md-soft", "--in-medium-ec", "on",
        "--seed", "8800", "--csv", semi_ts, "--flow-csv", semi_fl,
    ]
    print("Running semicentral Au+Au @ 800 AMeV (b=5.5 fm, N_test=240)...", flush=True)
    subprocess.run(cmd_semi, check=True, stdout=subprocess.DEVNULL)
    return scan_results, semi_fl


def load_flow_species(fl_csv, sp_name):
    y0, dndy0, v1, v1e, v2, v2e = [], [], [], [], [], []
    with open(fl_csv, "r") as f:
        for r in csv.DictReader(f):
            if r["species"] == sp_name:
                y0.append(float(r["y0"]))
                dndy0.append(float(r["dN_dy0"]))
                v1.append(float(r["v1"]))
                v1e.append(float(r["v1_err"]))
                v2.append(float(r["v2"]))
                v2e.append(float(r["v2_err"]))
    return np.array(y0), np.array(dndy0), np.array(v1), np.array(v1e), np.array(v2), np.array(v2e)


def make_figure():
    ms, gam, spec_a, _ = delta_width_and_spectral()
    s_piN, sig_dpp, sig_d0, s_NN, sig_ndpp, sig_pdp, sig_dpi = elementary_cross_sections()
    scan, semi_fl = run_auau_simulations()

    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.fontsize": 8.0, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.6), dpi=180)
    fig.suptitle(
        "Pion ($\\pi^-, \\pi^0, \\pi^+$) & $\\Delta(1232)$ Isobar Transport Coupled to Light Clusters ($A \\leq 4$) & MD-Skyrme Mean Field",
        fontsize=13.0, fontweight="bold", y=0.98
    )

    # (a) Delta(1232) spectral function & momentum-dependent width
    ax = axes[0, 0]
    mask = ms <= 1.75
    l1 = ax.plot(ms[mask], spec_a[mask], color="#c0392b", lw=2.2, label="$A_\\Delta(m)$ [$\\mathrm{GeV}^{-1}$]")
    ax.fill_between(ms[mask], 0, spec_a[mask], color="#e74c3c", alpha=0.15)
    ax.axvline(MDELTA, color="#555", ls="--", lw=1.1, label="$m_0 = 1.232$ GeV")
    ax.axvline(MP + MPI_C, color="#2980b9", ls=":", lw=1.2, label="$m_N + m_\\pi = 1.078$ GeV")
    ax.set_xlabel("Invariant Mass $m_\\Delta$ [GeV]")
    ax.set_ylabel("Spectral Density $A_\\Delta(m)$ [$\\mathrm{GeV}^{-1}$]", color="#c0392b")
    ax.set_xlim(1.05, 1.72)
    ax.set_ylim(0, 6.2)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    l2 = ax2.plot(ms[mask], gam[mask] * 1e3, color="#1f77b4", lw=1.9, ls="-.", label="$\\Gamma_\\Delta(m)$ [MeV]")
    ax2.set_ylabel("$p$-Wave Width $\\Gamma_\\Delta(m)$ [MeV]", color="#1f77b4")
    ax2.set_ylim(0, 320)
    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", frameon=True)
    ax.set_title("(a) $\\Delta(1232)$ $p$-Wave Breit-Wigner Spectral Function")

    # (b) Elementary cross sections: pi+N -> Delta and NN -> N Delta / d pi
    ax = axes[0, 1]
    ax.plot(s_piN, sig_dpp, color="#d62728", lw=2.2, label="$\\pi^+ p \\to \\Delta^{++}$ (peak $193$ mb)")
    ax.plot(s_piN, sig_d0, color="#ff7f0e", lw=1.9, ls="--", label="$\\pi^- p \\to \\Delta^0$ ($1/3 \\times \\Delta^{++}$)")
    ax.plot(s_NN, sig_ndpp * 5.0, color="#1f77b4", lw=2.0, label="$5 \\times \\sigma(pp \\to n\\Delta^{++})$")
    ax.plot(s_NN, sig_pdp * 5.0, color="#2ca02c", lw=1.8, ls="-.", label="$5 \\times \\sigma(pp \\to p\\Delta^{+})$")
    ax.plot(s_NN, sig_dpi * 15.0, color="#8e44ad", lw=1.8, ls=":", label="$15 \\times \\sigma(pp \\to d\\pi^{+})$")
    ax.set_xlabel("Center-of-Mass Energy $\\sqrt{s}$ [GeV]")
    ax.set_ylabel("Cross Section $\\sigma(\\sqrt{s})$ [mb]")
    ax.set_xlim(1.06, 2.75)
    ax.set_ylim(0, 215)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("(b) Resonant $\\pi N \\to \\Delta$ & Inelastic $NN \\to N\\Delta, d\\pi$")

    # (c) Time evolution of Delta(1232), pions, and deuterons in central Au+Au @ 800 AMeV
    ax = axes[0, 2]
    r800 = scan[800.0]
    nt800 = r800["ntest"]
    t_arr, n_del, n_pi, n_d = [], [], [], []
    with open(r800["ts_csv"], "r") as f:
        for row in csv.DictReader(f):
            t_arr.append(float(row["t_fm"]))
            n_del.append(float(row["n_delta"]) / nt800)
            n_pi.append(float(row["n_pi"]) / nt800)
            n_d.append(float(row["n_d"]) / nt800)
    ax.plot(t_arr, n_del, "o-", ms=3.5, color="#d62728", lw=2.1, label="$\\Delta(1232)$ ($\\Delta^{-,0,+,++}$)")
    ax.plot(t_arr, n_pi, "s-", ms=3.5, color="#1f77b4", lw=2.1, label="Free Pions ($\\pi^-, \\pi^0, \\pi^+$)")
    ax.plot(t_arr, np.array(n_del) + np.array(n_pi), "--", color="#2c3e50", lw=1.7, label="Total $\\Delta + \\pi$ Number")
    ax.plot(t_arr, np.array(n_d) * 0.5, ":", color="#27ae60", lw=1.8, label="$0.5 \\times$ Deuterons ($d$)")
    ax.set_xlabel("Collision Time $t$ [fm/$c$]")
    ax.set_ylabel("Multiplicity per Event")
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", frameon=True)
    ax.set_title("(c) $\\Delta$ Clock & Pion Freeze-Out (Au+Au @ 800 $A$ MeV)")

    # (d) Beam-energy excitation function of M(pi)/A_part and pi-/pi+ vs FOPI (NPA 781, 2007)
    ax = axes[1, 0]
    elabs = sorted(scan.keys())
    a_part_est = 342.0  # b = 2.0 fm Au+Au participant number
    mpi_apart = [(scan[e]["yields"]["pi-"] + scan[e]["yields"]["pi0"] + scan[e]["yields"]["pi+"]) / a_part_est for e in elabs]
    pim_pip = [scan[e]["yields"]["pi-"] / max(scan[e]["yields"]["pi+"], 1e-6) for e in elabs]

    # FOPI central Au+Au experimental systematics (Reisdorf et al., Nucl. Phys. A 781, 459 (2007))
    fopi_e = [400.0, 800.0, 1200.0]
    fopi_mpi_ap = [0.017, 0.076, 0.155]
    fopi_mpi_err = [0.003, 0.009, 0.016]
    fopi_ratio = [2.95, 2.16, 1.93]
    fopi_r_err = [0.22, 0.13, 0.11]

    l_m1 = ax.plot(elabs, mpi_apart, "o-", color="#1f77b4", lw=2.2, ms=6, label="BUU $\\langle M_\\pi \\rangle / A_{\\mathrm{part}}$")
    l_e1 = ax.errorbar(fopi_e, fopi_mpi_ap, yerr=fopi_mpi_err, fmt="D", color="#0b3c5d", ms=5.5, capsize=3, label="FOPI $\\langle M_\\pi \\rangle / A_{\\mathrm{part}}$")
    ax.set_xlabel("Beam Kinetic Energy $E_{\\mathrm{lab}}$ [$A$ MeV]")
    ax.set_ylabel("$\\langle M_\\pi \\rangle / A_{\\mathrm{part}}$", color="#1f77b4")
    ax.set_xlim(180, 1300)
    ax.set_ylim(0, 0.22)
    ax.grid(alpha=0.25)

    axr = ax.twinx()
    l_m2 = axr.plot(elabs, pim_pip, "s--", color="#c0392b", lw=2.0, ms=6, label="BUU $\\pi^- / \\pi^+$ Ratio")
    l_e2 = axr.errorbar(fopi_e, fopi_ratio, yerr=fopi_r_err, fmt="^", color="#7f1d1d", ms=6, capsize=3, label="FOPI $\\pi^- / \\pi^+$ Ratio")
    axr.axhline((118.0 / 79.0) ** 2, color="#c0392b", alpha=0.35, ls=":", lw=1.2)
    axr.set_ylabel("Charged Pion Ratio $\\pi^- / \\pi^+$", color="#c0392b")
    axr.set_ylim(1.2, 3.6)
    ax.legend([l_m1[0], l_e1, l_m2[0], l_e2],
              ["BUU $\\langle M_\\pi \\rangle / A_{\\mathrm{part}}$", "FOPI $\\langle M_\\pi \\rangle / A_{\\mathrm{part}}$",
               "BUU $\\pi^- / \\pi^+$", "FOPI $\\pi^- / \\pi^+$"], loc="upper left", fontsize=7.6)
    ax.set_title("(d) Au+Au Pion Yield & $\\pi^-/\\pi^+$ vs. FOPI ($b=2$ fm)")

    # (e) Rapidity spectra dN/dy0 in semicentral Au+Au @ 800 AMeV (b = 5.5 fm)
    ax = axes[1, 1]
    for sp, col, mk, scale, lbl in [
        ("pi-", "#d62728", "o", 1.0, "$\\pi^-$"),
        ("pi0", "#2ca02c", "s", 1.0, "$\\pi^0$"),
        ("pi+", "#1f77b4", "^", 1.0, "$\\pi^+$"),
        ("p",   "#34495e", "d", 0.10, "$0.10 \\times p$"),
        ("d",   "#8e44ad", "v", 0.25, "$0.25 \\times d$"),
    ]:
        y0, dndy0, _, _, _, _ = load_flow_species(semi_fl, sp)
        ax.plot(y0, dndy0 * scale, marker=mk, ms=4.2, lw=1.9, color=col, label=lbl)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ (per Event)")
    ax.set_xlim(-1.45, 1.45)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("(e) Pion, Proton & Deuteron $dN/dy_0$ (Au+Au 800 $A$ MeV, $b=5.5$ fm)")

    # (f) Directed flow v1(y0) and elliptic squeeze-out v2(y0) of pions vs baryons
    ax = axes[1, 2]
    y0_p, _, v1_p, v1e_p, v2_p, _ = load_flow_species(semi_fl, "p")
    y0_d, _, v1_d, v1e_d, v2_d, _ = load_flow_species(semi_fl, "d")
    y0_pim, dn_pim, v1_pim, v1e_pim, v2_pim, v2e_pim = load_flow_species(semi_fl, "pi-")
    y0_pi0, dn_pi0, v1_pi0, v1e_pi0, v2_pi0, v2e_pi0 = load_flow_species(semi_fl, "pi0")
    y0_pip, dn_pip, v1_pip, v1e_pip, v2_pip, v2e_pip = load_flow_species(semi_fl, "pi+")

    # Combine all pions weighted by bin counts for clean v1(y0) & v2(y0)
    w_sum = np.maximum(dn_pim + dn_pi0 + dn_pip, 1e-9)
    v1_pi = (dn_pim * v1_pim + dn_pi0 * v1_pi0 + dn_pip * v1_pip) / w_sum
    v2_pi = (dn_pim * v2_pim + dn_pi0 * v2_pi0 + dn_pip * v2_pip) / w_sum
    v1e_pi = np.sqrt((dn_pim * v1e_pim) ** 2 + (dn_pi0 * v1e_pi0) ** 2 + (dn_pip * v1e_pip) ** 2) / w_sum

    sel = np.abs(y0_p) <= 1.25
    ax.plot(y0_p[sel], v1_p[sel], "o-", ms=4.0, color="#34495e", lw=1.9, label="Proton $v_1(y_0)$")
    ax.plot(y0_d[sel], v1_d[sel], "s-", ms=4.0, color="#8e44ad", lw=1.9, label="Deuteron $v_1(y_0)$")
    ax.errorbar(y0_pim[sel], v1_pi[sel], yerr=v1e_pi[sel], fmt="^-", ms=4.5, capsize=2.5,
                color="#d62728", lw=2.0, label="All Pions $v_1(y_0)$")
    ax.plot(y0_pim[sel], v2_pi[sel], "D--", ms=3.8, color="#e67e22", lw=1.6, label="All Pions $v_2(y_0)$ (squeeze-out)")
    ax.axhline(0.0, color="#666", lw=0.9, ls=":")
    ax.axvline(0.0, color="#666", lw=0.9, ls=":")
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("Anisotropic Flow $v_1(y_0),\\; v_2(y_0)$")
    ax.set_xlim(-1.30, 1.30)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("(f) Pion vs. Baryon Flow $v_1(y_0)$ & $v_2(y_0)$ (Au+Au 800 $A$ MeV)")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(ASSET_PNG), exist_ok=True)
    fig.savefig(ASSET_PNG, dpi=180)
    print(f"\nSaved figure -> {ASSET_PNG}")
    for e in elabs:
        y = scan[e]["yields"]
        tot = y["pi-"] + y["pi0"] + y["pi+"]
        print(f"  E_lab = {e:6.0f} AMeV : pi- = {y['pi-']:6.2f}, pi0 = {y['pi0']:6.2f}, pi+ = {y['pi+']:6.2f} "
              f"| M_pi = {tot:6.2f} | pi-/pi+ = {y['pi-']/max(y['pi+'],1e-6):.2f} | d = {y['d']:5.2f}")


if __name__ == "__main__":
    make_figure()
