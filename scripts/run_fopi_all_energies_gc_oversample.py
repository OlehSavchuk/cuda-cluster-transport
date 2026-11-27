#!/usr/bin/env python3
"""
Full FOPI Au+Au Excitation-Function Benchmark (90 - 1500 A MeV, all 10 FOPI energies)
using FP32 CUDA Relativistic Hadron & Cluster Transport:
  - Grand-Canonical Poisson HCGF particlization (--sampling gc) at n_fr = rho_0 / 8
  - Freeze-out oversampling (--fo-oversample 10) + systematic backflow thinning/remelting (--backflow remelt)
  - 6D Pauli blocking (--pauli on), 3D spectral Coulomb (--coulomb on), Pratt-Torrieri anisotropy (--anisotropy on)
  - N_test = 1000 per baryon, 10 BUU events per energy (5 stratified central b0 < 0.15 + 5 semi-central 0.25 < b0 < 0.45)

Generates:
  - 10 per-energy 6-panel FOPI comparison figures: assets/fopi_gc_oversample_auau{E:04d}.png
  - 1 global excitation-function summary figure:   assets/fopi_gc_oversample_all_energies_summary.png
"""

import argparse
import csv
import json
import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIN_FP32 = ROOT / "build" / "cluster_transport_fp32"
FOPI_ALL_JSON = ROOT / "scripts" / "fopi_exp_auau_all.json"
OUT_ROOT = ROOT / "output" / "fopi_gc_oversample_ntest1000"
ASSETS_DIR = ROOT / "assets"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160
FOPI_ENERGIES = [90, 120, 150, 250, 400, 600, 800, 1000, 1200, 1500]

# Physical (steps, dt_fm, substeps) scaling across 0.09 - 1.50 A GeV so t_end >= 2.85 * t_overlap
ENERGY_TIME_GRID = {
    90:   (300, 0.40, 2),   # t_overlap = 42.3 fm/c -> t_end = 120.0 fm/c
    120:  (290, 0.37, 2),   # t_overlap = 36.9 fm/c -> t_end = 107.3 fm/c
    150:  (285, 0.34, 2),   # t_overlap = 33.2 fm/c -> t_end =  96.9 fm/c
    250:  (280, 0.30, 2),   # t_overlap = 26.4 fm/c -> t_end =  84.0 fm/c
    400:  (250, 0.26, 2),   # t_overlap = 21.6 fm/c -> t_end =  65.0 fm/c
    600:  (235, 0.24, 2),   # t_overlap = 18.4 fm/c -> t_end =  56.4 fm/c
    800:  (225, 0.22, 2),   # t_overlap = 16.6 fm/c -> t_end =  49.5 fm/c
    1000: (220, 0.21, 2),   # t_overlap = 15.3 fm/c -> t_end =  46.2 fm/c
    1200: (220, 0.20, 2),   # t_overlap = 14.4 fm/c -> t_end =  44.0 fm/c
    1500: (220, 0.19, 2),   # t_overlap = 13.5 fm/c -> t_end =  41.8 fm/c
}


def eos_label(eos_flag: str) -> str:
    if eos_flag in ("sm", "md-soft", "SM"):
        return "SM ($K_0 = 215\\;\\mathrm{MeV}$)"
    if eos_flag in ("hm", "md-stiff", "HM"):
        return "HM ($K_0 = 380\\;\\mathrm{MeV}$)"
    return eos_flag


def run_single_job(job):
    tag, elab, b_fm, ntest, m_over, eos_flag, save_cells, seed, runs_dir = job
    runs_dir.mkdir(parents=True, exist_ok=True)
    fl_csv = runs_dir / f"{tag}.csv"
    cells_csv = runs_dir / f"{tag}.cells.csv"
    t_file = runs_dir / f"{tag}.time"
    yld_csv = Path(str(fl_csv) + ".yields.csv")
    if yld_csv.exists() and (not save_cells or cells_csv.exists()):
        dt_prev = float(t_file.read_text().strip()) if t_file.exists() else 0.0
        return tag, elab, dt_prev, "cached", fl_csv

    steps, dt_fm, substeps = ENERGY_TIME_GRID[elab]
    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", f"{float(elab):.1f}", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "68.0", "--grid", "68",
        "--steps", str(steps), "--dt", f"{dt_fm:.2f}", "--substeps", str(substeps),
        "--eos", eos_flag, "--in-medium-ec", "on", "--pauli", "on", "--coulomb", "on",
        "--cluster-mode", "freezeout", "--backflow", "remelt",
        "--sampling", "gc", "--fo-oversample", str(m_over), "--anisotropy", "on",
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "2",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    if save_cells:
        cmd += ["--freezeout-cells-csv", str(cells_csv)]

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    t_file.write_text(f"{dt_s:.2f}\n")
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, elab, dt_s, (fo_ln[-1] if fo_ln else "done"), fl_csv


def pool_mean_err(vals_errs_cnts):
    n_tot = sum(n for _, _, n in vals_errs_cnts)
    if n_tot <= 0.0:
        return 0.0, 0.0, 0.0
    mean = sum(n * v for v, _, n in vals_errs_cnts) / n_tot
    if n_tot <= 1.0:
        return mean, 0.0, n_tot
    sum_sq = sum(n * (n * e * e + v * v) for v, e, n in vals_errs_cnts)
    var_mean = max(0.0, sum_sq / n_tot - mean * mean) / n_tot
    return mean, math.sqrt(var_mean), n_tot


def merge_ensemble(csv_paths, dst_csv: Path):
    rap_tables = [list(csv.DictReader(open(p))) for p in csv_paths]
    ut0_tables = [list(csv.DictReader(open(str(p) + ".ut0.csv"))) for p in csv_paths]
    yld_tables = [list(csv.DictReader(open(str(p) + ".yields.csv"))) for p in csv_paths]
    m_runs = len(csv_paths)

    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "bin", "y", "y0", "dN_dy", "dN_dy0", "dN_dy0_err",
                    "v1", "v1_err", "v2", "v2_err", "n_stat",
                    "v1_ut08", "v1_ut08_err", "v2_ut08", "v2_ut08_err", "n_ut08",
                    "v1_ut04", "v1_ut04_err", "v2_ut04", "v2_ut04_err", "n_ut04"])
        for row_idx in range(len(rap_tables[0])):
            r0 = rap_tables[0][row_idx]
            dndy0s = [float(t[row_idx]["dN_dy0"]) for t in rap_tables]
            dndy_m = sum(float(t[row_idx]["dN_dy"]) for t in rap_tables) / m_runs
            dndy0_m = sum(dndy0s) / m_runs
            dndy0_e = math.sqrt(sum((x - dndy0_m) ** 2 for x in dndy0s) / (m_runs * max(1, m_runs - 1))) if m_runs > 1 else 0.0
            v1_m, v1_e, n_st = pool_mean_err(
                [(float(t[row_idx]["v1"]), float(t[row_idx]["v1_err"]), float(t[row_idx]["n_stat"])) for t in rap_tables])
            v2_m, v2_e, _ = pool_mean_err(
                [(float(t[row_idx]["v2"]), float(t[row_idx]["v2_err"]), float(t[row_idx]["n_stat"])) for t in rap_tables])
            u8_v1_m, u8_v1_e, n_u8 = pool_mean_err(
                [(float(t[row_idx]["v1_ut08"]), float(t[row_idx]["v1_ut08_err"]), float(t[row_idx]["n_ut08"])) for t in rap_tables])
            u8_v2_m, u8_v2_e, _ = pool_mean_err(
                [(float(t[row_idx]["v2_ut08"]), float(t[row_idx]["v2_ut08_err"]), float(t[row_idx]["n_ut08"])) for t in rap_tables])
            u4_v1_m, u4_v1_e, n_u4 = pool_mean_err(
                [(float(t[row_idx]["v1_ut04"]), float(t[row_idx]["v1_ut04_err"]), float(t[row_idx]["n_ut04"])) for t in rap_tables])
            u4_v2_m, u4_v2_e, _ = pool_mean_err(
                [(float(t[row_idx]["v2_ut04"]), float(t[row_idx]["v2_ut04_err"]), float(t[row_idx]["n_ut04"])) for t in rap_tables])
            w.writerow([r0["species"], r0["bin"], r0["y"], r0["y0"],
                        f"{dndy_m:.6f}", f"{dndy0_m:.6f}", f"{dndy0_e:.6f}",
                        f"{v1_m:.6f}", f"{v1_e:.6f}", f"{v2_m:.6f}", f"{v2_e:.6f}", f"{n_st:.0f}",
                        f"{u8_v1_m:.6f}", f"{u8_v1_e:.6f}", f"{u8_v2_m:.6f}", f"{u8_v2_e:.6f}", f"{n_u8:.0f}",
                        f"{u4_v1_m:.6f}", f"{u4_v1_e:.6f}", f"{u4_v2_m:.6f}", f"{u4_v2_e:.6f}", f"{n_u4:.0f}"])

    with open(str(dst_csv) + ".ut0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "bin", "ut0", "v2", "v2_err", "n_stat"])
        for row_idx in range(len(ut0_tables[0])):
            r0 = ut0_tables[0][row_idx]
            mv, ev, nu = pool_mean_err(
                [(float(t[row_idx]["v2"]), float(t[row_idx]["v2_err"]), float(t[row_idx]["n_stat"])) for t in ut0_tables])
            w.writerow([r0["species"], r0["bin"], r0["ut0"], f"{mv:.6f}", f"{ev:.6f}", f"{nu:.0f}"])

    with open(str(dst_csv) + ".yields.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "yield_per_event", "yield_err"])
        for s_idx in range(len(yld_tables[0])):
            sp = yld_tables[0][s_idx]["species"]
            ys = [float(t[s_idx]["yield_per_event"]) for t in yld_tables]
            ym = sum(ys) / m_runs
            ye = math.sqrt(sum((y - ym) ** 2 for y in ys) / (m_runs * max(1, m_runs - 1))) if m_runs > 1 else 0.0
            w.writerow([sp, f"{ym:.6f}", f"{ye:.6f}"])


def load_yields(p: Path):
    return {r["species"]: (float(r["yield_per_event"]), float(r.get("yield_err", 0.0)))
            for r in csv.DictReader(open(str(p) + ".yields.csv"))}


def load_rap(p: Path, sp: str, ut_cut: float):
    rs = [r for r in csv.DictReader(open(p)) if r["species"] == sp]
    col_v1 = "v1_ut08" if ut_cut > 0.6 else "v1_ut04"
    col_e1 = f"{col_v1}_err"
    col_n1 = "n_ut08" if ut_cut > 0.6 else "n_ut04"
    return (np.array([float(r["y0"]) for r in rs]),
            np.array([float(r["dN_dy0"]) for r in rs]),
            np.array([float(r.get("dN_dy0_err", 0.0)) for r in rs]),
            np.array([float(r[col_v1]) for r in rs]),
            np.array([float(r[col_e1]) for r in rs]),
            np.array([float(r[col_n1]) for r in rs]))


def load_ut0(p: Path, sp: str):
    rs = [r for r in csv.DictReader(open(str(p) + ".ut0.csv")) if r["species"] == sp]
    return (np.array([float(r["ut0"]) for r in rs]),
            np.array([float(r["v2"]) for r in rs]),
            np.array([float(r["v2_err"]) for r in rs]),
            np.array([float(r["n_stat"]) for r in rs]))


def load_T_hist(paths):
    ts, ws = [], []
    for p in paths:
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            ts.append(float(r["T_mev"]))
            ws.append(float(r["baryons_converted"]))
    return np.array(ts), np.array(ws)


def weighted_quantiles(values, weights, qs):
    if len(values) == 0 or np.sum(weights) <= 0:
        return [0.0 for _ in qs]
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cdf = np.cumsum(w) / np.sum(w)
    return [float(np.interp(q, cdf, v)) for q in qs]


def fit_midrapidity_slope(y0, v1, mask):
    ym, vm = y0[mask], v1[mask]
    sel = np.abs(ym) <= 0.45
    if np.count_nonzero(sel) < 3:
        return float("nan")
    c = np.polyfit(ym[sel], vm[sel], 1)
    return float(c[0])


def interp_v2_at_ut0(ut0, v2, mask, target=1.0):
    um, vm = ut0[mask], v2[mask]
    if len(um) < 2 or target < um[0] or target > um[-1]:
        return float("nan")
    return float(np.interp(target, um, vm))


def plot_energy_comparison(elab: int, c1_csv: Path, c2_csv: Path, cell_paths, fopi_e: dict,
                           eos_flag: str, ntest: int, m_over: int, n_c1: int, n_c2: int, out_png: Path):
    plt.rcParams.update({
        "font.size": 9.4, "axes.titlesize": 10.2, "axes.labelsize": 9.4,
        "legend.fontsize": 7.1, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.6, 10.4), dpi=185)
    ut_cut = float(fopi_e["v1_ut0_cut"])
    e_str = f"{elab}\\;A\\;\\mathrm{{MeV}}" if elab < 1000 else f"{elab / 1000.0:.2f}\\;A\\;\\mathrm{{GeV}}"
    fig.suptitle(
        f"FOPI Au+Au @ ${e_str}$: Grand-Canonical HCGF + ${m_over}\\times$ Oversampling "
        f"(FP32, $N_{{\\mathrm{{test}}}}={ntest}$, {n_c1}+{n_c2} Events, {eos_label(eos_flag)})",
        fontsize=11.2, fontweight="bold", y=0.98
    )

    # (a) Baryon-weighted freeze-out temperature distribution P(T_fr) at n_fr = rho0 / 8
    ax = axes[0, 0]
    T_c1, w_c1 = load_T_hist(cell_paths["c1"])
    T_c2, w_c2 = load_T_hist(cell_paths["c2"])
    t_hi = max(24.0, min(78.0, (np.percentile(T_c1, 99.5) * 1.18) if len(T_c1) > 0 else 40.0))
    bins = np.linspace(1.0, t_hi, 42)
    t_mean_c1 = 0.0
    if len(T_c1) > 0:
        t_mean_c1 = float(np.average(T_c1, weights=w_c1))
        q16, q84 = weighted_quantiles(T_c1, w_c1, [0.16, 0.84])
        ax.hist(T_c1, bins=bins, weights=w_c1, density=True, color="#c0392b", alpha=0.62,
                label=f"Central ($b_0<0.15$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {t_mean_c1:.1f}$ MeV\n"
                      f"16–84% range: $[{q16:.1f},\\,{q84:.1f}]$ MeV")
        ax.axvline(t_mean_c1, color="#922b21", ls="--", lw=2.0)
    if len(T_c2) > 0:
        t_mean_c2 = float(np.average(T_c2, weights=w_c2))
        ax.hist(T_c2, bins=bins, weights=w_c2, density=True, histtype="step", color="#1f618d", lw=2.1,
                label=f"Semi-Central ($0.25<b_0<0.45$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {t_mean_c2:.1f}$ MeV")
        ax.axvline(t_mean_c2, color="#1f618d", ls="-.", lw=1.8)
    ax.set_xlabel("Cell Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.set_ylabel("Baryon-Weighted Probability Density")
    ax.set_title("(a) Freeze-Out Temperature $P(T_{\\mathrm{fr}})$ at $n_{\\mathrm{fr}} = \\rho_0 / 8$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.0)

    # (b) Central (b0 < 0.15) log-scale multiplicities across A = 1..6 (+ pions) vs FOPI
    ax = axes[0, 1]
    y_c1 = load_yields(c1_csv)
    show_pions = (elab >= 250) or (y_c1.get("pi-", (0.0, 0.0))[0] > 1.0e-3)
    if show_pions:
        sps = ["p", "d", "t", "3He", "4He", "6He", "6Li", "pi-", "pi0", "pi+"]
        lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$",
                "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$", "$\\pi^-$", "$\\pi^0$", "$\\pi^+$"]
    else:
        sps = ["p", "d", "t", "3He", "4He", "6He", "6Li"]
        lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$",
                "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$"]
    xp = np.arange(len(sps))
    vals = [max(1e-4, y_c1.get(s, (0.0, 0.0))[0]) for s in sps]
    errs = [y_c1.get(s, (0.0, 0.0))[1] for s in sps]
    ax.bar(xp, vals, yerr=errs, width=0.52, color="#1f618d", alpha=0.88, capsize=2.5,
           label=f"GC + ${m_over}\\times$ Oversample ({eos_label(eos_flag)})")
    exp_ylds = fopi_e["central_yields_b0_lt_015"]
    exp_idx = [i for i, s in enumerate(sps) if s in exp_ylds]
    if exp_idx:
        ax.errorbar(xp[exp_idx], [exp_ylds[sps[i]]["val"] for i in exp_idx],
                    yerr=[exp_ylds[sps[i]]["err"] for i in exp_idx],
                    fmt="ko", ms=6.0, capsize=3.2, lw=1.6, zorder=8,
                    label=f"FOPI Exp. ($b_0 < 0.15$)")
    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 240.0)
    ax.set_xticks(xp)
    ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (Log Scale)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(b) Central Yields ($A = 1 \\dots 6$ & Pions, Log Scale)")

    # (c) Linear-scale central yield comparison on measured FOPI species at this energy
    ax = axes[0, 2]
    f_sps = [s for s in ["p", "d", "t", "3He", "4He", "pi-", "pi+"] if s in exp_ylds]
    lbl_map = {"p": "$p$", "d": "$d$", "t": "$t$", "3He": "$^3\\mathrm{He}$",
               "4He": "$^4\\mathrm{He}$", "pi-": "$\\pi^-$", "pi+": "$\\pi^+$"}
    f_lbls = [lbl_map[s] for s in f_sps]
    xpf = np.arange(len(f_sps))
    wf = 0.35
    ax.bar(xpf - 0.5 * wf, [exp_ylds[s]["val"] for s in f_sps],
           yerr=[exp_ylds[s]["err"] for s in f_sps], width=wf, color="#2c3e50", capsize=2.8,
           label="FOPI Exp. ($b_0 < 0.15$)")
    ax.bar(xpf + 0.5 * wf, [y_c1[s][0] for s in f_sps],
           yerr=[y_c1[s][1] for s in f_sps], width=wf, color="#c0392b", capsize=2.8,
           label=f"GC + ${m_over}\\times$ Oversample")
    ax.set_xticks(xpf)
    ax.set_xticklabels(f_lbls)
    ax.set_ylabel("Central Multiplicity / Event (Linear Scale)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.1)
    ax.set_title("(c) Linear Central Multiplicity Comparison vs. FOPI")

    # (d) Semi-central (0.25 < b0 < 0.45) Rapidity Spectra dN/dy0
    ax = axes[1, 0]
    spec_curves = [
        ("p",   "#2c3e50", 0.40, "$p \\times 0.4$"),
        ("d",   "#1f77b4", 1.00, "$d$"),
        ("t",   "#27ae60", 1.00, "$t$"),
        ("4He", "#8e44ad", 1.50, "$^4\\mathrm{He} \\times 1.5$"),
        ("6Li", "#c0392b", 25.0, "$^6\\mathrm{Li} \\times 25$"),
    ]
    if elab >= 400:
        spec_curves += [
            ("pi-", "#e67e22", 1.00, "$\\pi^-$"),
            ("pi+", "#d35400", 1.00, "$\\pi^+$"),
        ]
    for sp, col, mult, lbl in spec_curves:
        ys, ds, es, _, _, _ = load_rap(c2_csv, sp, ut_cut)
        ls = "--" if sp.startswith("pi") else "-"
        ax.plot(ys, ds * mult, ls, color=col, lw=2.1, label=lbl)
        ax.fill_between(ys, (ds - es) * mult, (ds + es) * mult, color=col, alpha=0.14)
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ ($0.25 < b_0 < 0.45$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=6.6)
    ax.set_title("(d) Semi-Central Rapidity Spectra $dN/dy_0$")

    # (e) Directed Flow v1(y0) (u_t0 > 0.8 for <=250 AMeV, u_t0 > 0.4 for >=400 AMeV) vs FOPI
    ax = axes[1, 1]
    v1_exp = fopi_e["v1_y0"]
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("A3",  "#d62728", "^", "$A=3$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        ys, _, _, v1s, e1s, ns = load_rap(c2_csv, sp, ut_cut)
        sel = (np.abs(ys) <= 1.08) & (ns >= 15)
        ax.plot(ys[sel], v1s[sel], "-", color=col, lw=2.2, label=f"{lbl} Model")
        ax.fill_between(ys[sel], v1s[sel] - e1s[sel], v1s[sel] + e1s[sel], color=col, alpha=0.14)
        if sp in v1_exp:
            ep = v1_exp[sp]
            stride = 2 if len(ep) > 14 else 1
            ax.errorbar([p["y0"] for p in ep[::stride]], [p["v1"] for p in ep[::stride]],
                        yerr=[p["err"] for p in ep[::stride]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.35, capsize=2, label=f"{lbl} FOPI")
    # Also overlay rare 6Li directed flow resolved via freeze-out oversampling
    y6, _, _, v6, e6, n6 = load_rap(c2_csv, "6Li", ut_cut)
    s6 = (np.abs(y6) <= 1.02) & (n6 >= 25) & (e6 <= 0.09)
    if np.count_nonzero(s6) >= 4:
        ax.plot(y6[s6], v6[s6], "-.", color="#e67e22", lw=1.8, label="$^6\\mathrm{Li}$ Model")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.axvline(0, color="#777", ls=":", lw=0.9)
    vmax = 0.58 if elab <= 150 else (0.76 if elab <= 400 else 0.88)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-vmax, vmax)
    ax.set_xlabel(f"Normalized CM Rapidity $y_0$ ($u_{{t0}} > {ut_cut:.1f}$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=6.2)
    ax.set_title(f"(e) Directed Flow $v_1(y_0)$ ($u_{{t0}} > {ut_cut:.1f}$) vs. FOPI")

    # (f) Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4) vs FOPI
    ax = axes[1, 2]
    v2_exp = fopi_e["v2_ut0_y0_lt_04"]
    u_min = 0.35 if elab <= 250 else 0.25
    u_max = 1.85 if elab <= 250 else 1.98
    min_plotted_v2 = -0.15
    max_plotted_v2 = 0.04
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("t",   "#d62728", "^", "$t$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        us, v2s, e2s, nsu = load_ut0(c2_csv, sp)
        sel = (us >= u_min) & (us <= u_max) & (nsu >= 100) & (e2s <= 0.042)
        if np.count_nonzero(sel) >= 2:
            ax.plot(us[sel], v2s[sel], "-", color=col, lw=2.2, label=f"{lbl} Model")
            ax.fill_between(us[sel], v2s[sel] - e2s[sel], v2s[sel] + e2s[sel], color=col, alpha=0.14)
            min_plotted_v2 = min(min_plotted_v2, float(np.min(v2s[sel] - e2s[sel])))
            max_plotted_v2 = max(max_plotted_v2, float(np.max(v2s[sel] + e2s[sel])))
        if sp in v2_exp:
            eu = v2_exp[sp]
            stride = 2 if len(eu) > 14 else 1
            ax.errorbar([p["ut0"] for p in eu[::stride]], [p["v2"] for p in eu[::stride]],
                        yerr=[p["err"] for p in eu[::stride]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.35, capsize=2, label=f"{lbl} FOPI")
            min_plotted_v2 = min(min_plotted_v2, min(p["v2"] - p["err"] for p in eu))
            max_plotted_v2 = max(max_plotted_v2, max(p["v2"] + p["err"] for p in eu))
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(u_min - 0.04, u_max + 0.04)
    y2_lo = max(-0.38, min_plotted_v2 - 0.018)
    y2_hi = min(0.14, max_plotted_v2 + 0.015)
    ax.set_ylim(y2_lo, y2_hi)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Flow / Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncol=2, fontsize=6.4)
    ax.set_title("(f) Elliptic Flow $v_2(u_{t0})$ ($|y_0| < 0.4$) vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=185)
    plt.close(fig)
    return t_mean_c1


def plot_excitation_summary(energies, merged_map, cell_map, fopi_data, eos_flag, ntest, m_over, out_png: Path):
    plt.rcParams.update({
        "font.size": 9.4, "axes.titlesize": 10.3, "axes.labelsize": 9.4,
        "legend.fontsize": 7.2, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.8, 10.5), dpi=185)
    fig.suptitle(
        f"FOPI Au+Au Excitation Functions ($0.09 - 1.50\\;A\\;\\mathrm{{GeV}}$): "
        f"Grand-Canonical HCGF + ${m_over}\\times$ Oversampling (FP32, $N_{{\\mathrm{{test}}}}={ntest}$, {eos_label(eos_flag)})",
        fontsize=11.3, fontweight="bold", y=0.98
    )
    e_agev = np.array([e / 1000.0 for e in energies])
    ylds = {e: load_yields(merged_map[e]["c1"]) for e in energies}

    # (a) Central baryon & light-cluster yields vs E_lab
    ax = axes[0, 0]
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("t",   "#27ae60", "^", "$t$"),
                             ("3He", "#e67e22", "v", "$^3\\mathrm{He}$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        ym = np.array([ylds[e][sp][0] for e in energies])
        ye = np.array([ylds[e][sp][1] for e in energies])
        ax.plot(e_agev, ym, "-", color=col, lw=2.2, label=f"{lbl} Model")
        ax.fill_between(e_agev, ym - ye, ym + ye, color=col, alpha=0.15)
        ex_x, ex_y, ex_e = [], [], []
        for e in energies:
            cy = fopi_data[str(e)]["central_yields_b0_lt_015"]
            if sp in cy:
                ex_x.append(e / 1000.0)
                ex_y.append(cy[sp]["val"])
                ex_e.append(cy[sp]["err"])
        if ex_x:
            ax.errorbar(ex_x, ex_y, yerr=ex_e, fmt=mk, color=col, ms=5.4,
                        mfc="none", mew=1.4, capsize=2.5, label=f"{lbl} FOPI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_ylim(0.5, 180.0)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Central Multiplicity / Event ($b_0 < 0.15$)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower left", ncol=2, fontsize=6.6)
    ax.set_title("(a) Light-Cluster Excitation Functions ($A = 1 \\dots 4$)")

    # (b) Rare A=6 isotopes & charged pions vs E_lab
    ax = axes[0, 1]
    for sp, col, mk, lbl in [("pi-", "#c0392b", "o", "$\\pi^-$"),
                             ("pi+", "#1f618d", "s", "$\\pi^+$"),
                             ("6He", "#16a085", "^", "$^6\\mathrm{He}$"),
                             ("6Li", "#8e44ad", "D", "$^6\\mathrm{Li}$")]:
        ym = np.array([max(1e-4, ylds[e][sp][0]) for e in energies])
        ye = np.array([ylds[e][sp][1] for e in energies])
        sel = ym > 2.0e-4
        if np.count_nonzero(sel) >= 2:
            ax.plot(e_agev[sel], ym[sel], "-", color=col, lw=2.2, label=f"{lbl} Model")
            ax.fill_between(e_agev[sel], np.maximum(1e-4, ym[sel] - ye[sel]), ym[sel] + ye[sel], color=col, alpha=0.15)
        ex_x, ex_y, ex_e = [], [], []
        for e in energies:
            cy = fopi_data[str(e)]["central_yields_b0_lt_015"]
            if sp in cy:
                ex_x.append(e / 1000.0)
                ex_y.append(cy[sp]["val"])
                ex_e.append(cy[sp]["err"])
        if ex_x:
            ax.errorbar(ex_x, ex_y, yerr=ex_e, fmt=mk, color=col, ms=5.6,
                        mfc="none", mew=1.4, capsize=2.5, label=f"{lbl} FOPI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_ylim(2e-3, 45.0)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Central Multiplicity / Event ($b_0 < 0.15$)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", ncol=2, fontsize=6.9)
    ax.set_title("(b) Pion ($\\pi^\\pm$) & Heavy $A=6$ ($^6\\mathrm{He}, ^6\\mathrm{Li}$) Yields")

    # (c) Isobaric t / 3He and Pion pi- / pi+ Yield Ratios vs FOPI
    ax = axes[0, 2]
    r_t3 = [ylds[e]["t"][0] / max(1e-6, ylds[e]["3He"][0]) for e in energies]
    ax.plot(e_agev, r_t3, "s-", color="#27ae60", lw=2.2, ms=5.0, label="$t / {}^3\\mathrm{He}$ Model")
    ex_e_t3, ex_r_t3, ex_er_t3 = [], [], []
    for e in energies:
        cy = fopi_data[str(e)]["central_yields_b0_lt_015"]
        if "t" in cy and "3He" in cy:
            v_t, e_t = cy["t"]["val"], cy["t"]["err"]
            v_h, e_h = cy["3He"]["val"], cy["3He"]["err"]
            r = v_t / v_h
            ex_e_t3.append(e / 1000.0)
            ex_r_t3.append(r)
            ex_er_t3.append(r * math.hypot(e_t / v_t, e_h / v_h))
    if ex_e_t3:
        ax.errorbar(ex_e_t3, ex_r_t3, yerr=ex_er_t3, fmt="s", color="#27ae60", ms=6.0,
                    mfc="none", mew=1.5, capsize=2.5, label="$t / {}^3\\mathrm{He}$ FOPI")

    e_pi = [e for e in energies if e >= 400 and ylds[e]["pi+"][0] > 0.05]
    if e_pi:
        ax.plot([e / 1000.0 for e in e_pi],
                [ylds[e]["pi-"][0] / ylds[e]["pi+"][0] for e in e_pi],
                "o-", color="#c0392b", lw=2.2, ms=5.0, label="$\\pi^- / \\pi^+$ Model")
    ex_e_pi, ex_r_pi, ex_er_pi = [], [], []
    for e in energies:
        cy = fopi_data[str(e)]["central_yields_b0_lt_015"]
        if "pi-" in cy and "pi+" in cy:
            vm, em = cy["pi-"]["val"], cy["pi-"]["err"]
            vp, ep = cy["pi+"]["val"], cy["pi+"]["err"]
            r = vm / vp
            ex_e_pi.append(e / 1000.0)
            ex_r_pi.append(r)
            ex_er_pi.append(r * math.hypot(em / vm, ep / vp))
    if ex_e_pi:
        ax.errorbar(ex_e_pi, ex_r_pi, yerr=ex_er_pi, fmt="o", color="#c0392b", ms=6.0,
                    mfc="none", mew=1.5, capsize=2.5, label="$\\pi^- / \\pi^+$ FOPI")
    ax.axhline(118.0 / 79.0, color="#7f8c8d", ls=":", lw=1.3, label="$N/Z = 1.49$ (Au)")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_ylim(0.8, 3.6)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Central Isobaric Yield Ratio")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(c) Isobaric Ratios $t / {}^3\\mathrm{He}$ & $\\pi^- / \\pi^+$ vs. FOPI")

    # (d) Freeze-out temperature <T_fr>(E_lab) on the n_fr = rho0 / 8 hypersurface
    ax = axes[1, 0]
    t_means, t_q16, t_q84, t_c2_means = [], [], [], []
    for e in energies:
        tc1, wc1 = load_T_hist(cell_map[e]["c1"])
        tc2, wc2 = load_T_hist(cell_map[e]["c2"])
        t_means.append(float(np.average(tc1, weights=wc1)) if len(tc1) > 0 else 0.0)
        q16, q84 = weighted_quantiles(tc1, wc1, [0.16, 0.84])
        t_q16.append(q16)
        t_q84.append(q84)
        t_c2_means.append(float(np.average(tc2, weights=wc2)) if len(tc2) > 0 else 0.0)
    ax.plot(e_agev, t_means, "o-", color="#c0392b", lw=2.3, ms=5.6,
            label="Central ($b_0 < 0.15$) $\\langle T_{\\mathrm{fr}}\\rangle$")
    ax.fill_between(e_agev, t_q16, t_q84, color="#c0392b", alpha=0.18, label="Central 16–84% Cell Spread")
    ax.plot(e_agev, t_c2_means, "s--", color="#1f618d", lw=2.0, ms=5.0,
            label="Semi-Central ($0.25 < b_0 < 0.45$) $\\langle T_{\\mathrm{fr}}\\rangle$")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=7.0)
    ax.set_title("(d) Hypersurface Temperature $\\langle T_{\\mathrm{fr}}\\rangle(E_{\\mathrm{lab}})$ at $\\rho_0 / 8$")

    # (e) Midrapidity directed-flow slope dv1/dy0|_{y0=0} vs FOPI
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("A3",  "#d62728", "^", "$A=3$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        sl_mod, sl_exp, e_exp = [], [], []
        for e in energies:
            ut_cut = float(fopi_data[str(e)]["v1_ut0_cut"])
            ys, _, _, v1s, _, ns = load_rap(merged_map[e]["c2"], sp, ut_cut)
            sl_mod.append(fit_midrapidity_slope(ys, v1s, ns >= 15))
            v1_e = fopi_data[str(e)]["v1_y0"]
            if sp in v1_e:
                ey = np.array([p["y0"] for p in v1_e[sp]])
                ev = np.array([p["v1"] for p in v1_e[sp]])
                s_val = fit_midrapidity_slope(ey, ev, np.ones_like(ey, dtype=bool))
                if not math.isnan(s_val):
                    e_exp.append(e / 1000.0)
                    sl_exp.append(s_val)
        ax.plot(e_agev, sl_mod, "-", color=col, lw=2.2, label=f"{lbl} Model")
        if e_exp:
            ax.plot(e_exp, sl_exp, mk, color=col, ms=5.8, mfc="none", mew=1.5, label=f"{lbl} FOPI")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Midrapidity Slope $dv_1 / dy_0|_{|y_0| \\leq 0.45}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", ncol=2, fontsize=6.8)
    ax.set_title("(e) Directed-Flow Slope Excitation Function vs. FOPI")

    # (f) Midrapidity Elliptic Flow v2(u_t0 = 1.0, |y0| < 0.4) vs FOPI
    ax = axes[1, 2]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"),
                             ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$")]:
        v2_mod, v2_ex, e_ex = [], [], []
        for e in energies:
            us, v2s, _, nsu = load_ut0(merged_map[e]["c2"], sp)
            v2_mod.append(interp_v2_at_ut0(us, v2s, nsu >= 35, target=1.0))
            v2_d = fopi_data[str(e)]["v2_ut0_y0_lt_04"]
            if sp in v2_d:
                eu = np.array([p["ut0"] for p in v2_d[sp]])
                ev = np.array([p["v2"] for p in v2_d[sp]])
                val = interp_v2_at_ut0(eu, ev, np.ones_like(eu, dtype=bool), target=1.0)
                if not math.isnan(val):
                    e_ex.append(e / 1000.0)
                    v2_ex.append(val)
        ax.plot(e_agev, v2_mod, "-", color=col, lw=2.2, label=f"{lbl} Model")
        if e_ex:
            ax.plot(e_ex, v2_ex, mk, color=col, ms=5.8, mfc="none", mew=1.5, label=f"{lbl} FOPI")
    ax.axhline(0.0, color="#777", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 1.65)
    ax.set_xlabel("Beam Energy $E_{\\mathrm{lab}}$ [$A$ GeV]")
    ax.set_ylabel("Elliptic Flow $v_2(u_{t0} = 1.0,\\;|y_0| < 0.4)$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower left", ncol=2, fontsize=7.0)
    ax.set_title("(f) In-Plane to Squeeze-Out $v_2(E_{\\mathrm{lab}})$ Transition vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=185)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--fo-oversample", type=int, default=10)
    ap.add_argument("--n-central", type=int, default=5)
    ap.add_argument("--n-semicentral", type=int, default=5)
    ap.add_argument("--eos", type=str, default="sm", choices=["sm", "hm", "md-soft", "md-stiff"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--energies", type=int, nargs="*", default=FOPI_ENERGIES)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    fopi_all = json.load(open(FOPI_ALL_JSON))["data"]
    energies = [e for e in args.energies if str(e) in fopi_all]
    eos_tag = "sm" if args.eos in ("sm", "md-soft") else "hm"
    out_dir = OUT_ROOT / f"{eos_tag}_ntest{args.ntest}_m{args.fo_oversample}"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    b1_hi = 0.15 * B_MAX_AUAU
    b2_lo, b2_hi = 0.25 * B_MAX_AUAU, 0.45 * B_MAX_AUAU

    jobs = []
    for elab in energies:
        for k in range(args.n_central):
            b1 = math.sqrt(((k + 0.5) / args.n_central) * (b1_hi * b1_hi))
            seed = 8100000 + elab * 100 + k * 17
            jobs.append((f"c1_{eos_tag}_{elab:04d}_{k:02d}", elab, b1, args.ntest,
                         args.fo_oversample, args.eos, k < 2, seed, runs_dir))
        for k in range(args.n_semicentral):
            b2 = math.sqrt(b2_lo * b2_lo + ((k + 0.5) / args.n_semicentral) * (b2_hi * b2_hi - b2_lo * b2_lo))
            seed = 8200000 + elab * 100 + k * 17
            jobs.append((f"c2_{eos_tag}_{elab:04d}_{k:02d}", elab, b2, args.ntest,
                         args.fo_oversample, args.eos, k < 2, seed, runs_dir))

    if not args.plot_only:
        print(f"Running {len(jobs)} FP32 GC+Oversampled BUU jobs ({len(energies)} energies x "
              f"{args.n_central}+{args.n_semicentral} events, N_test={args.ntest}, M_over={args.fo_oversample}, "
              f"EOS={args.eos}, workers={args.workers})...", flush=True)
        t_start = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single_job, j) for j in jobs]
            for fut in as_completed(futs):
                tag, elab, dt_s, info, _ = fut.result()
                done += 1
                elapsed = time.time() - t_start
                print(f"  [{done:3d}/{len(jobs)}] {tag:20s} ({dt_s:5.1f}s | wall {elapsed:6.1f}s) | {info}",
                      flush=True)

    merged_map = {}
    cell_map = {}
    summary_rows = []
    for elab in energies:
        c1_paths = [runs_dir / f"c1_{eos_tag}_{elab:04d}_{k:02d}.csv" for k in range(args.n_central)]
        c2_paths = [runs_dir / f"c2_{eos_tag}_{elab:04d}_{k:02d}.csv" for k in range(args.n_semicentral)]
        c1_merged = out_dir / f"c1_{eos_tag}_{elab:04d}.csv"
        c2_merged = out_dir / f"c2_{eos_tag}_{elab:04d}.csv"
        merge_ensemble(c1_paths, c1_merged)
        merge_ensemble(c2_paths, c2_merged)
        merged_map[elab] = {"c1": c1_merged, "c2": c2_merged}
        cell_map[elab] = {
            "c1": [runs_dir / f"c1_{eos_tag}_{elab:04d}_{k:02d}.cells.csv" for k in range(min(2, args.n_central))],
            "c2": [runs_dir / f"c2_{eos_tag}_{elab:04d}_{k:02d}.cells.csv" for k in range(min(2, args.n_semicentral))],
        }
        png_e = ASSETS_DIR / f"fopi_gc_oversample_auau{elab:04d}.png"
        t_mean = plot_energy_comparison(
            elab, c1_merged, c2_merged, cell_map[elab], fopi_all[str(elab)],
            args.eos, args.ntest, args.fo_oversample, args.n_central, args.n_semicentral, png_e
        )
        yd = load_yields(c1_merged)
        summary_rows.append((elab, t_mean, yd))
        print(f"Saved -> {png_e} | <T_fr>={t_mean:5.2f} MeV | "
              f"p={yd['p'][0]:6.2f} d={yd['d'][0]:5.2f} t={yd['t'][0]:5.2f} "
              f"3He={yd['3He'][0]:5.2f} 4He={yd['4He'][0]:5.2f} "
              f"pi-={yd['pi-'][0]:5.2f} pi+={yd['pi+'][0]:5.2f}", flush=True)

    if len(energies) >= 2:
        sum_png = ASSETS_DIR / "fopi_gc_oversample_all_energies_summary.png"
        plot_excitation_summary(energies, merged_map, cell_map, fopi_all, args.eos,
                                args.ntest, args.fo_oversample, sum_png)
        print(f"Saved excitation summary -> {sum_png}", flush=True)


if __name__ == "__main__":
    main()
