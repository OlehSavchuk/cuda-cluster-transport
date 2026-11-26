#!/usr/bin/env python3
"""
Compare FP64 (double) vs FP32 (float) CUDA Cluster Transport physics & precision:
  1. Threshold kinematics: Naive vs Cancellation-Free |E2*p1 - E1*p2|^2 - |p1 x p2|^2
  2. Conservation laws (|dH|/H, |dP|) across --in-medium-ec off / on / exact
  3. Full FOPI Au+Au @ 250 A MeV observables: dN/dy^(0), v1(y0), v2(ut0), and cluster yields
"""

import csv
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_cluster_transport")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_FP64 = REPO_ROOT / "build" / "cluster_transport"
BIN_FP32 = REPO_ROOT / "build" / "cluster_transport_fp32"
OUT_DIR = REPO_ROOT / "output"
ASSETS_DIR = REPO_ROOT / "assets"
FOPI_JSON = Path(__file__).resolve().parent / "fopi_exp_auau250.json"

BG_DARK = "#0b0f19"
PANEL_BG = "#111827"
GRID_COL = "#243049"
TEXT_MAIN = "#e5e7eb"
TEXT_DIM = "#9ca3af"

COL_P = "#38bdf8"
COL_D = "#fbbf24"
COL_T = "#34d399"
COL_HE3 = "#2dd4bf"
COL_HE4 = "#fb923c"
COL_FP32 = "#f472b6"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))


def run_fp32_job(job):
    k, b_fm, ntest, seed, tmp_dir = job
    out_csv = tmp_dir / f"fp32_{k:02d}.csv"
    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--eos", "md-soft", "--in-medium-ec", "exact",
        "--ntest", str(ntest), "--box", "48.0", "--grid", "24",
        "--steps", "240", "--dt", "0.30", "--seed", str(seed),
        "--flow-csv", str(out_csv),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return k, out_csv


def compute_threshold_kinematics_curves():
    """Compare naive vs cancellation-free p_cm error for boosted 4He + 4He in FP32 & FP64."""
    m = 3.7273794  # 4He mass [GeV]
    beta = 0.52    # Boost velocity along tilted axis
    gam = 1.0 / math.sqrt(1.0 - beta * beta)
    ekin_cm = np.logspace(-7, 0, 90)  # 0.1 keV to 1 GeV
    err_naive_32, err_cf_32, err_naive_64, err_cf_64 = [], [], [], []
    for ek in ekin_cm:
        e_star = m + 0.5 * ek
        p_true = math.sqrt(max(0.0, e_star * e_star - m * m))
        # Boost (+p_true, 0, 0) rotated by 35 deg and boosted along z by beta in exact high precision
        px0, pz0 = p_true * math.sin(0.61), p_true * math.cos(0.61)
        p1 = np.array([gam * (e_star + beta * pz0),  px0, 0.0, gam * (pz0 + beta * e_star)], dtype=np.float64)
        p2 = np.array([gam * (e_star - beta * pz0), -px0, 0.0, gam * (-pz0 + beta * e_star)], dtype=np.float64)
        for arr, dt, e_n, e_c in [(p1, np.float32, err_naive_32, err_cf_32),
                                  (p1, np.float64, err_naive_64, err_cf_64)]:
            q1, q2, mf = p1.astype(dt), p2.astype(dt), dt(m)
            q1[0] = np.sqrt(mf * mf + q1[1]*q1[1] + q1[2]*q1[2] + q1[3]*q1[3])
            q2[0] = np.sqrt(mf * mf + q2[1]*q2[1] + q2[2]*q2[2] + q2[3]*q2[3])
            # 1. Naive: (p1.p2)^2 - m^4
            dot = q1[0]*q2[0] - (q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
            q2_naive = max(dt(0.0), dot * dot - (mf * mf) * (mf * mf))
            s_naive = mf * mf + mf * mf + dt(2.0) * dot
            pcm_naive = float(np.sqrt(q2_naive) / np.sqrt(max(dt(1e-12), s_naive)))
            # 2. Cancellation-free: |E2*p1 - E1*p2|^2 - |p1 x p2|^2
            dvec = q2[0] * q1[1:] - q1[0] * q2[1:]
            cvec = np.cross(q1[1:], q2[1:])
            q2_cf = max(dt(0.0), np.dot(dvec, dvec) - np.dot(cvec, cvec))
            s_cf = dt(2.0) * mf * mf + dt(2.0) * np.sqrt((mf * mf)**2 + q2_cf)
            pcm_cf = float(np.sqrt(q2_cf) / np.sqrt(s_cf))
            e_n.append(max(1e-16, abs(pcm_naive - p_true) / p_true))
            e_c.append(max(1e-16, abs(pcm_cf - p_true) / p_true))
    return ekin_cm * 1e3, np.array(err_naive_32), np.array(err_cf_32), np.array(err_naive_64), np.array(err_cf_64)

import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fopi_highstat import merge_ensemble_csvs
from generate_fopi_comparison import style_dark, load_sim_flow, sym_curve, poly_fit_band


def measure_conservation_modes():
    out = {}
    for prec, bin_p in [("FP64", BIN_FP64), ("FP32", BIN_FP32)]:
        out[prec] = {}
        for mode_ec in ["off", "on", "exact"]:
            cmd = [str(bin_p), "--mode", "aniso", "--eos", "md-stiff", "--in-medium-ec", mode_ec,
                   "--npart", "20000", "--ntest", "16", "--box", "20.0", "--grid", "10",
                   "--temp", "80.0", "--steps", "45", "--dt", "0.20", "--seed", "7777"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for ln in res.stdout.splitlines():
                m = re.search(r"\|dH\|/H = ([\deE\+\-\.]+).*?\|dP\| = ([\deE\+\-\.]+)", ln)
                if m:
                    out[prec][mode_ec] = (max(1e-16, float(m.group(1))), max(1e-16, float(m.group(2))))
    return out


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()

    # 1. Run 10-seed b*db FP32 ensemble (10 x N_test=500 = 5,000 events, exact same seeds as FP64!)
    fp32_csv = OUT_DIR / "fopi_c2_mdsoft_fp32.csv"
    tmp_dir = OUT_DIR / "tmp_fp32_runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        jobs.append((k, B_MAX_AUAU * math.sqrt(b0_sq), 500, 100000 + k * 7919, tmp_dir))
    print("Running 5,000-event FP32 (float) Au+Au ensemble (same seeds as FP64)...")
    t0 = time.time()
    paths = {}
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as ex:
        for k, p in ex.map(run_fp32_job, jobs):
            paths[k] = p
    merge_ensemble_csvs([paths[k] for k in range(10)], fp32_csv)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"FP32 5,000-event ensemble done in {time.time() - t0:.1f}s!")

    cons = measure_conservation_modes()
    ek_mev, en32, ec32, en64, ec64 = compute_threshold_kinematics_curves()
    fopi = json.loads(FOPI_JSON.read_text())
    rap64, ut64, y64 = load_sim_flow(OUT_DIR / "fopi_c2_mdsoft.csv")
    rap32, ut32, y32 = load_sim_flow(fp32_csv)

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), dpi=150)
    fig.subplots_adjust(left=0.058, right=0.985, top=0.89, bottom=0.08, wspace=0.25, hspace=0.31)
    fig.suptitle("Precision Benchmark: FP64 (double) vs. FP32 (float) in CUDA Cluster Transport",
                 fontsize=13.8, fontweight="bold", color=TEXT_MAIN, y=0.965)
    fig.text(0.5, 0.930,
             "Identical 5,000-Event Au + Au @ 250 A MeV Ensembles (1.97M Test Nucleons, MD-Soft, Exact Seeds)",
             ha="center", fontsize=9.2, color=TEXT_DIM)

    # Panel (a): Threshold Kinematics Error
    ax = axes[0, 0]
    ax.loglog(ek_mev, en32, "--", lw=2.0, color="#ef4444", label="FP32 Naive (p₁·p₂)² - m⁴")
    ax.loglog(ek_mev, ec32, "-",  lw=2.3, color=COL_FP32,  label="FP32 Cancellation-Free (Ours)")
    ax.loglog(ek_mev, en64, ":",  lw=1.8, color="#94a3b8", label="FP64 Naive (p₁·p₂)² - m⁴")
    ax.loglog(ek_mev, ec64, "-",  lw=2.1, color=COL_P,     label="FP64 Cancellation-Free (Ours)")
    ax.set_xlabel(r"Pair CM Kinetic Energy $E_{\rm kin,cm}$ [MeV]")
    ax.set_ylabel(r"Relative Error $|\Delta p_{\rm cm}| / p_{\rm cm}$")
    ax.set_title(r"(a) Boosted ${}^4{\rm He}+{}^4{\rm He}$ Threshold Kinematics")
    ax.set_ylim(1e-16, 3.0)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Panel (b): Hamiltonian & Momentum Conservation Across Modes (FP64 vs FP32)
    ax = axes[0, 1]
    modes = ["off", "on", "exact"]
    x_m = np.arange(len(modes))
    w = 0.34
    dh64 = [cons["FP64"][m][0] for m in modes]
    dh32 = [cons["FP32"][m][0] for m in modes]
    ax.bar(x_m - 0.5 * w, dh64, w, color=COL_P, alpha=0.88, label="FP64 (double, 64-bit)")
    ax.bar(x_m + 0.5 * w, dh32, w, color=COL_FP32, alpha=0.88, label="FP32 (float, 32-bit)")
    ax.set_yscale("log")
    ax.set_xticks(x_m)
    ax.set_xticklabels(["--in-medium-ec off", "--in-medium-ec on", "--in-medium-ec exact"], fontsize=8.3)
    ax.set_ylim(1e-16, 3e-1)
    ax.set_ylabel(r"Relative Hamiltonian Drift $|\Delta H| / H$")
    ax.set_title("(b) Energy Conservation: FP64 vs. FP32 (MD-Stiff)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8.0, facecolor=PANEL_BG, edgecolor=GRID_COL)
    for xm, v64, v32 in zip(x_m, dh64, dh32):
        ax.text(xm - 0.5 * w, v64 * 2.2, f"{v64:.1e}", ha="center", fontsize=7.2, color=COL_P)
        ax.text(xm + 0.5 * w, v32 * 2.2, f"{v32:.1e}", ha="center", fontsize=7.2, color=COL_FP32)

    # Panel (c): Rapidity Distributions dN/dy^(0) — FP64 Lines vs FP32 Open Markers
    ax = axes[0, 2]
    for key, lbl, col, mk, sc in [
        ("n", "n", "#93c5fd", "s", 1.0), ("p", "p", COL_P, "o", 1.0),
        ("d", "d", COL_D, "^", 1.0),     ("t", "t", COL_T, "v", 1.0),
        ("3He", "³He", COL_HE3, "D", 1.0), ("4He", "⁴He (×5)", COL_HE4, "*", 5.0),
    ]:
        r6, r3 = rap64[key], rap32[key]
        ax.plot([x["y0"] for x in r6], [x["dN_dy0"] * sc for x in r6], "-", lw=1.9, color=col, label=f"{lbl} (FP64)")
        ax.plot([x["y0"] for x in r3], [x["dN_dy0"] * sc for x in r3], mk, ms=5.0, mfc="none", mec=col, mew=1.3)
    ax.plot([], [], "o", mfc="none", mec=TEXT_MAIN, ms=5.0, label="Open markers = FP32")
    ax.set_xlabel(r"Scaled Rapidity $y_0 = y / y_{\rm proj}$")
    ax.set_ylabel(r"$dN / dy_0$")
    ax.set_title(r"(c) Rapidity Spectra $dN/dy_0$: FP64 vs. FP32")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.1, ncol=2, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Panel (d): Directed Flow v1(y0) (ut0 > 0.8) — FP64 vs FP32 vs FOPI
    ax = axes[1, 0]
    yg = np.linspace(-1.05, 1.05, 120)
    ax.axhline(0.0, color=TEXT_DIM, lw=0.8, alpha=0.6)
    for sp, lbl, col, mk in [("p", "p", COL_P, "o"), ("d", "d", COL_D, "s"), ("t", "t", COL_T, "^")]:
        de = fopi["v1_y0_ut0_gt_08"][sp]
        ax.errorbar([x["y0"] for x in de], [x["v1"] for x in de], yerr=[x["err"] for x in de],
                    fmt=mk, ms=4.2, mfc="none", mec=col, mew=1.1, ecolor=col, alpha=0.45)
        y6, v6, e6 = sym_curve(rap64[sp], "v1_ut08", "v1_ut08_err", "n_ut08", odd=True, min_n=18)
        y3, v3, e3 = sym_curve(rap32[sp], "v1_ut08", "v1_ut08_err", "n_ut08", odd=True, min_n=18)
        f6, _ = poly_fit_band(y6[np.abs(y6)<=1.05], v6[np.abs(y6)<=1.05], e6[np.abs(y6)<=1.05], [1, 3], yg)
        f3, _ = poly_fit_band(y3[np.abs(y3)<=1.05], v3[np.abs(y3)<=1.05], e3[np.abs(y3)<=1.05], [1, 3], yg)
        ax.plot(yg, f6, "-",  lw=2.2, color=col, label=f"FP64 {lbl}")
        ax.plot(yg, f3, "--", lw=2.0, color=COL_FP32 if sp=="p" else col, alpha=0.9, label=f"FP32 {lbl}")
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.68, 0.68)
    ax.set_xlabel(r"Scaled Rapidity $y_0$"); ax.set_ylabel(r"$v_1(u_{t0} > 0.8)$")
    ax.set_title(r"(d) Directed Flow $v_1(y_0)$: FP64 (solid) vs. FP32 (dashed)")
    ax.grid(True); ax.legend(loc="upper left", fontsize=7.3, ncol=2, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Panel (e): Differential Squeeze-Out v2(ut0) — FP64 vs FP32 vs FOPI
    ax = axes[1, 1]
    ax.axhline(0.0, color=TEXT_DIM, lw=0.8, alpha=0.6)
    for sp, lbl, col, mk, umax in [("p", "p", COL_P, "o", 2.0), ("d", "d", COL_D, "s", 1.85), ("t", "t", COL_T, "^", 1.45)]:
        de = fopi["v2_ut0_y0_lt_04"][sp]
        ax.errorbar([x["ut0"] for x in de], [x["v2"] for x in de], yerr=[x["err"] for x in de],
                    fmt=mk, ms=4.2, mfc="none", mec=col, mew=1.1, ecolor=col, alpha=0.45)
        ug_i = np.linspace(0.05, umax, 80)
        for ut_d, ls, tag in [(ut64[sp], "-", "FP64"), (ut32[sp], "--", "FP32")]:
            pts = [r for r in ut_d if r["n_stat"] >= 18]
            fu, _ = poly_fit_band(np.array([r["ut0"] for r in pts]), np.array([r["v2"] for r in pts]),
                                  np.array([r["v2_err"] for r in pts]), [1, 2], ug_i)
            ax.plot(ug_i, fu, ls, lw=2.1, color=(COL_FP32 if (tag=="FP32" and sp=="p") else col), label=f"{tag} {lbl}")
    ax.set_xlim(0.0, 2.05); ax.set_ylim(-0.28, 0.08)
    ax.set_xlabel(r"Scaled Transverse 4-Velocity $u_{t0}$"); ax.set_ylabel(r"$v_2(|y_0| < 0.4)$")
    ax.set_title(r"(e) Squeeze-Out $v_2(u_{t0})$: FP64 (solid) vs. FP32 (dashed)")
    ax.grid(True); ax.legend(loc="lower left", fontsize=7.3, ncol=2, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Panel (f): Relative Multiplicity Difference (Y_FP32 - Y_FP64) / Y_FP64 [%]
    ax = axes[1, 2]
    sps = ["p", "n", "d", "t", "3He", "4He"]
    lbls = ["p", "n", "d", "t", "³He", "⁴He"]
    rel_pct = [100.0 * (y32[s] - y64[s]) / y64[s] for s in sps]
    # 1-sigma Poisson uncertainty of difference between two independent 5,000-event ensembles
    stat_pct = [100.0 * math.sqrt(2.0 / max(1.0, 5000.0 * y64[s])) for s in sps]
    xp = np.arange(len(sps))
    ax.axhline(0.0, color=TEXT_MAIN, lw=1.0, alpha=0.8)
    ax.bar(xp, rel_pct, 0.52, yerr=stat_pct, capsize=3.5,
           color=[COL_P, "#93c5fd", COL_D, COL_T, COL_HE3, COL_HE4], alpha=0.88, ecolor="#e2e8f0")
    for xi, rv in zip(xp, rel_pct):
        off = 0.25 if rv >= 0 else -0.45
        ax.text(xi, rv + off, f"{rv:+.2f}%", ha="center", fontsize=8.0, color=TEXT_MAIN)
    ax.set_xticks(xp); ax.set_xticklabels(lbls, fontsize=9.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_ylabel(r"$(Y_{\rm FP32} - Y_{\rm FP64}) / Y_{\rm FP64}$ [%]")
    ax.set_title("(f) Freeze-Out Yield Drift: FP32 vs. FP64 (5,000 Events)")
    ax.grid(True, axis="y")

    out_png = ASSETS_DIR / "fp64_vs_fp32_comparison.png"
    fig.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)
    print(f"[Generated] {out_png}")
    for s, rv, sv in zip(sps, rel_pct, stat_pct):
        print(f"  Yield {s:4s}: FP64={y64[s]:6.2f}, FP32={y32[s]:6.2f}, diff={rv:+.3f}% (+- {sv:.3f}% stat)")


if __name__ == "__main__":
    main()
