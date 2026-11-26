#!/usr/bin/env python3
"""
Demonstration of 1 single BUU run + freeze-out oversampling (--fo-oversample 50)
with systematic backflow resampling-to-N_test & Mott remelting (--backflow remelt)
in FOPI Au+Au @ 1.2 A GeV (HM EOS, K0 = 380 MeV, dx = 1.00 fm = 68^3):
  - Single BUU run (N_test = 80, M_over = 1  ->    80 eff. events, ~45s)
  - Single BUU run (N_test = 80, M_over = 50 -> 4,000 eff. cluster events, ~47s!)
  - Multi-run 1,600-event HM ensemble reference (4 runs x N_test = 400) & FOPI Data.
"""

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
FOPI_JSON = ROOT / "scripts" / "fopi_exp_auau1200.json"
ENS_DIR = ROOT / "output" / "highstat_auau1200"
OUT_DIR = ROOT / "output" / "single_buu_oversample_1200"
PNG_OUT = ROOT / "assets" / "fopi_auau1200_single_buu_oversample.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160


def run_one_buu(job):
    tag, b_fm, ntest, m_over, seed = job
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fl_csv = OUT_DIR / f"{tag}.csv"
    cells_csv = OUT_DIR / f"{tag}.cells.csv"
    t_file = OUT_DIR / f"{tag}.time"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and t_file.exists():
        return tag, float(t_file.read_text().strip()), "cached", fl_csv
    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "1200.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "68.0", "--grid", "68",
        "--steps", "220", "--dt", "0.20", "--substeps", "2",
        "--eos", "hm", "--in-medium-ec", "on", "--pauli", "on", "--coulomb", "on",
        "--cluster-mode", "freezeout", "--backflow", "remelt",
        "--sampling", "gc", "--fo-oversample", str(m_over), "--anisotropy", "on",
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "2",
        "--seed", str(seed), "--flow-csv", str(fl_csv), "--freezeout-cells-csv", str(cells_csv),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    t_file.write_text(f"{dt_s:.2f}\n")
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "done"), fl_csv


def load_yields(p: Path):
    return {r["species"]: (float(r["yield_per_event"]), float(r.get("yield_err", 0.0)))
            for r in csv.DictReader(open(str(p) + ".yields.csv"))}


def load_rap(p: Path, sp: str):
    rs = [r for r in csv.DictReader(open(p)) if r["species"] == sp]
    return (np.array([float(r["y0"]) for r in rs]),
            np.array([float(r["dN_dy0"]) for r in rs]),
            np.array([float(r["v1_ut04"]) for r in rs]),
            np.array([float(r["v1_ut04_err"]) for r in rs]),
            np.array([float(r["n_ut04"]) for r in rs]))


def load_ut0(p: Path, sp: str):
    rs = [r for r in csv.DictReader(open(str(p) + ".ut0.csv")) if r["species"] == sp]
    return (np.array([float(r["ut0"]) for r in rs]),
            np.array([float(r["v2"]) for r in rs]),
            np.array([float(r["v2_err"]) for r in rs]),
            np.array([float(r["n_stat"]) for r in rs]))


def plot_demo(paths, dt_map, fopi):
    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.1, "axes.labelsize": 9.3,
        "legend.fontsize": 7.0, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.6, 10.4), dpi=185)
    fig.suptitle(
        "1 Single BUU Run ($N_{\\mathrm{test}}=80$) + $50\\times$ Freeze-Out Oversampling (Backflow Resampled to $N_{\\mathrm{test}}$ + Remelt) — FOPI Au+Au @ 1.2 $A$ GeV (HM)",
        fontsize=11.1, fontweight="bold", y=0.98
    )

    # (a) Central (b_rms = 1.42 fm) Multiplicities: 1 BUU (M=1) vs 1 BUU (M=50) vs 1,600-ev Ensemble vs FOPI
    ax = axes[0, 0]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li", "pi-", "pi+"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$",
            "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$", "$\\pi^-$", "$\\pi^+$"]
    xp = np.arange(len(sps)); w = 0.25
    y_m1  = load_yields(paths["c1_m1"])
    y_m50 = load_yields(paths["c1_m50"])
    y_ens = load_yields(ENS_DIR / "c1_hm_1200.csv")
    ax.bar(xp - w, [max(1e-4, y_m1[s][0]) for s in sps],  width=w, color="#bdc3c7",
           label=f"1 BUU Run ($M=1$, {dt_map['c1_m1']:.0f}s)")
    ax.bar(xp,     [max(1e-4, y_m50[s][0]) for s in sps], width=w, color="#1f618d",
           label=f"1 BUU Run ($M=50$, {dt_map['c1_m50']:.0f}s)")
    ax.bar(xp + w, [max(1e-4, y_ens[s][0]) for s in sps], yerr=[y_ens[s][1] for s in sps],
           width=w, color="#e67e22", capsize=2, label="4-Run Ensemble ($N_{\\mathrm{ev}}=1600$, 400s)")
    e_idx = [i for i, s in enumerate(sps) if s in fopi["central_yields_b0_lt_015"]]
    ax.errorbar(xp[e_idx], [fopi["central_yields_b0_lt_015"][sps[i]]["val"] for i in e_idx],
                yerr=[fopi["central_yields_b0_lt_015"][sps[i]]["err"] for i in e_idx],
                fmt="ko", ms=5.6, capsize=3, lw=1.5, zorder=8, label="FOPI Exp. ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 220.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (Log Scale)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=6.7)
    ax.set_title("(a) Central Yields: 1 Oversampled BUU Run vs. Full Ensemble")

    # (b) Direct M=1 vs M=50 Squeeze-Out Noise Reduction on Triton (t) & Alpha (4He) from 1 BUU Run
    ax = axes[0, 1]
    for sp, col, lbl in [("t", "#d62728", "$t$"), ("4He", "#8e44ad", "$^4\\mathrm{He}$")]:
        u1,  v1,  e1,  n1  = load_ut0(paths["c2_m1"], sp)
        u50, v50, e50, n50 = load_ut0(paths["c2_m50"], sp)
        s1  = (u1 >= 0.25)  & (u1 <= 1.75)  & (n1 >= 3)
        s50 = (u50 >= 0.25) & (u50 <= 1.75) & (n50 >= 120)
        ax.errorbar(u1[s1], v1[s1], yerr=e1[s1], fmt="o--", color=col, alpha=0.38, ms=4.2, lw=1.1,
                    capsize=2, label=f"{lbl} 1 BUU ($M=1$, noisy)")
        ax.plot(u50[s50], v50[s50], "-", color=col, lw=2.4, label=f"{lbl} 1 BUU ($M=50$, $4000$ eff. ev)")
        ax.fill_between(u50[s50], v50[s50] - e50[s50], v50[s50] + e50[s50], color=col, alpha=0.18)
    eu_t = fopi["v2_ut0_y0_lt_04"]["t"]
    ax.errorbar([p["ut0"] for p in eu_t[::2]], [p["v2"] for p in eu_t[::2]],
                yerr=[p["err"] for p in eu_t[::2]], fmt="^", color="#111", ms=5.5,
                mfc="none", mew=1.5, capsize=2.5, zorder=9, label="$t$ FOPI Exp.")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.22, 1.65); ax.set_ylim(-0.26, 0.06)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=6.7)
    ax.set_title("(b) Variance Collapse on $t$ & $^4\\mathrm{He}$ $v_2$: $M=1$ vs. $M=50$")

    # (c) Semi-Central Rapidity Spectra dN/dy0 (including rare 6Li x 40!): M=50 vs M=1
    ax = axes[0, 2]
    for sp, col, mult, lbl in [("p",   "#2c3e50", 0.35, "$p \\times 0.35$"),
                               ("d",   "#1f77b4", 1.0,  "$d$"),
                               ("t",   "#27ae60", 1.0,  "$t$"),
                               ("4He", "#8e44ad", 2.0,  "$^4\\mathrm{He} \\times 2$"),
                               ("6Li", "#c0392b", 40.0, "$^6\\mathrm{Li} \\times 40$")]:
        y1,  d1,  _, _, _ = load_rap(paths["c2_m1"], sp)
        y50, d50, _, _, _ = load_rap(paths["c2_m50"], sp)
        ax.plot(y1,  d1 * mult,  ":", color=col, lw=1.3, alpha=0.65, label=f"{lbl} ($M=1$)")
        ax.plot(y50, d50 * mult, "-", color=col, lw=2.2, label=f"{lbl} ($M=50$)")
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ ($b = 4.87\\;\\mathrm{fm}$)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.4)
    ax.set_title("(c) 1 BUU Run $dN/dy_0$: $M=50$ (solid) vs. $M=1$ (dotted)")
    # (d) Directed Flow v1(y0) (u_t0 > 0.4): 1 BUU Run (M=50, solid) vs 1,600-ev Ensemble (dashed) vs FOPI
    ax = axes[1, 0]
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("A3",  "#d62728", "^", "$A=3$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        y50, _, v50, e50, n50 = load_rap(paths["c2_m50"], sp)
        ye,  _, ve,  _,   ne  = load_rap(ENS_DIR / "c2_hm_1200.csv", sp)
        s50 = (np.abs(y50) <= 1.08) & (n50 >= 15)
        se  = (np.abs(ye)  <= 1.08) & (ne >= 10)
        ax.plot(y50[s50], v50[s50], "-",  color=col, lw=2.3, label=f"{lbl} 1 BUU ($M=50$)")
        ax.fill_between(y50[s50], v50[s50] - e50[s50], v50[s50] + e50[s50], color=col, alpha=0.14)
        ax.plot(ye[se],   ve[se],   "--", color=col, lw=1.6, alpha=0.80, label=f"{lbl} Ens.")
        if sp in fopi["v1_y0_ut0_gt_04"]:
            ep = fopi["v1_y0_ut0_gt_04"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.86, 0.86)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.4$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=3, fontsize=6.1)
    ax.set_title("(d) Directed Flow $v_1(y_0)$: 1 Oversampled BUU Run vs. FOPI")

    # (e) Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4): 1 BUU Run (M=50) vs FOPI Data
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"),
                             ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        u50, v50, e50, n50 = load_ut0(paths["c2_m50"], sp)
        s50 = (u50 >= 0.25) & (u50 <= 1.98) & (n50 >= 120)
        ax.plot(u50[s50], v50[s50], "-", color=col, lw=2.3, label=f"{lbl} 1 BUU ($M=50$)")
        ax.fill_between(u50[s50], v50[s50] - e50[s50], v50[s50] + e50[s50], color=col, alpha=0.15)
        if sp in fopi["v2_ut0_y0_lt_04"]:
            eu = fopi["v2_ut0_y0_lt_04"][sp]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.22, 2.02); ax.set_ylim(-0.26, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=2, fontsize=6.4)
    ax.set_title("(e) Smooth Squeeze-Out $v_2(u_{t0})$ from 1 Oversampled BUU Run")

    # (f) Full A = 1..6 Mass-Scaling Hierarchy (including rare 6He & 6Li!) from 1 BUU Run (M=50)
    ax = axes[1, 2]
    for sp, col, lw_v, lbl in [("p",   "#2c3e50", 1.8, "$p\\;(A=1)$"),
                               ("d",   "#1f77b4", 1.9, "$d\\;(A=2)$"),
                               ("A3",  "#27ae60", 2.0, "$A=3\\;(t+{}^3\\mathrm{He})$"),
                               ("4He", "#8e44ad", 2.1, "$^4\\mathrm{He}\\;(A=4)$"),
                               ("6He", "#e67e22", 2.2, "$^6\\mathrm{He}\\;(A=6)$"),
                               ("6Li", "#c0392b", 2.4, "$^6\\mathrm{Li}\\;(A=6)$")]:
        y50, _, v50, e50, n50 = load_rap(paths["c2_m50"], sp)
        s50 = (np.abs(y50) <= 1.08) & (n50 >= 6)
        ax.plot(y50[s50], v50[s50], "-", color=col, lw=lw_v, label=lbl)
        ax.fill_between(y50[s50], v50[s50] - e50[s50], v50[s50] + e50[s50], color=col, alpha=0.12)
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.92, 0.92)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.4$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=6.9)
    ax.set_title("(f) $A = 1 \\dots 6$ Flow Hierarchy (incl. $^6\\mathrm{He}, ^6\\mathrm{Li}$) from 1 BUU Run!")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_OUT, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_OUT}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    b_c1 = math.sqrt(0.5) * 0.15 * B_MAX_AUAU
    b_c2 = math.sqrt(0.5 * ((0.25 * B_MAX_AUAU) ** 2 + (0.45 * B_MAX_AUAU) ** 2))
    jobs = [
        ("c1_m1",  b_c1, 80, 1,  771001),
        ("c1_m50", b_c1, 80, 50, 771001),
        ("c2_m1",  b_c2, 80, 1,  772001),
        ("c2_m50", b_c2, 80, 50, 772001),
    ]
    print("Running 1 central + 1 semi-central BUU run (M=1 vs M=50 paired seeds, N_test=80)...", flush=True)
    paths, dt_map = {}, {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one_buu, j) for j in jobs]):
            tag, dt_s, info, p = fut.result()
            paths[tag] = p; dt_map[tag] = dt_s
            print(f"  {tag:8s} ({dt_s:5.1f}s) | {info}", flush=True)
    plot_demo(paths, dt_map, fopi)
    for tag in ["c1_m1", "c1_m50"]:
        yd = load_yields(paths[tag])
        print(f"  {tag:8s} ({dt_map[tag]:4.1f}s): p={yd['p'][0]:6.2f} | d={yd['d'][0]:5.2f} | t={yd['t'][0]:5.2f} | "
              f"3He={yd['3He'][0]:5.2f} | 4He={yd['4He'][0]:5.2f} | 6He={yd['6He'][0]:.3f} | 6Li={yd['6Li'][0]:.3f} | "
              f"pi-={yd['pi-'][0]:5.2f} | pi+={yd['pi+'][0]:5.2f}")


if __name__ == "__main__":
    main()
