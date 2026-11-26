#!/usr/bin/env python3
"""
High-statistics FOPI Au+Au @ 1.2 A GeV (1200 A MeV) benchmark comparing
SM (MD-Soft, K0=215 MeV) vs. HM (MD-Stiff, K0=380 MeV) in Grand-Canonical
Poisson HCGF (--sampling gc) on dx = 1.00 fm (68^3) cells with 3D Coulomb,
exact T^00 potential, 6D Pauli blocking, and coupled Delta(1232) + pion transport.
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
OUT_DIR = ROOT / "output" / "highstat_auau1200"
RUNS_DIR = OUT_DIR / "runs"
PNG_OUT = ROOT / "assets" / "fopi_auau1200_sm_vs_hm.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160


def run_job(job):
    tag, b_fm, ntest, eos_flag, cmode, sampling_mode, save_cells, seed = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    cells_csv = RUNS_DIR / f"{tag}.cells.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and (not save_cells or cells_csv.exists()):
        return tag, 0.0, "cached", fl_csv

    # At 1.2 A GeV (beta_cm = 0.625 c, t_overlap = 14.4 fm/c), dt = 0.20 fm/c x 220 steps = 44.0 fm/c
    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "1200.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "68.0", "--grid", "68",
        "--steps", "220", "--dt", "0.20", "--substeps", "2",
        "--eos", eos_flag, "--in-medium-ec", "on", "--pauli", "on", "--coulomb", "on",
        "--cluster-mode", cmode, "--backflow", "remelt",
        "--sampling", sampling_mode, "--anisotropy", "on",
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "2",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    if save_cells and cmode != "microscopic":
        cmd += ["--freezeout-cells-csv", str(cells_csv)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "micro-done"), fl_csv


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
            dndy0_e = math.sqrt(sum((x - dndy0_m) ** 2 for x in dndy0s) / (m_runs * max(1, m_runs - 1)))
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
            ye = math.sqrt(sum((y - ym) ** 2 for y in ys) / (m_runs * max(1, m_runs - 1)))
            w.writerow([sp, f"{ym:.6f}", f"{ye:.6f}"])


def load_yields(p: Path):
    return {r["species"]: (float(r["yield_per_event"]), float(r.get("yield_err", 0.0)))
            for r in csv.DictReader(open(str(p) + ".yields.csv"))}


def load_rap(p: Path, sp: str):
    rs = [r for r in csv.DictReader(open(p)) if r["species"] == sp]
    return (np.array([float(r["y0"]) for r in rs]),
            np.array([float(r["dN_dy0"]) for r in rs]),
            np.array([float(r.get("dN_dy0_err", 0.0)) for r in rs]),
            np.array([float(r["v1_ut04"]) for r in rs]),
            np.array([float(r["v1_ut04_err"]) for r in rs]),
            np.array([float(r["n_ut04"]) for r in rs]))


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


def run_ensembles():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    b1_hi = 0.15 * B_MAX_AUAU
    b2_lo, b2_hi = 0.25 * B_MAX_AUAU, 0.45 * B_MAX_AUAU

    # Paired-seed SM (MD-Soft, K0=215 MeV) vs HM (MD-Stiff, K0=380 MeV) Grand-Canonical jobs
    # at E_lab = 1200 A MeV on dx = 1.00 fm (68^3): 4 b*db points x 400 test particles = 1,600 events/EOS/bin
    for k in range(4):
        b1 = math.sqrt(((k + 0.5) / 4.0) * (b1_hi * b1_hi))
        b2 = math.sqrt(b2_lo * b2_lo + ((k + 0.5) / 4.0) * (b2_hi * b2_hi - b2_lo * b2_lo))
        jobs.append((f"c1_sm_1200_{k:02d}", b1, 400, "md-soft",  "freezeout", "gc", k < 2, 1201000 + k * 157))
        jobs.append((f"c2_sm_1200_{k:02d}", b2, 400, "md-soft",  "freezeout", "gc", k < 2, 1202000 + k * 157))
        jobs.append((f"c1_hm_1200_{k:02d}", b1, 400, "md-stiff", "freezeout", "gc", k < 2, 1201000 + k * 157))
        jobs.append((f"c2_hm_1200_{k:02d}", b2, 400, "md-stiff", "freezeout", "gc", k < 2, 1202000 + k * 157))

    print(f"Running {len(jobs)} Au+Au @ 1.2 AGeV dx = 1.00 fm (68^3) SM vs HM jobs across 4 workers...", flush=True)
    t0 = time.time()
    out = {}
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            tag, dt_s, info, p = fut.result()
            out[tag] = p
            done += 1
            if dt_s > 0.0:
                print(f"  [{done:2d}/{len(jobs)}] {tag:17s} ({dt_s:5.1f}s) | {info}", flush=True)

    c1_sm = OUT_DIR / "c1_sm_1200.csv"
    c2_sm = OUT_DIR / "c2_sm_1200.csv"
    c1_hm = OUT_DIR / "c1_hm_1200.csv"
    c2_hm = OUT_DIR / "c2_hm_1200.csv"
    merge_ensemble([out[f"c1_sm_1200_{k:02d}"] for k in range(4)], c1_sm)
    merge_ensemble([out[f"c2_sm_1200_{k:02d}"] for k in range(4)], c2_sm)
    merge_ensemble([out[f"c1_hm_1200_{k:02d}"] for k in range(4)], c1_hm)
    merge_ensemble([out[f"c2_hm_1200_{k:02d}"] for k in range(4)], c2_hm)
    print(f"Completed & merged 1.2 AGeV SM + HM ensembles in {time.time() - t0:.1f}s!", flush=True)
    return {"c1_sm": c1_sm, "c2_sm": c2_sm, "c1_hm": c1_hm, "c2_hm": c2_hm}


def plot_benchmark(merged, fopi):
    plt.rcParams.update({
        "font.size": 9.4, "axes.titlesize": 10.2, "axes.labelsize": 9.4,
        "legend.fontsize": 7.1, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.6, 10.4), dpi=185)
    fig.suptitle(
        "FOPI Au+Au @ 1.2 $A$ GeV: SM ($K_0 = 215\\;\\mathrm{MeV}$) vs. HM ($K_0 = 380\\;\\mathrm{MeV}$) in Grand-Canonical HCGF ($\\Delta x = 1.0\\;\\mathrm{fm}$, 3D Coulomb, $\\Delta/\\pi$)",
        fontsize=11.3, fontweight="bold", y=0.98
    )

    # (a) Freeze-out cell temperature P(T_fr): 1.2 AGeV SM vs HM (plus 0.25 AGeV SM reference)
    ax = axes[0, 0]
    T_025, w_025 = load_T_hist([ROOT / "output/highstat_gc_dx1fm/runs/c1_gc_dx1_00.cells.csv"])
    T_sm, w_sm = load_T_hist([RUNS_DIR / f"c1_sm_1200_{k:02d}.cells.csv" for k in range(2)])
    T_hm, w_hm = load_T_hist([RUNS_DIR / f"c1_hm_1200_{k:02d}.cells.csv" for k in range(2)])
    bins = np.linspace(2.0, 68.0, 44)
    if len(T_025) > 0:
        m025 = np.average(T_025, weights=w_025)
        ax.hist(T_025, bins=bins, weights=w_025, density=True, color="#bdc3c7", alpha=0.50,
                label=f"$0.25A\\;\\mathrm{{GeV}}$ SM: $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {m025:.1f}$ MeV")
        ax.axvline(m025, color="#7f8c8d", ls=":", lw=1.6)
    if len(T_sm) > 0:
        msm = np.average(T_sm, weights=w_sm)
        ax.hist(T_sm, bins=bins, weights=w_sm, density=True, color="#c0392b", alpha=0.62,
                label=f"$1.2A\\;\\mathrm{{GeV}}$ SM ($K_0=215$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {msm:.1f}$ MeV")
        ax.axvline(msm, color="#922b21", ls="--", lw=2.0)
    if len(T_hm) > 0:
        mhm = np.average(T_hm, weights=w_hm)
        ax.hist(T_hm, bins=bins, weights=w_hm, density=True, histtype="step", color="#1f618d", lw=2.2,
                label=f"$1.2A\\;\\mathrm{{GeV}}$ HM ($K_0=380$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {mhm:.1f}$ MeV")
        ax.axvline(mhm, color="#1f618d", ls="-.", lw=1.9)
    ax.set_xlabel("Cell Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.set_ylabel("Baryon-Weighted Probability Density")
    ax.set_title("(a) Freeze-Out Temperature $P(T_{\\mathrm{fr}})$ at $n_{\\mathrm{fr}} = \\rho_0 / 8$")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.0)

    # (b) Central (b0 < 0.15) log-scale baryon + cluster + pion multiplicities vs FOPI 1.2 AGeV
    ax = axes[0, 1]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li", "pi-", "pi0", "pi+"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$",
            "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$", "$\\pi^-$", "$\\pi^0$", "$\\pi^+$"]
    xp = np.arange(len(sps)); w = 0.34
    y_sm = load_yields(merged["c1_sm"])
    y_hm = load_yields(merged["c1_hm"])
    ax.bar(xp - 0.5 * w, [max(1e-4, y_sm[s][0]) for s in sps], yerr=[y_sm[s][1] for s in sps], width=w,
           color="#c0392b", capsize=2, label="SM Grand-Canonical ($K_0 = 215$ MeV)")
    ax.bar(xp + 0.5 * w, [max(1e-4, y_hm[s][0]) for s in sps], yerr=[y_hm[s][1] for s in sps], width=w,
           color="#1f618d", capsize=2, label="HM Grand-Canonical ($K_0 = 380$ MeV)")
    exp_idx = [i for i, s in enumerate(sps) if s in fopi["central_yields_b0_lt_015"]]
    ax.errorbar(xp[exp_idx], [fopi["central_yields_b0_lt_015"][sps[i]]["val"] for i in exp_idx],
                yerr=[fopi["central_yields_b0_lt_015"][sps[i]]["err"] for i in exp_idx],
                fmt="ko", ms=5.8, capsize=3, lw=1.6, zorder=8, label="FOPI Exp. $1.2A$ GeV ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 220.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (Log Scale)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=6.9)
    ax.set_title("(b) Central Cluster + Pion Yields ($b_0 < 0.15$, Log Scale)")

    # (c) Linear-scale central yield comparison on all 7 measured FOPI species (p, d, t, 3He, 4He, pi-, pi+)
    ax = axes[0, 2]
    f_sps = ["p", "d", "t", "3He", "4He", "pi-", "pi+"]
    f_lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$", "$\\pi^-$", "$\\pi^+$"]
    xp7 = np.arange(len(f_sps)); w7 = 0.26
    ax.bar(xp7 - w7, [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps],
           yerr=[fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps],
           width=w7, color="#2c3e50", capsize=2.5, label="FOPI Exp. ($1.2A$ GeV)")
    ax.bar(xp7,      [y_sm[s][0] for s in f_sps], yerr=[y_sm[s][1] for s in f_sps],
           width=w7, color="#c0392b", capsize=2.5, label="SM GC ($K_0 = 215$ MeV)")
    ax.bar(xp7 + w7, [y_hm[s][0] for s in f_sps], yerr=[y_hm[s][1] for s in f_sps],
           width=w7, color="#1f618d", capsize=2.5, label="HM GC ($K_0 = 380$ MeV)")
    ax.set_xticks(xp7); ax.set_xticklabels(f_lbls)
    ax.set_ylabel("Central Multiplicity / Event (Linear Scale)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(c) Linear Central Yields: Baryons, Clusters & Pions")

    # (d) Semicentral (0.25 < b0 < 0.45) Rapidity Spectra dN/dy0: baryons, clusters & pions
    ax = axes[1, 0]
    for sp, col, mult, lbl in [("p",   "#2c3e50", 0.35, "$p \\times 0.35$"),
                               ("d",   "#1f77b4", 1.0,  "$d$"),
                               ("t",   "#27ae60", 1.0,  "$t$"),
                               ("4He", "#8e44ad", 2.0,  "$^4\\mathrm{He} \\times 2$"),
                               ("pi-", "#e67e22", 1.0,  "$\\pi^-$"),
                               ("pi+", "#d35400", 1.0,  "$\\pi^+$")]:
        ys, ds, es, _, _, _ = load_rap(merged["c2_sm"], sp)
        yh, dh, _,  _, _, _ = load_rap(merged["c2_hm"], sp)
        ax.plot(ys, ds * mult, "-",  color=col, lw=2.1, label=f"{lbl} SM")
        ax.fill_between(ys, (ds - es) * mult, (ds + es) * mult, color=col, alpha=0.12)
        ax.plot(yh, dh * mult, "--", color=col, lw=1.6, alpha=0.88, label=f"{lbl} HM")
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ ($0.25 < b_0 < 0.45$)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.3)
    ax.set_title("(d) Semi-Central $dN/dy_0$: SM (solid) vs. HM (dashed)")

    # (e) Directed Flow v1(y0) (u_t0 > 0.4, FOPI NPA 876 Fig. 7): SM vs HM vs FOPI Data
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p",   "#2c3e50", "o", "$p$"),
                             ("d",   "#1f77b4", "s", "$d$"),
                             ("A3",  "#d62728", "^", "$A=3$"),
                             ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        ys, _, _, v1s, e1s, ns4 = load_rap(merged["c2_sm"], sp)
        yh, _, _, v1h, _,   nh4 = load_rap(merged["c2_hm"], sp)
        ss = (np.abs(ys) <= 1.08) & (ns4 >= 10)
        sh = (np.abs(yh) <= 1.08) & (nh4 >= 10)
        ax.plot(ys[ss], v1s[ss], "-",  color=col, lw=2.2, label=f"{lbl} SM")
        ax.fill_between(ys[ss], v1s[ss] - e1s[ss], v1s[ss] + e1s[ss], color=col, alpha=0.13)
        ax.plot(yh[sh], v1h[sh], "--", color=col, lw=1.7, alpha=0.88, label=f"{lbl} HM")
        if sp in fopi["v1_y0_ut0_gt_04"]:
            ep = fopi["v1_y0_ut0_gt_04"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.86, 0.86)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.4$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=3, fontsize=6.2)
    ax.set_title("(e) Directed Flow $v_1(y_0)$ ($u_{t0} > 0.4$): SM vs. HM vs. FOPI")

    # (f) Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4, FOPI NPA 876 Fig. 26): SM vs HM vs FOPI Data
    ax = axes[1, 2]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"),
                             ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$")]:
        us, v2s, e2s, ns_u = load_ut0(merged["c2_sm"], sp)
        uh, v2h, _,   nh_u = load_ut0(merged["c2_hm"], sp)
        ss = (us >= 0.25) & (us <= 1.98) & (ns_u >= 65)
        sh = (uh >= 0.25) & (uh <= 1.98) & (nh_u >= 65)
        ax.plot(us[ss], v2s[ss], "-",  color=col, lw=2.2, label=f"{lbl} SM")
        ax.fill_between(us[ss], v2s[ss] - e2s[ss], v2s[ss] + e2s[ss], color=col, alpha=0.13)
        ax.plot(uh[sh], v2h[sh], "--", color=col, lw=1.7, alpha=0.88, label=f"{lbl} HM")
        if sp in fopi["v2_ut0_y0_lt_04"]:
            eu = fopi["v2_ut0_y0_lt_04"][sp]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.22, 2.02); ax.set_ylim(-0.20, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=3, fontsize=6.3)
    ax.set_title("(f) Squeeze-Out $v_2(u_{t0})$ ($|y_0| < 0.4$): SM vs. HM vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_OUT, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_OUT}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_ensembles()
    plot_benchmark(merged, fopi)
    for nm, p in [("SM GC (K0=215 MeV, 1.2 AGeV)", merged["c1_sm"]),
                  ("HM GC (K0=380 MeV, 1.2 AGeV)", merged["c1_hm"])]:
        yd = load_yields(p)
        print(f"  {nm}: p={yd['p'][0]:6.2f}+-{yd['p'][1]:4.2f} | n={yd['n'][0]:6.2f} | "
              f"d={yd['d'][0]:5.2f}+-{yd['d'][1]:4.2f} | t={yd['t'][0]:5.2f}+-{yd['t'][1]:4.2f} | "
              f"3He={yd['3He'][0]:5.2f}+-{yd['3He'][1]:4.2f} | 4He={yd['4He'][0]:5.2f}+-{yd['4He'][1]:4.2f} | "
              f"pi-={yd['pi-'][0]:5.2f}+-{yd['pi-'][1]:4.2f} | pi0={yd['pi0'][0]:5.2f} | pi+={yd['pi+'][0]:5.2f}+-{yd['pi+'][1]:4.2f}")


if __name__ == "__main__":
    main()
