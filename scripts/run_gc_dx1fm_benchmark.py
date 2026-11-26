#!/usr/bin/env python3
"""
High-statistics FOPI Au+Au @ 250 A MeV benchmark for Grand-Canonical Poisson
HCGF sampling (--sampling gc) on dx = 1.00 fm (68^3) cells with:
  1. 3D Poisson Coulomb potential + Vlasov forces (--coulomb on),
  2. Exact Skyrme + Lorentzian MD potential in T^00,
  3. 6D phase-space Pauli blocking + downward n_fr = rho_0 / 8 crossing,
  4. Pratt-Torrieri det(Lambda) = 1 local momentum anisotropy.
"""

import csv
import glob
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
FOPI_JSON = ROOT / "scripts" / "fopi_exp_auau250.json"
OUT_DIR = ROOT / "output" / "highstat_gc_dx1fm"
RUNS_DIR = OUT_DIR / "runs"
PNG_OUT = ROOT / "assets" / "fopi_gc_dx1fm_benchmark.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160


def run_job(job):
    tag, b_fm, ntest, eos_flag, sampling_mode, save_cells, seed = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    cells_csv = RUNS_DIR / f"{tag}.cells.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and (not save_cells or cells_csv.exists()):
        return tag, 0.0, "cached", fl_csv

    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "68.0", "--grid", "68",
        "--steps", "280", "--dt", "0.30", "--substeps", "2",
        "--eos", eos_flag, "--in-medium-ec", "on", "--pauli", "on", "--coulomb", "on",
        "--cluster-mode", "freezeout", "--backflow", "remelt",
        "--sampling", sampling_mode, "--anisotropy", "on",
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "2",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    if save_cells:
        cmd += ["--freezeout-cells-csv", str(cells_csv)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "done"), fl_csv


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
                    "v1_ut08", "v1_ut08_err", "v2_ut08", "v2_ut08_err", "n_ut08"])
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
            w.writerow([r0["species"], r0["bin"], r0["y"], r0["y0"],
                        f"{dndy_m:.6f}", f"{dndy0_m:.6f}", f"{dndy0_e:.6f}",
                        f"{v1_m:.6f}", f"{v1_e:.6f}", f"{v2_m:.6f}", f"{v2_e:.6f}", f"{n_st:.0f}",
                        f"{u8_v1_m:.6f}", f"{u8_v1_e:.6f}", f"{u8_v2_m:.6f}", f"{u8_v2_e:.6f}", f"{n_u8:.0f}"])

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
            np.array([float(r["v1_ut08"]) for r in rs]),
            np.array([float(r["v1_ut08_err"]) for r in rs]),
            np.array([float(r["n_ut08"]) for r in rs]))


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

    # SM (MD-Soft, K0=215 MeV, cached!) + HM (MD-Stiff, K0=380 MeV, paired seeds!) on dx = 1.00 fm (68^3)
    for k in range(4):
        b1 = math.sqrt(((k + 0.5) / 4.0) * (b1_hi * b1_hi))
        b2 = math.sqrt(b2_lo * b2_lo + ((k + 0.5) / 4.0) * (b2_hi * b2_hi - b2_lo * b2_lo))
        jobs.append((f"c1_gc_dx1_{k:02d}",    b1, 500, "md-soft",  "gc", k < 2, 981000 + k * 151))
        jobs.append((f"c2_gc_dx1_{k:02d}",    b2, 500, "md-soft",  "gc", k < 2, 982000 + k * 151))
        jobs.append((f"c1_gc_hm_dx1_{k:02d}", b1, 500, "md-stiff", "gc", k < 2, 981000 + k * 151))
        jobs.append((f"c2_gc_hm_dx1_{k:02d}", b2, 500, "md-stiff", "gc", k < 2, 982000 + k * 151))

    print(f"Running {len(jobs)} dx = 1.00 fm (68^3) SM vs HM Grand-Canonical jobs across 4 workers...", flush=True)
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

    c1_sm = OUT_DIR / "c1_gc_dx1.csv"
    c2_sm = OUT_DIR / "c2_gc_dx1.csv"
    c1_hm = OUT_DIR / "c1_gc_hm_dx1.csv"
    c2_hm = OUT_DIR / "c2_gc_hm_dx1.csv"
    merge_ensemble([out[f"c1_gc_dx1_{k:02d}"]    for k in range(4)], c1_sm)
    merge_ensemble([out[f"c2_gc_dx1_{k:02d}"]    for k in range(4)], c2_sm)
    merge_ensemble([out[f"c1_gc_hm_dx1_{k:02d}"] for k in range(4)], c1_hm)
    merge_ensemble([out[f"c2_gc_hm_dx1_{k:02d}"] for k in range(4)], c2_hm)

    c1_par = OUT_DIR / "c1_parcel_dx1.csv"
    merge_ensemble(sorted(Path(p) for p in glob.glob(str(ROOT / "output/highstat_dx1fm_v2/c1_dx1_0*.csv"))
                          if not p.endswith(".ut0.csv") and not p.endswith(".yields.csv") and not p.endswith(".cells.csv")), c1_par)
    c1_dx24 = OUT_DIR / "c1_coul_dx24.csv"
    merge_ensemble(sorted(Path(p) for p in glob.glob(str(ROOT / "output/highstat_coulomb_c1/c1_coul_0*.csv"))
                          if not p.endswith(".ut0.csv") and not p.endswith(".yields.csv") and not p.endswith(".cells.csv")), c1_dx24)

    print(f"Completed & merged SM + HM ensembles in {time.time() - t0:.1f}s!", flush=True)
    return {
        "c1_sm": c1_sm, "c2_sm": c2_sm,
        "c1_hm": c1_hm, "c2_hm": c2_hm,
        "c1_parcel": c1_par, "c1_dx24": c1_dx24,
    }


def plot_benchmark(merged, fopi):
    plt.rcParams.update({
        "font.size": 9.4, "axes.titlesize": 10.3, "axes.labelsize": 9.4,
        "legend.fontsize": 7.2, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.3), dpi=185)
    fig.suptitle(
        "SM ($K_0 = 215\\;\\mathrm{MeV}$) vs. HM ($K_0 = 380\\;\\mathrm{MeV}$) EOS in Grand-Canonical HCGF ($\\Delta x = 1.0\\;\\mathrm{fm}$, 3D Coulomb, Pauli) — FOPI Au+Au @ 250 $A$ MeV",
        fontsize=11.3, fontweight="bold", y=0.98
    )

    # (a) Freeze-out cell temperature distribution: SM (dx=1 fm) vs HM (dx=1 fm) vs Coarse (dx=2.4 fm)
    ax = axes[0, 0]
    T_24, w_24 = load_T_hist([ROOT / "output/highstat_coulomb_c1/c1_coul_00.cells.csv"])
    T_sm, w_sm = load_T_hist([RUNS_DIR / f"c1_gc_dx1_{k:02d}.cells.csv" for k in range(2)])
    T_hm, w_hm = load_T_hist([RUNS_DIR / f"c1_gc_hm_dx1_{k:02d}.cells.csv" for k in range(2)])
    bins = np.linspace(1.5, 34.0, 38)
    if len(T_24) > 0:
        m24 = np.average(T_24, weights=w_24)
        ax.hist(T_24, bins=bins, weights=w_24, density=True, color="#bdc3c7", alpha=0.52,
                label=f"SM $\\Delta x=2.4\\;\\mathrm{{fm}}$: $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {m24:.2f}$ MeV")
        ax.axvline(m24, color="#7f8c8d", ls=":", lw=1.7)
    if len(T_sm) > 0:
        msm = np.average(T_sm, weights=w_sm)
        ax.hist(T_sm, bins=bins, weights=w_sm, density=True, color="#c0392b", alpha=0.62,
                label=f"SM ($K_0=215\\;\\mathrm{{MeV}}, 1\\;\\mathrm{{fm}}$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {msm:.2f}$ MeV")
        ax.axvline(msm, color="#922b21", ls="--", lw=2.0)
    if len(T_hm) > 0:
        mhm = np.average(T_hm, weights=w_hm)
        ax.hist(T_hm, bins=bins, weights=w_hm, density=True, histtype="step", color="#1f618d", lw=2.2,
                label=f"HM ($K_0=380\\;\\mathrm{{MeV}}, 1\\;\\mathrm{{fm}}$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {mhm:.2f}$ MeV")
        ax.axvline(mhm, color="#1f618d", ls="-.", lw=1.9)
    ax.set_xlabel("Cell Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.set_ylabel("Baryon-Weighted Probability Density")
    ax.set_title("(a) Freeze-Out Temperature $P(T_{\\mathrm{fr}})$: SM vs. HM ($\\Delta x = 1.0\\;\\mathrm{fm}$)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.1)

    # (b) Central (b0 < 0.15) log-scale multiplicities across A = 1..6: SM vs HM vs Parcel vs FOPI
    ax = axes[0, 1]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$", "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$"]
    xp = np.arange(len(sps)); w = 0.24
    y_pa = load_yields(merged["c1_parcel"])
    y_sm = load_yields(merged["c1_sm"])
    y_hm = load_yields(merged["c1_hm"])
    ax.bar(xp - w, [y_pa[s][0] for s in sps], yerr=[y_pa[s][1] for s in sps], width=w,
           color="#95a5a6", capsize=2, label="SM Parcel ($\\Delta x = 1.0\\;\\mathrm{fm}$)")
    ax.bar(xp,     [y_sm[s][0] for s in sps], yerr=[y_sm[s][1] for s in sps], width=w,
           color="#c0392b", capsize=2, label="SM Grand-Canonical ($K_0 = 215$ MeV)")
    ax.bar(xp + w, [y_hm[s][0] for s in sps], yerr=[y_hm[s][1] for s in sps], width=w,
           color="#1f618d", capsize=2, label="HM Grand-Canonical ($K_0 = 380$ MeV)")
    f_sps = ["p", "d", "t", "3He", "4He"]
    ax.errorbar(np.arange(5) + 0.5 * w, [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps],
                yerr=[fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps],
                fmt="ko", ms=5.8, capsize=3, lw=1.6, zorder=8, label="FOPI Exp. ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(4e-2, 160.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (Log Scale)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=6.9)
    ax.set_title("(b) Central Isotope Yields ($A = 1 \\dots 6$): SM vs. HM vs. FOPI")

    # (c) Linear-scale comparison on FOPI light charged species (p, d, t, 3He, 4He)
    ax = axes[0, 2]
    xp5 = np.arange(5); w5 = 0.20
    f_vals = [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps]
    f_errs = [fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps]
    ax.bar(xp5 - 1.5 * w5, f_vals, yerr=f_errs, width=w5, color="#2c3e50", capsize=2.5, label="FOPI Exp. ($0.25A$ GeV)")
    ax.bar(xp5 - 0.5 * w5, [y_sm[s][0] for s in f_sps], yerr=[y_sm[s][1] for s in f_sps], width=w5,
           color="#c0392b", capsize=2.5, label="SM GC ($K_0 = 215$ MeV, $1\\;\\mathrm{fm}$)")
    ax.bar(xp5 + 0.5 * w5, [y_hm[s][0] for s in f_sps], yerr=[y_hm[s][1] for s in f_sps], width=w5,
           color="#1f618d", capsize=2.5, label="HM GC ($K_0 = 380$ MeV, $1\\;\\mathrm{fm}$)")
    ax.bar(xp5 + 1.5 * w5, [y_pa[s][0] for s in f_sps], yerr=[y_pa[s][1] for s in f_sps], width=w5,
           color="#95a5a6", capsize=2.5, label="SM Parcel ($1\\;\\mathrm{fm}$)")
    ax.set_xticks(xp5); ax.set_xticklabels(lbls[:5])
    ax.set_ylabel("Central Multiplicity / Event (Linear Scale)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(c) Linear Yield Comparison: SM ($215$) vs. HM ($380$ MeV)")

    # (d) Semicentral (0.25 < b0 < 0.45) Rapidity Spectra dN/dy0: SM (solid) vs HM (dashed)
    ax = axes[1, 0]
    for sp, col, mult, lbl in [("p", "#2c3e50", 0.5, "$p \\times 0.5$"),
                               ("d", "#1f77b4", 1.0, "$d$"),
                               ("t", "#27ae60", 1.0, "$t$"),
                               ("4He", "#c0392b", 1.0, "$^4\\mathrm{He}$"),
                               ("6Li", "#e67e22", 20.0, "$^6\\mathrm{Li}\\times 20$")]:
        ys, ds, es, _, _, _ = load_rap(merged["c2_sm"], sp)
        yh, dh, _,  _, _, _ = load_rap(merged["c2_hm"], sp)
        ax.plot(ys, ds * mult, "-",  color=col, lw=2.2, label=f"{lbl} SM ($215$)")
        ax.fill_between(ys, (ds - es) * mult, (ds + es) * mult, color=col, alpha=0.13)
        ax.plot(yh, dh * mult, "--", color=col, lw=1.7, alpha=0.88, label=f"{lbl} HM ($380$)")
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ ($0.25 < b_0 < 0.45$)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.5)
    ax.set_title("(d) Semi-Central $dN/dy_0$: SM (solid) vs. HM (dashed)")

    # (e) Directed Flow v1(y0) (u_t0 > 0.8): SM (solid) vs HM (dashed) vs FOPI Data
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"), ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        ys, _, _, v1s, e1s, ns8 = load_rap(merged["c2_sm"], sp)
        yh, _, _, v1h, _,   nh8 = load_rap(merged["c2_hm"], sp)
        ss = (np.abs(ys) <= 1.08) & (ns8 >= 12)
        sh = (np.abs(yh) <= 1.08) & (nh8 >= 12)
        ax.plot(ys[ss], v1s[ss], "-",  color=col, lw=2.2, label=f"{lbl} SM")
        ax.fill_between(ys[ss], v1s[ss] - e1s[ss], v1s[ss] + e1s[ss], color=col, alpha=0.13)
        ax.plot(yh[sh], v1h[sh], "--", color=col, lw=1.7, alpha=0.88, label=f"{lbl} HM")
        if sp in fopi["v1_y0_ut0_gt_08"]:
            ep = fopi["v1_y0_ut0_gt_08"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.76, 0.76)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.8$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=3, fontsize=6.3)
    ax.set_title("(e) Directed Flow $v_1(y_0)$: SM ($215$) vs. HM ($380$) vs. FOPI")

    # (f) Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4): SM (solid) vs HM (dashed) vs FOPI Data
    ax = axes[1, 2]
    for sp, ekey, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                   ("d", "d", "#1f77b4", "s", "$d$"),
                                   ("t", "t", "#d62728", "^", "$t$"),
                                   ("4He", "4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        us, v2s, e2s, ns_u = load_ut0(merged["c2_sm"], sp)
        uh, v2h, _,   nh_u = load_ut0(merged["c2_hm"], sp)
        ss = (us >= 0.45) & (us <= 1.85) & (ns_u >= 25)
        sh = (uh >= 0.45) & (uh <= 1.85) & (nh_u >= 25)
        ax.plot(us[ss], v2s[ss], "-",  color=col, lw=2.2, label=f"{lbl} SM")
        ax.fill_between(us[ss], v2s[ss] - e2s[ss], v2s[ss] + e2s[ss], color=col, alpha=0.13)
        ax.plot(uh[sh], v2h[sh], "--", color=col, lw=1.7, alpha=0.88, label=f"{lbl} HM")
        if ekey in fopi["v2_ut0_y0_lt_04"]:
            eu = fopi["v2_ut0_y0_lt_04"][ekey]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.40, 1.88); ax.set_ylim(-0.24, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=3, fontsize=6.1)
    ax.set_title("(f) Squeeze-Out $v_2(u_{t0})$: SM ($215$) vs. HM ($380$) vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_OUT, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_OUT}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_ensembles()
    plot_benchmark(merged, fopi)
    for nm, p in [("SM Grand-Canonical (K0=215, 1fm)", merged["c1_sm"]),
                  ("HM Grand-Canonical (K0=380, 1fm)", merged["c1_hm"]),
                  ("SM Parcel Sampled  (K0=215, 1fm)", merged["c1_parcel"]),
                  ("SM Coarse Grid    (K0=215,2.4fm)", merged["c1_dx24"])]:
        yd = load_yields(p)
        print(f"  {nm}: p={yd['p'][0]:6.2f}+-{yd['p'][1]:4.2f} | n={yd['n'][0]:6.2f} | "
              f"d={yd['d'][0]:5.2f}+-{yd['d'][1]:4.2f} | t={yd['t'][0]:5.2f}+-{yd['t'][1]:4.2f} | "
              f"3He={yd['3He'][0]:5.2f}+-{yd['3He'][1]:4.2f} | 4He={yd['4He'][0]:5.2f}+-{yd['4He'][1]:4.2f} | "
              f"6He={yd['6He'][0]:.3f} | 6Li={yd['6Li'][0]:.3f}")


if __name__ == "__main__":
    main()
