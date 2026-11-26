#!/usr/bin/env python3
"""
High-statistics FOPI Au+Au @ 250 A MeV benchmark with:
  1. 6D phase-space Pauli blocking (1 - f_3)(1 - f_4) + UrQMD cold-nucleus stability,
  2. Post-compression downward n_B = n_fr = rho_0 / 8 hypersurface crossing (dn_B/dt <= 0),
  3. Interacting Eckart energy e_tr = e_kin* + V_pot/n_B matched to Quantum Fermi/Bose INCG EoS,
  4. Pratt-Torrieri local momentum anisotropy (p*_i = Lambda_ij q_j, det Lambda = 1).
"""

import csv
import json
import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIN_FP32 = ROOT / "build" / "cluster_transport_fp32"
FOPI_JSON = ROOT / "scripts" / "fopi_exp_auau250.json"
OLD_DIR = ROOT / "output" / "highstat_aniso"
OUT_DIR = ROOT / "output" / "highstat_pauli_hcgf"
RUNS_DIR = OUT_DIR / "runs"
PNG_OUT = ROOT / "assets" / "fopi_pauli_hcgf_benchmark.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160


def run_job(job):
    tag, b_fm, ntest, cmode, aniso_flag, save_cells, seed = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    cells_csv = RUNS_DIR / f"{tag}.cells.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and (not save_cells or cells_csv.exists()):
        return tag, 0.0, "cached", fl_csv

    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "72.0", "--grid", "30",
        "--steps", "280", "--dt", "0.30", "--substeps", "2",
        "--eos", "md-soft", "--in-medium-ec", "on", "--pauli", "on",
        "--cluster-mode", cmode, "--backflow", "remelt",
        "--anisotropy", aniso_flag,
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "2",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    if save_cells:
        cmd += ["--freezeout-cells-csv", str(cells_csv)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "micro-pauli"), fl_csv


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
        for idx in range(len(rap_tables[0])):
            r0 = rap_tables[0][idx]
            dn_vals = [float(t[idx]["dN_dy0"]) for t in rap_tables]
            dn_dy = sum(float(t[idx]["dN_dy"]) for t in rap_tables) / m_runs
            dn_dy0 = sum(dn_vals) / m_runs
            dn_err = np.std(dn_vals, ddof=1) / math.sqrt(m_runs) if m_runs > 1 else 0.0
            v1, e1, n_st = pool_mean_err([(float(t[idx]["v1"]), float(t[idx]["v1_err"]), float(t[idx]["n_stat"])) for t in rap_tables])
            v2, e2, _    = pool_mean_err([(float(t[idx]["v2"]), float(t[idx]["v2_err"]), float(t[idx]["n_stat"])) for t in rap_tables])
            v1u, e1u, nu = pool_mean_err([(float(t[idx]["v1_ut08"]), float(t[idx]["v1_ut08_err"]), float(t[idx]["n_ut08"])) for t in rap_tables])
            v2u, e2u, _  = pool_mean_err([(float(t[idx]["v2_ut08"]), float(t[idx]["v2_ut08_err"]), float(t[idx]["n_ut08"])) for t in rap_tables])
            w.writerow([r0["species"], r0["bin"], r0["y"], r0["y0"],
                        f"{dn_dy:.6g}", f"{dn_dy0:.6g}", f"{dn_err:.6g}",
                        f"{v1:.6g}", f"{e1:.6g}", f"{v2:.6g}", f"{e2:.6g}", f"{n_st:.0f}",
                        f"{v1u:.6g}", f"{e1u:.6g}", f"{v2u:.6g}", f"{e2u:.6g}", f"{nu:.0f}"])

    with open(str(dst_csv) + ".ut0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "bin", "ut0", "v2", "v2_err", "n_stat"])
        for idx in range(len(ut0_tables[0])):
            r0 = ut0_tables[0][idx]
            v2, e2, n_u = pool_mean_err([(float(t[idx]["v2"]), float(t[idx]["v2_err"]), float(t[idx]["n_stat"])) for t in ut0_tables])
            w.writerow([r0["species"], r0["bin"], r0["ut0"], f"{v2:.6g}", f"{e2:.6g}", f"{n_u:.0f}"])

    with open(str(dst_csv) + ".yields.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "yield_per_event", "yield_err"])
        for idx in range(len(yld_tables[0])):
            sp = yld_tables[0][idx]["species"]
            ys = [float(t[idx]["yield_per_event"]) for t in yld_tables]
            yv = sum(ys) / m_runs
            ye = np.std(ys, ddof=1) / math.sqrt(m_runs) if m_runs > 1 else 0.0
            w.writerow([sp, f"{yv:.6g}", f"{ye:.6g}"])


def load_rap(csv_p, sp):
    y0, dn, dne, v1, v1e, nu8 = [], [], [], [], [], []
    for r in csv.DictReader(open(csv_p)):
        if r["species"] == sp:
            y0.append(float(r["y0"])); dn.append(float(r["dN_dy0"])); dne.append(float(r["dN_dy0_err"]))
            v1.append(float(r["v1_ut08"])); v1e.append(float(r["v1_ut08_err"])); nu8.append(float(r["n_ut08"]))
    return tuple(np.array(a) for a in (y0, dn, dne, v1, v1e, nu8))


def load_ut0(csv_p, sp):
    u, v2, v2e, nst = [], [], [], []
    for r in csv.DictReader(open(str(csv_p) + ".ut0.csv")):
        if r["species"] == sp:
            u.append(float(r["ut0"])); v2.append(float(r["v2"])); v2e.append(float(r["v2_err"])); nst.append(float(r["n_stat"]))
    return tuple(np.array(a) for a in (u, v2, v2e, nst))


def load_yields(csv_p):
    return {r["species"]: (float(r["yield_per_event"]), float(r["yield_err"]))
            for r in csv.DictReader(open(str(csv_p) + ".yields.csv"))}


def load_T_hist(cells_paths):
    Ts, ws = [], []
    for p in cells_paths:
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            Ts.append(float(r["T_mev"]))
            ws.append(float(r["baryons_converted"]))
    return np.array(Ts), np.array(ws)


def run_ensembles():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Semicentral 0.25 < b0 < 0.45: 10 b*db points x N_test=400 = 4,000 events each
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        b_fm = B_MAX_AUAU * math.sqrt(b0_sq)
        seed = 410000 + k * 7919
        jobs.append((f"c2_p_pt_{k:02d}",  b_fm, 400, "freezeout",   "on",  (k < 4), seed))
        jobs.append((f"c2_p_iso_{k:02d}", b_fm, 400, "freezeout",   "off", False,   seed))
        jobs.append((f"c2_p_mic_{k:02d}", b_fm, 400, "microscopic", "off", False,   seed))

    # Central b0 < 0.15: 5 b*db points x N_test=400 = 2,000 events each
    for k in range(5):
        b_fm = B_MAX_AUAU * 0.15 * math.sqrt((k + 0.5) / 5.0)
        seed = 520000 + k * 7919
        jobs.append((f"c1_p_pt_{k:02d}",  b_fm, 400, "freezeout",   "on",  (k < 2), seed))
        jobs.append((f"c1_p_mic_{k:02d}", b_fm, 400, "microscopic", "off", False,   seed))

    workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(jobs)} Pauli-blocked HCGF & Microscopic jobs ({workers} workers, 16,000 events)...", flush=True)
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            tag, dt_s, info, p = fut.result()
            out[tag] = p; done += 1
            if dt_s > 0.0:
                print(f"  [{done:2d}/{len(jobs)}] {tag:14s} ({dt_s:5.1f}s) | {info}", flush=True)

    merged = {}
    for key, nk in [("c2_p_pt", 10), ("c2_p_iso", 10), ("c2_p_mic", 10),
                    ("c1_p_pt", 5),  ("c1_p_mic", 5)]:
        dst = OUT_DIR / f"{key}.csv"
        merge_ensemble([out[f"{key}_{k:02d}"] for k in range(nk)], dst)
        merged[key] = dst
    print(f"Merged all Pauli-blocked ensembles in {time.time() - t0:.1f}s!", flush=True)
    return merged


def plot_benchmark(merged, fopi):
    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.2, "axes.labelsize": 9.3,
        "legend.fontsize": 7.3, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), dpi=185)
    fig.suptitle(
        "6D Phase-Space Pauli Blocking $(1-f_3)(1-f_4)$ + Downward $n_{\\mathrm{fr}}=\\rho_0/8$ Crossing + Interacting Quantum INCG + Pratt–Torrieri HCGF — FOPI Au+Au @ 250 $A$ MeV",
        fontsize=11.4, fontweight="bold", y=0.98
    )

    # (a) Freeze-out cell temperature distribution: Unblocked/Bare-E vs Pauli + Interacting INCG
    ax = axes[0, 0]
    T_old, w_old = load_T_hist([OLD_DIR / "runs" / f"c2_pt8_a6_{k:02d}.cells.csv" for k in range(4)])
    T_new, w_new = load_T_hist([RUNS_DIR / f"c2_p_pt_{k:02d}.cells.csv" for k in range(4)])
    bins = np.linspace(2.0, 48.0, 40)
    if len(T_old) > 0:
        ax.hist(T_old, bins=bins, weights=w_old, density=True, color="#95a5a6", alpha=0.60,
                label=f"Before (No Pauli, bare $E$): $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {np.average(T_old, weights=w_old):.1f}$ MeV")
        ax.axvline(np.average(T_old, weights=w_old), color="#7f8c8d", ls=":", lw=1.8)
    if len(T_new) > 0:
        ax.hist(T_new, bins=bins, weights=w_new, density=True, color="#c0392b", alpha=0.72,
                label=f"Pauli + $V_{{\\mathrm{{pot}}}}$ + $dn_B/dt\\leq 0$: $\\langle T_{{\\mathrm{{fr}}}}\\rangle = {np.average(T_new, weights=w_new):.1f}$ MeV")
        ax.axvline(np.average(T_new, weights=w_new), color="#922b21", ls="--", lw=2.0)
    ax.set_xlabel("Cell Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.set_ylabel("Baryon-Weighted Probability Density")
    ax.set_title("(a) Freeze-Out Cooling from Pauli Blocking & Interacting $e_{\\mathrm{tr}}$")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.2)

    # (b) Central Au+Au (b0 < 0.15) isotope yields: Before vs Pauli-HCGF vs Pauli-Micro vs FOPI
    ax = axes[0, 1]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$", "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$"]
    xp = np.arange(len(sps)); w = 0.22
    y_old = load_yields(OLD_DIR / "c1_pt8_a6.csv")
    y_new = load_yields(merged["c1_p_pt"])
    y_mic = load_yields(merged["c1_p_mic"])
    ax.bar(xp - w, [y_old[s][0] for s in sps], yerr=[y_old[s][1] for s in sps], width=w, color="#95a5a6", capsize=2, label="Unblocked HCGF ($\\langle T_{\\mathrm{fr}}\\rangle\\approx 22$ MeV)")
    ax.bar(xp,     [y_new[s][0] for s in sps], yerr=[y_new[s][1] for s in sps], width=w, color="#c0392b", capsize=2, label="Pauli + Interacting HCGF ($\\rho_0/8$)")
    ax.bar(xp + w, [y_mic[s][0] for s in sps], yerr=[y_mic[s][1] for s in sps], width=w, color="#1f77b4", capsize=2, label="Pauli Microscopic BUU ($A\\leq 6$)")
    f_sps = ["p", "d", "t", "3He", "4He"]
    ax.errorbar(np.arange(5), [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps],
                yerr=[fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps],
                fmt="ko", ms=5.6, capsize=3, lw=1.5, zorder=7, label="FOPI ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(3e-3, 180.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=6.9)
    ax.set_title("(b) Central Isotope Yields ($A = 1 \\dots 6$) vs. FOPI Data")

    # (c) Cluster enhancement ratio Y(Pauli + Interacting HCGF) / Y(Unblocked HCGF)
    ax = axes[0, 2]
    rat = [y_new[s][0] / max(y_old[s][0], 1e-9) for s in sps]
    bars = ax.bar(xp, rat, width=0.52, color=["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad", "#c0392b", "#16a085", "#e67e22"])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.05, f"{h:.2f}x", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.axhline(1.0, color="#444", ls="--", lw=1.1)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylim(0.0, max(rat) * 1.22)
    ax.set_ylabel("Yield Ratio $Y(\\mathrm{Pauli+INCG}) \\;/\\; Y(\\mathrm{Unblocked})$")
    ax.grid(alpha=0.25)
    ax.set_title("(c) Heavy-Cluster Boost from Lower $T_{\\mathrm{fr}}$ ($^4\\mathrm{He}, ^6\\mathrm{He}, ^6\\mathrm{Li}$)")

    # (d) Semicentral (0.25 < b0 < 0.45) Rapidity Spectra dN/dy0 up to 6He, 6Li
    ax = axes[1, 0]
    for sp, col, mult, lbl in [("d", "#1f77b4", 1.0, "$d$"), ("t", "#27ae60", 1.0, "$t$"),
                               ("3He", "#8e44ad", 1.0, "$^3\\mathrm{He}$"),
                               ("4He", "#d62728", 3.0, "$^4\\mathrm{He}\\times 3$"),
                               ("6Li", "#e67e22", 40.0, "$^6\\mathrm{Li}\\times 40$")]:
        yp_r, dp_r, dpe, _, _, _ = load_rap(merged["c2_p_pt"], sp)
        ym_r, dm_r, _, _, _, _   = load_rap(merged["c2_p_mic"], sp)
        ax.plot(yp_r, dp_r * mult, "-",  color=col, lw=2.2, label=f"{lbl} Pauli-HCGF")
        ax.fill_between(yp_r, (dp_r - dpe) * mult, (dp_r + dpe) * mult, color=col, alpha=0.14)
        ax.plot(ym_r, dm_r * mult, "--", color=col, lw=1.6, alpha=0.80, label=f"{lbl} Pauli-Micro")
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ (4,000 Events)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.5)
    ax.set_title("(d) Rapidity Spectra $dN/dy_0$ ($0.25 < b_0 < 0.45$)")

    # (e) Directed Flow v1(y0) (u_t0 > 0.8) vs FOPI Data
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"), ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        yp_r, _, _, v1p, e1p, np8 = load_rap(merged["c2_p_pt"], sp)
        ym_r, _, _, v1m, _, nm8   = load_rap(merged["c2_p_mic"], sp)
        sp_m = (np.abs(yp_r) <= 1.08) & (np8 >= 15)
        sm_m = (np.abs(ym_r) <= 1.08) & (nm8 >= 15)
        ax.plot(yp_r[sp_m], v1p[sp_m], "-",  color=col, lw=2.2, label=f"{lbl} Pauli-HCGF")
        ax.fill_between(yp_r[sp_m], v1p[sp_m] - e1p[sp_m], v1p[sp_m] + e1p[sp_m], color=col, alpha=0.14)
        ax.plot(ym_r[sm_m], v1m[sm_m], "--", color=col, lw=1.6, alpha=0.80, label=f"{lbl} Pauli-Micro")
        if sp in fopi["v1_y0_ut0_gt_08"]:
            ep = fopi["v1_y0_ut0_gt_08"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.8$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=3, fontsize=6.4)
    ax.set_title("(e) Directed Flow $v_1(y_0)$: Pauli-HCGF vs. Micro vs. FOPI")

    # (f) Elliptic Squeeze-Out v2(u_t0): PT-Aniso vs Iso vs Micro vs FOPI
    ax = axes[1, 2]
    for sp, ekey, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                   ("d", "d", "#1f77b4", "s", "$d$"),
                                   ("A3", "A3", "#d62728", "^", "$A=3$"),
                                   ("4He", "4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        up_a, v2p, e2p, np_u = load_ut0(merged["c2_p_pt"], sp)
        ui_a, v2i, _,   ni_u = load_ut0(merged["c2_p_iso"], sp)
        um_a, v2m, _,   nm_u = load_ut0(merged["c2_p_mic"], sp)
        sp_u = (up_a >= 0.40) & (up_a <= 1.75) & (np_u >= 35)
        si_u = (ui_a >= 0.40) & (ui_a <= 1.75) & (ni_u >= 35)
        sm_u = (um_a >= 0.40) & (um_a <= 1.75) & (nm_u >= 35)
        ax.plot(up_a[sp_u], v2p[sp_u], "-",  color=col, lw=2.2, label=f"{lbl} PT-Aniso")
        ax.fill_between(up_a[sp_u], v2p[sp_u] - e2p[sp_u], v2p[sp_u] + e2p[sp_u], color=col, alpha=0.14)
        ax.plot(ui_a[si_u], v2i[si_u], ":",  color=col, lw=1.9, label=f"{lbl} Iso ($\\Lambda=I$)")
        ax.plot(um_a[sm_u], v2m[sm_u], "-.", color=col, lw=1.5, alpha=0.75, label=f"{lbl} Micro")
        if ekey in fopi["v2_ut0_y0_lt_04"]:
            eu = fopi["v2_ut0_y0_lt_04"][ekey]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.35, 1.80); ax.set_ylim(-0.175, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=3, fontsize=6.1)
    ax.set_title("(f) Squeeze-Out $v_2(u_{t0})$: PT-Aniso vs. Iso vs. Micro vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_OUT, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_OUT}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_ensembles()
    plot_benchmark(merged, fopi)
    for nm, p in [("Unblocked HCGF (old)  ", OLD_DIR / "c1_pt8_a6.csv"),
                  ("Pauli + PT-Aniso HCGF ", merged["c1_p_pt"]),
                  ("Pauli Microscopic BUU ", merged["c1_p_mic"])]:
        yd = load_yields(p)
        print(f"  {nm}: p={yd['p'][0]:6.2f} | d={yd['d'][0]:5.2f} | t={yd['t'][0]:5.2f} | "
              f"3He={yd['3He'][0]:5.2f} | 4He={yd['4He'][0]:5.2f} | 6He={yd['6He'][0]:.4f} | 6Li={yd['6Li'][0]:.4f}")


if __name__ == "__main__":
    main()
