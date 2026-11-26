#!/usr/bin/env python3
"""
High-statistics FOPI Au+Au @ 250 A MeV benchmark of Pratt-Torrieri local momentum
anisotropy (p*_i = Lambda_ij q_j, det Lambda = 1, matching 3x3 Eckart stress P*_ij)
in HCGF freeze-out (n_fr = rho_0 / 8, A <= 6, 58 species) vs. Isotropic HCGF,
Microscopic BUU, and FOPI Collaboration experimental data.
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
ISO_DIR = ROOT / "output" / "highstat_a456"
OUT_DIR = ROOT / "output" / "highstat_aniso"
RUNS_DIR = OUT_DIR / "runs"
PNG_OUT = ROOT / "assets" / "fopi_hcgf_anisotropy_comparison.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160


def run_job(job):
    tag, b_fm, ntest, aniso_flag, save_cells, seed = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    cells_csv = RUNS_DIR / f"{tag}.cells.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and (not save_cells or cells_csv.exists()):
        return tag, 0.0, "cached", fl_csv

    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "72.0", "--grid", "30",
        "--steps", "280", "--dt", "0.30", "--substeps", "2",
        "--eos", "md-soft", "--in-medium-ec", "on",
        "--cluster-mode", "freezeout", "--backflow", "remelt",
        "--anisotropy", aniso_flag,
        "--max-cluster-A", "6", "--nfr", f"{RHO0 / 8.0:.5f}", "--freezeout-every", "4",
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
    out = {}
    for r in csv.DictReader(open(str(csv_p) + ".yields.csv")):
        out[r["species"]] = (float(r["yield_per_event"]), float(r["yield_err"]))
    return out


def run_ensembles():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Exact same impact parameters & Philox seeds as highstat_a456 (event-by-event paired comparison!)
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        b_fm = B_MAX_AUAU * math.sqrt(b0_sq)
        seed = 410000 + k * 7919
        jobs.append((f"c2_pt8_a6_{k:02d}", b_fm, 500, "on", (k < 4), seed))

    for k in range(5):
        b_fm = B_MAX_AUAU * 0.15 * math.sqrt((k + 0.5) / 5.0)
        seed = 520000 + k * 7919
        jobs.append((f"c1_pt8_a6_{k:02d}", b_fm, 500, "on", False, seed))

    workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(jobs)} Pratt-Torrieri anisotropic HCGF jobs across {workers} workers (7,500 events)...", flush=True)
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            tag, dt_s, info, p = fut.result()
            out[tag] = p
            done += 1
            if dt_s > 0.0:
                print(f"  [{done:2d}/{len(jobs)}] {tag:14s} ({dt_s:5.1f}s) | {info}", flush=True)

    merged = {}
    for key, nk in [("c2_pt8_a6", 10), ("c1_pt8_a6", 5)]:
        dst = OUT_DIR / f"{key}.csv"
        merge_ensemble([out[f"{key}_{k:02d}"] for k in range(nk)], dst)
        merged[key] = dst
    print(f"Merged Pratt-Torrieri ensembles in {time.time() - t0:.1f}s!", flush=True)
    return merged


def load_cell_anisotropy():
    stretches, shears, weights = [], [], []
    for k in range(4):
        p = RUNS_DIR / f"c2_pt8_a6_{k:02d}.cells.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if int(r["clusters_formed"]) <= 0:
                continue
            stretches.append(float(r["aniso_stretch"]))
            shears.append(-float(r["shear_xy"]))  # (P*_yy - P*_xx)/(P*_xx + P*_yy): positive = out-of-plane squeeze
            weights.append(float(r["baryons_converted"]))
    return np.array(stretches), np.array(shears), np.array(weights)


def plot_anisotropy(merged, fopi):
    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.3, "axes.labelsize": 9.3,
        "legend.fontsize": 7.5, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), dpi=185)
    fig.suptitle(
        "HCGF Pratt–Torrieri Local Momentum Anisotropy ($p^*_i = \\Lambda_{ij} q_j,\\; \\det\\Lambda = 1$) vs. Isotropic HCGF & Microscopic BUU — FOPI Au+Au @ 250 $A$ MeV",
        fontsize=11.9, fontweight="bold", y=0.98
    )

    c2_pt  = merged["c2_pt8_a6"]
    c1_pt  = merged["c1_pt8_a6"]
    c2_iso = ISO_DIR / "c2_fo8_a6.csv"
    c1_iso = ISO_DIR / "c1_fo8_a6.csv"
    c2_mic = ISO_DIR / "c2_mic_a6.csv"
    c1_mic = ISO_DIR / "c1_mic_a6.csv"

    # (a) Local Eckart cell anisotropy distribution across freeze-out hypersurface (n_B < rho_0/8)
    ax = axes[0, 0]
    st_arr, sh_arr, w_arr = load_cell_anisotropy()
    if len(st_arr) > 0:
        ax.hist(st_arr, bins=np.linspace(1.0, 2.8, 38), weights=w_arr, density=True,
                color="#c0392b", alpha=0.72, edgecolor="white", label="$\\alpha_{\\max}/\\alpha_{\\min}$ of $\\Lambda_{ij}$")
        ax.axvline(np.average(st_arr, weights=w_arr), color="#922b21", ls="--", lw=1.8,
                   label=f"Mean stretch = {np.average(st_arr, weights=w_arr):.2f}")
    ax.set_xlabel("Pratt–Torrieri Principal Stretch $\\alpha_{\\max} / \\alpha_{\\min}$")
    ax.set_ylabel("Baryon-Weighted Cell Probability Density")
    ax.set_title("(a) Local Cell Shear Stretch at $n_B < \\rho_0/8$ ($0.25 < b_0 < 0.45$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    if len(sh_arr) > 0:
        ax_in = ax.inset_axes([0.46, 0.40, 0.50, 0.36])
        ax_in.hist(sh_arr, bins=np.linspace(-0.45, 0.55, 32), weights=w_arr, density=True,
                   color="#2980b9", alpha=0.75)
        ax_in.axvline(0.0, color="k", ls=":", lw=1.0)
        ax_in.axvline(np.average(sh_arr, weights=w_arr), color="#1b4f72", ls="--", lw=1.4)
        ax_in.set_title("$(P^*_{yy}-P^*_{xx})/(P^*_{xx}+P^*_{yy})$", fontsize=7.5)
        ax_in.tick_params(labelsize=6.8)

    # (b) Central Au+Au (b0 < 0.15) yields: det(Lambda) = 1 preserves exact chemical yields!
    ax = axes[0, 1]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$", "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$"]
    xp = np.arange(len(sps)); w = 0.22
    y_iso = load_yields(c1_iso)
    y_pt  = load_yields(c1_pt)
    y_mic = load_yields(c1_mic)
    ax.bar(xp - w, [y_iso[s][0] for s in sps], yerr=[y_iso[s][1] for s in sps], width=w, color="#7f8c8d", capsize=2, label="Isotropic HCGF ($\\Lambda = I$)")
    ax.bar(xp,     [y_pt[s][0]  for s in sps], yerr=[y_pt[s][1]  for s in sps], width=w, color="#c0392b", capsize=2, label="Pratt–Torrieri HCGF ($\\det\\Lambda=1$)")
    ax.bar(xp + w, [y_mic[s][0] for s in sps], yerr=[y_mic[s][1] for s in sps], width=w, color="#1f77b4", capsize=2, label="Microscopic BUU ($A \\leq 6$)")
    f_sps = ["p", "d", "t", "3He", "4He"]
    ax.errorbar(np.arange(5), [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps],
                yerr=[fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps],
                fmt="ko", ms=5.5, capsize=3, lw=1.5, zorder=7, label="FOPI ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 180.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (2,500 ev)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(b) Central Isotope Yields: Exact $\\det\\Lambda=1$ Invariance")

    # (c) Rapidity Spectra dN/dy0 (0.25 < b0 < 0.45): longitudinal vs transverse stress P*_zz / P*_perp
    ax = axes[0, 2]
    for sp, col, mult, lbl in [("d", "#1f77b4", 1.0, "$d$"), ("t", "#27ae60", 1.0, "$t$"),
                               ("4He", "#d62728", 5.0, "$^4\\mathrm{He}\\times 5$"),
                               ("6Li", "#e67e22", 80.0, "$^6\\mathrm{Li}\\times 80$")]:
        yp_r, dp_r, dpe, _, _, _ = load_rap(c2_pt, sp)
        yi_r, di_r, _, _, _, _   = load_rap(c2_iso, sp)
        ym_r, dm_r, _, _, _, _   = load_rap(c2_mic, sp)
        ax.plot(yp_r, dp_r * mult, "-",  color=col, lw=2.2, label=f"{lbl} PT-Aniso")
        ax.fill_between(yp_r, (dp_r - dpe) * mult, (dp_r + dpe) * mult, color=col, alpha=0.13)
        ax.plot(yi_r, di_r * mult, ":",  color=col, lw=2.0, label=f"{lbl} Iso ($\\Lambda=I$)")
        ax.plot(ym_r, dm_r * mult, "-.", color=col, lw=1.4, alpha=0.72, label=f"{lbl} Micro")
    ax.set_xlim(-1.38, 1.38)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ (5,000 Events)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.4)
    ax.set_title("(c) Rapidity Spectra $dN/dy_0$: PT-Aniso vs. Isotropic HCGF")

    # (d) Directed Flow v1(y0) (u_t0 > 0.8): reaction-plane tilt shear P*_xz
    ax = axes[1, 0]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"), ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        yp_r, _, _, v1p, e1p, np8 = load_rap(c2_pt, sp)
        yi_r, _, _, v1i, _, ni8   = load_rap(c2_iso, sp)
        sp_m = (np.abs(yp_r) <= 1.08) & (np8 >= 20)
        si_m = (np.abs(yi_r) <= 1.08) & (ni8 >= 20)
        ax.plot(yp_r[sp_m], v1p[sp_m], "-", color=col, lw=2.2, label=f"{lbl} PT-Aniso")
        ax.fill_between(yp_r[sp_m], v1p[sp_m] - e1p[sp_m], v1p[sp_m] + e1p[sp_m], color=col, alpha=0.14)
        ax.plot(yi_r[si_m], v1i[si_m], ":", color=col, lw=2.0, label=f"{lbl} Iso ($\\Lambda=I$)")
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
    ax.set_title("(d) Directed Flow $v_1(y_0)$: PT-Aniso vs. Iso HCGF vs. FOPI")

    # (e) Differential Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4, 5,000 events)
    ax = axes[1, 1]
    for sp, ekey, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                   ("d", "d", "#1f77b4", "s", "$d$"),
                                   ("A3", "A3", "#d62728", "^", "$A=3$"),
                                   ("4He", "4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        up_a, v2p, e2p, np_u = load_ut0(c2_pt, sp)
        ui_a, v2i, _,   ni_u = load_ut0(c2_iso, sp)
        um_a, v2m, _,   nm_u = load_ut0(c2_mic, sp)
        sp_u = (up_a >= 0.40) & (up_a <= 1.80) & (np_u >= 40)
        si_u = (ui_a >= 0.40) & (ui_a <= 1.80) & (ni_u >= 40)
        sm_u = (um_a >= 0.40) & (um_a <= 1.80) & (nm_u >= 40)
        ax.plot(up_a[sp_u], v2p[sp_u], "-",  color=col, lw=2.2, label=f"{lbl} PT-Aniso")
        ax.fill_between(up_a[sp_u], v2p[sp_u] - e2p[sp_u], v2p[sp_u] + e2p[sp_u], color=col, alpha=0.14)
        ax.plot(ui_a[si_u], v2i[si_u], ":",  color=col, lw=2.0, label=f"{lbl} Iso ($\\Lambda=I$)")
        ax.plot(um_a[sm_u], v2m[sm_u], "-.", color=col, lw=1.5, alpha=0.75, label=f"{lbl} Micro")
        v2_dict = fopi["v2_ut0_y0_lt_04"]
        if ekey in v2_dict:
            eu = v2_dict[ekey]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.35, 1.85); ax.set_ylim(-0.165, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=3, fontsize=6.2)
    ax.set_title("(e) Squeeze-Out $v_2(u_{t0})$: PT-Aniso vs. Iso HCGF vs. Micro")

    # (f) Intra-cell local shear contribution Delta v2 = v2(PT-Aniso) - v2(Iso)
    ax = axes[1, 2]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("A3", "#d62728", "^", "$A=3$ ($t+{}^3\\mathrm{He}$)")]:
        up_a, v2p, e2p, np_u = load_ut0(c2_pt, sp)
        ui_a, v2i, e2i, ni_u = load_ut0(c2_iso, sp)
        mask = (up_a >= 0.45) & (up_a <= 1.45) & (np_u >= 350) & (ni_u >= 350)
        dv2 = v2p[mask] - v2i[mask]
        edv = np.sqrt(e2p[mask] ** 2 + e2i[mask] ** 2)
        ax.plot(up_a[mask], dv2, "-" + mk, color=col, lw=2.1, ms=5.0, label=lbl)
        ax.fill_between(up_a[mask], dv2 - edv, dv2 + edv, color=col, alpha=0.15)
    ax.axhline(0.0, color="#444", ls="--", lw=1.2)
    ax.set_xlim(0.42, 1.45); ax.set_ylim(-0.075, 0.045)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("$\\Delta v_2(u_{t0}) = v_2^{\\mathrm{PT\\text{-}Aniso}} - v_2^{\\mathrm{Iso}}$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", fontsize=8.2)
    ax.set_title("(f) Extra Squeeze-Out ($\\Delta v_2 < 0$) from Local Shear $\\pi^*_{ij}$")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_OUT, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_OUT}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_ensembles()
    plot_anisotropy(merged, fopi)
    y_iso = load_yields(ISO_DIR / "c1_fo8_a6.csv")
    y_pt  = load_yields(merged["c1_pt8_a6"])
    print("\n=== Central Au+Au (b0 < 0.15, 2,500 Events) Yield Check (det Lambda = 1) ===")
    for nm, yd in [("Isotropic HCGF (Lambda=I)  ", y_iso), ("Pratt-Torrieri (detLambda=1)", y_pt)]:
        print(f"  {nm}: p={yd['p'][0]:6.2f} | d={yd['d'][0]:5.2f} | t={yd['t'][0]:5.2f} | "
              f"3He={yd['3He'][0]:4.2f} | 4He={yd['4He'][0]:4.2f} | 6He={yd['6He'][0]:.4f} | 6Li={yd['6Li'][0]:.4f}")


if __name__ == "__main__":
    main()
