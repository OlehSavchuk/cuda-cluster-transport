#!/usr/bin/env python3
"""
High-statistics FOPI Au+Au @ 250 A MeV benchmark for the extended A = 4, 5, 6
isotope & TUNL resonance table (58 species: 4H*, 4Li*, 5H, 5He, 5Li, 6H, 6He, 6Li, 6Be):
  - Semicentral 0.25 < b0 < 0.45 (5,000 events each): Eckart rho_0/8 (A<=6) & Microscopic (A<=6)
  - Central b0 < 0.15 (2,500 events each): Eckart rho_0/8 (A<=6) & Microscopic (A<=6)
    compared directly against cached A<=4 runs and Published FOPI Collaboration data!
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
A4_DIR = ROOT / "output" / "highstat_rho08"
OUT_DIR = ROOT / "output" / "highstat_a456"
RUNS_DIR = OUT_DIR / "runs"
PNG_A456 = ROOT / "assets" / "fopi_a456_isotopes_benchmark.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))
RHO0 = 0.160
HBARC = 0.1973269804
MP = 0.93827208816
MN = 0.93956542052


def run_job(job):
    tag, b_fm, ntest, cmode, nfr, max_A, seed = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists():
        return tag, 0.0, "cached", fl_csv

    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "72.0", "--grid", "30",
        "--steps", "280", "--dt", "0.30", "--substeps", "2",
        "--eos", "md-soft", "--in-medium-ec", "on",
        "--cluster-mode", cmode, "--backflow", "remelt",
        "--max-cluster-A", str(max_A), "--nfr", f"{nfr:.5f}", "--freezeout-every", "4",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "microscopic-A6"), fl_csv


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


def compute_thermal_feeddown_curve():
    # Analytical Boltzmann Saha curve at n_B = rho_0/8 = 0.020 fm^-3, Y_p = 0.402 (Au)
    # showing primordial 4He vs 5He/5Li/6Li*/6He*/6Be feed-down + bound 6He, 6Li vs T!
    specs = [
        ("p", 1, 1, 2, MP), ("n", 1, 0, 2, MN), ("d", 2, 1, 3, 1.875613),
        ("t", 3, 1, 2, 2.808921), ("3He", 3, 2, 2, 2.808392), ("4He", 4, 2, 1, 3.727379),
        ("5He", 5, 2, 4, 4.667743), ("5He*", 5, 2, 2, 4.669013),
        ("5Li", 5, 3, 4, 4.667616), ("5Li*", 5, 3, 2, 4.669106),
        ("6He", 6, 2, 1, 5.605535), ("6He*", 6, 2, 5, 5.607385),
        ("6Li", 6, 3, 3, 5.601518), ("6Li*3+", 6, 3, 7, 5.603704),
        ("6Li*2+", 6, 3, 5, 5.605830), ("6Li*2+1", 6, 3, 5, 5.606884),
        ("6Li*1+", 6, 3, 3, 5.607168), ("6Be", 6, 4, 1, 5.605295), ("6Be*", 6, 4, 5, 5.606965),
    ]
    Ts = np.linspace(4.0, 28.0, 60)
    nb = 0.020; yp = 79.0 / 197.0
    prim_4he, a5_feed, a6_feed, b_6li, b_6he = [], [], [], [], []
    for T_mev in Ts:
        T = T_mev * 1e-3
        phis = []
        for nm, A, Z, g, m in specs:
            bsh = m - (Z * MP + (A - Z) * MN)
            ph = g * ((m * T / (2.0 * math.pi * HBARC * HBARC)) ** 1.5) * math.exp(-bsh / T)
            phis.append(ph)
        up = math.log(yp * nb / phis[0]); un = math.log((1 - yp) * nb / phis[1])
        for _ in range(18):
            fp = -yp * nb; fn = -(1 - yp) * nb; jpp = jpn = jnn = 0.0
            for idx, (_, A, Z, _, _) in enumerate(specs):
                N = A - Z
                na = phis[idx] * math.exp(max(-150.0, min(60.0, Z * up + N * un)))
                fp += Z * na; fn += N * na
                jpp += Z * Z * na; jpn += Z * N * na; jnn += N * N * na
            det = jpp * jnn - jpn * jpn
            up -= max(-1.5, min(1.5, (jnn * fp - jpn * fn) / det))
            un -= max(-1.5, min(1.5, (-jpn * fp + jpp * fn) / det))
        ns = {specs[i][0]: phis[i] * math.exp(specs[i][2] * up + (specs[i][1] - specs[i][2]) * un) for i in range(len(specs))}
        scale = 340.0 / nb  # Central Au+Au baryon count ~ 340
        prim_4he.append(scale * ns["4He"])
        a5_feed.append(scale * (ns["5He"] + ns["5He*"] + ns["5Li"] + ns["5Li*"]))
        a6_feed.append(scale * (ns["6He*"] + ns["6Li*3+"] + ns["6Li*2+"] + ns["6Li*2+1"] + ns["6Li*1+"] + ns["6Be"] + ns["6Be*"]))
        b_6li.append(scale * ns["6Li"]); b_6he.append(scale * ns["6He"])
    return Ts, np.array(prim_4he), np.array(a5_feed), np.array(a6_feed), np.array(b_6li), np.array(b_6he)


def run_ensembles():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Semicentral 0.25 < b0 < 0.45: 10 b*db points x N_test=500 = 5,000 events each (A<=6)
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        b_fm = B_MAX_AUAU * math.sqrt(b0_sq)
        seed = 410000 + k * 7919
        jobs.append((f"c2_fo8_a6_{k:02d}", b_fm, 500, "freezeout",   RHO0 / 8.0, 6, seed))
        jobs.append((f"c2_mic_a6_{k:02d}", b_fm, 500, "microscopic", RHO0 / 8.0, 6, seed))

    # Central b0 < 0.15: 5 b*db points x N_test=500 = 2,500 events each (A<=6, plus A<=4 check)
    for k in range(5):
        b_fm = B_MAX_AUAU * 0.15 * math.sqrt((k + 0.5) / 5.0)
        seed = 520000 + k * 7919
        jobs.append((f"c1_fo8_a6_{k:02d}", b_fm, 500, "freezeout",   RHO0 / 8.0, 6, seed))
        jobs.append((f"c1_fo4_a6_{k:02d}", b_fm, 500, "freezeout",   RHO0 / 4.0, 6, seed))
        jobs.append((f"c1_mic_a6_{k:02d}", b_fm, 500, "microscopic", RHO0 / 8.0, 6, seed))

    workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(jobs)} A<=6 Au+Au jobs across {workers} workers (17,500 events)...", flush=True)
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
    for key, nk in [("c2_fo8_a6", 10), ("c2_mic_a6", 10),
                    ("c1_fo8_a6", 5),  ("c1_fo4_a6", 5), ("c1_mic_a6", 5)]:
        dst = OUT_DIR / f"{key}.csv"
        merge_ensemble([out[f"{key}_{k:02d}"] for k in range(nk)], dst)
        merged[key] = dst
    print(f"Merged all A<=6 ensembles in {time.time() - t0:.1f}s!", flush=True)
    return merged


def plot_a456(merged, fopi):
    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.3, "axes.labelsize": 9.3,
        "legend.fontsize": 7.4, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), dpi=185)
    fig.suptitle(
        "Extended $A = 4, 5, 6$ Isotope & TUNL Resonance Network ($^4\\mathrm{H}^*, ^4\\mathrm{Li}^*, ^5\\mathrm{H}, ^5\\mathrm{He}, ^5\\mathrm{Li}, ^6\\mathrm{H}, ^6\\mathrm{He}, ^6\\mathrm{Li}, ^6\\mathrm{Be}$) — FOPI Au+Au @ 250 $A$ MeV",
        fontsize=12.2, fontweight="bold", y=0.98
    )

    # (a) Thermal feed-down anatomy at n_B = rho_0/8 (Yp = 0.402)
    ax = axes[0, 0]
    Ts, p4, f5, f6, b6li, b6he = compute_thermal_feeddown_curve()
    ax.plot(Ts, p4 + f5 + f6, color="#c0392b", lw=2.3, label="Total $^4\\mathrm{He}$ (Primordial + $A=5,6$ Feed-Down)")
    ax.plot(Ts, p4, "--", color="#e67e22", lw=1.9, label="Primordial $^4\\mathrm{He}$ ($A \\leq 4$ Only)")
    ax.fill_between(Ts, p4, p4 + f5, color="#f39c12", alpha=0.25, label="$^5\\mathrm{He}, ^5\\mathrm{Li} \\to \\alpha + N$ Feed-Down")
    ax.fill_between(Ts, p4 + f5, p4 + f5 + f6, color="#e74c3c", alpha=0.25, label="$^6\\mathrm{Li}^*, ^6\\mathrm{He}^*, ^6\\mathrm{Be} \\to \\alpha + X$ Feed-Down")
    ax.plot(Ts, b6li * 5.0, "-.", color="#2980b9", lw=1.9, label="$5 \\times$ Bound $^6\\mathrm{Li}$ (g.s.)")
    ax.plot(Ts, b6he * 5.0, ":", color="#8e44ad", lw=1.9, label="$5 \\times$ Bound $^6\\mathrm{He}$ (g.s.)")
    ax.set_xlim(4.0, 28.0); ax.set_ylim(0, 22.0)
    ax.set_xlabel("Eckart Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.set_ylabel("Equilibrium Yield ($A_B = 340,\\; n_B = \\rho_0/8$)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=7.1)
    ax.set_title("(a) $A = 5, 6 \\to {}^4\\mathrm{He}$ Feed-Down & Bound $^6\\mathrm{He}, ^6\\mathrm{Li}$ ($\\rho_0/8$)")

    # (b) Central Au+Au (b0 < 0.15) yields across p, d, t, 3He, 4He, 6He, 6Li
    ax = axes[0, 1]
    sps = ["p", "d", "t", "3He", "4He", "6He", "6Li"]
    lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$", "$^6\\mathrm{He}$", "$^6\\mathrm{Li}$"]
    xp = np.arange(len(sps)); w = 0.20
    y_a4 = load_yields(A4_DIR / "c1_fo8.csv")
    y_a6 = load_yields(merged["c1_fo8_a6"])
    y_m6 = load_yields(merged["c1_mic_a6"])
    ax.bar(xp - w, [y_a4.get(s, (0.0, 0.0))[0] for s in sps], width=w, color="#7f8c8d", label="Eckart $\\rho_0/8$ ($A \\leq 4$)")
    ax.bar(xp,     [y_a6[s][0] for s in sps], yerr=[y_a6[s][1] for s in sps], width=w, color="#c0392b", capsize=2, label="Eckart $\\rho_0/8$ ($\\mathbf{A \\leq 6}$)")
    ax.bar(xp + w, [y_m6[s][0] for s in sps], yerr=[y_m6[s][1] for s in sps], width=w, color="#1f77b4", capsize=2, label="Microscopic BUU ($\\mathbf{A \\leq 6}$)")
    f_sps = ["p", "d", "t", "3He", "4He"]
    ax.errorbar(np.arange(5), [fopi["central_yields_b0_lt_015"][s]["val"] for s in f_sps],
                yerr=[fopi["central_yields_b0_lt_015"][s]["err"] for s in f_sps],
                fmt="ko", ms=5.8, capsize=3, lw=1.5, zorder=7, label="FOPI ($b_0 < 0.15$)")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 180.0)
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("Central Multiplicity / Event (2,500 ev)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=7.1)
    ax.set_title("(b) Central Isotope Yields ($A=1\\dots 6$): $A\\leq 4$ vs. $A\\leq 6$ vs. FOPI")

    # (c) Feed-down enhancement ratio Y(A<=6) / Y(A<=4) in Eckart Freeze-Out (rho_0/8 & rho_0/4)
    ax = axes[0, 2]
    y4_fo4 = load_yields(A4_DIR / "c1_fo4.csv")
    y6_fo4 = load_yields(merged["c1_fo4_a6"])
    r_sps = ["p", "d", "t", "3He", "4He"]
    r_lbl = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$"]
    xr = np.arange(len(r_sps))
    rat8 = [y_a6[s][0] / max(y_a4[s][0], 1e-9) for s in r_sps]
    rat4 = [y6_fo4[s][0] / max(y4_fo4[s][0], 1e-9) for s in r_sps]
    b1 = ax.bar(xr - 0.17, rat8, width=0.32, color="#c0392b", label="Eckart $n_{\\mathrm{fr}} = \\rho_0/8$")
    b2 = ax.bar(xr + 0.17, rat4, width=0.32, color="#27ae60", label="Eckart $n_{\\mathrm{fr}} = \\rho_0/4$")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}x", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, color="#555", ls="--", lw=1.1)
    ax.set_xticks(xr); ax.set_xticklabels(r_lbl)
    ax.set_ylim(0.7, max(max(rat8), max(rat4)) * 1.22)
    ax.set_ylabel("Yield Ratio $Y(A \\leq 6) \\;/\\; Y(A \\leq 4)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left")
    ax.set_title("(c) Resonance Feed-Down Boost from $A = 4^*, 5^*, 6^*$ States")

    # (d) Semicentral (5,000 ev) Rapidity Spectra dN/dy0 including 6He & 6Li!
    ax = axes[1, 0]
    for sp, col, mult, lbl in [("d", "#1f77b4", 1.0, "$d$"), ("t", "#27ae60", 1.0, "$t$"),
                               ("3He", "#8e44ad", 1.0, "$^3\\mathrm{He}$"),
                               ("4He", "#d62728", 5.0, "$^4\\mathrm{He}\\times 5$"),
                               ("6Li", "#e67e22", 80.0, "$^6\\mathrm{Li}\\times 80$"),
                               ("6He", "#16a085", 150.0, "$^6\\mathrm{He}\\times 150$")]:
        y8, d8, d8e, _, _, _ = load_rap(merged["c2_fo8_a6"], sp)
        ym, dm, _, _, _, _   = load_rap(merged["c2_mic_a6"], sp)
        ax.plot(y8, d8 * mult, "--", color=col, lw=2.1, label=f"{lbl} ($\\rho_0/8$)")
        ax.fill_between(y8, (d8 - d8e) * mult, (d8 + d8e) * mult, color=col, alpha=0.13)
        if sp in ("d", "t", "4He", "6Li"):
            ax.plot(ym, dm * mult, "-", color=col, lw=1.6, alpha=0.80, label=f"{lbl} (Micro)")
    ax.set_xlim(-1.40, 1.40)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ (5,000 Events)")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", ncol=2, fontsize=6.6)
    ax.set_title("(d) Rapidity Spectra $dN/dy_0$ Up to $^6\\mathrm{He}, ^6\\mathrm{Li}$ ($0.25 < b_0 < 0.45$)")

    # (e) Mass-ordered Directed Flow v1(y0) (u_t0 > 0.8) from A=1 to A=6!
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"), ("4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        y8, _, _, v18, e18, n8 = load_rap(merged["c2_fo8_a6"], sp)
        ym, _, _, v1m, _, nm   = load_rap(merged["c2_mic_a6"], sp)
        s8 = (np.abs(y8) <= 1.08) & (n8 >= 20)
        sm = (np.abs(ym) <= 1.08) & (nm >= 20)
        ax.plot(y8[s8], v18[s8], "--", color=col, lw=2.1, label=f"{lbl} $\\rho_0/8$")
        ax.fill_between(y8[s8], v18[s8] - e18[s8], v18[s8] + e18[s8], color=col, alpha=0.14)
        ax.plot(ym[sm], v1m[sm], "-", color=col, lw=1.7, alpha=0.80, label=f"{lbl} Micro")
        if sp in fopi["v1_y0_ut0_gt_08"]:
            ep = fopi["v1_y0_ut0_gt_08"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.8$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", ncol=3, fontsize=6.5)
    ax.set_title("(e) Directed Flow $v_1(y_0)$ ($A \\leq 6$): $\\rho_0/8$ vs. Micro vs. FOPI")

    # (f) Differential Elliptic Squeeze-Out v2(u_t0) (|y0| < 0.4, 5,000 events)
    ax = axes[1, 2]
    for sp, ekey, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                   ("d", "d", "#1f77b4", "s", "$d$"),
                                   ("A3", "A3", "#d62728", "^", "$A=3$"),
                                   ("4He", "4He", "#8e44ad", "D", "$^4\\mathrm{He}$")]:
        u8, v28, e28, n8 = load_ut0(merged["c2_fo8_a6"], sp)
        um, v2m, e2m, nm = load_ut0(merged["c2_mic_a6"], sp)
        s8 = (u8 >= 0.40) & (u8 <= 1.80) & (n8 >= 40)
        sm = (um >= 0.40) & (um <= 1.80) & (nm >= 40)
        ax.plot(u8[s8], v28[s8], "--", color=col, lw=2.1, label=f"{lbl} $\\rho_0/8$")
        ax.fill_between(u8[s8], v28[s8] - e28[s8], v28[s8] + e28[s8], color=col, alpha=0.14)
        ax.plot(um[sm], v2m[sm], "-", color=col, lw=1.7, alpha=0.80, label=f"{lbl} Micro")
        v2_dict = fopi["v2_ut0_y0_lt_04"]
        if ekey in v2_dict:
            eu = v2_dict[ekey]
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.35, 1.85); ax.set_ylim(-0.16, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25); ax.legend(loc="lower left", ncol=3, fontsize=6.4)
    ax.set_title("(f) Squeeze-Out $v_2(u_{t0})$ ($A \\leq 6$): $\\rho_0/8$ vs. Micro vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PNG_A456, dpi=185)
    plt.close(fig)
    print(f"Saved -> {PNG_A456}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_ensembles()
    plot_a456(merged, fopi)
    print("\n=== Central Au+Au (b0 < 0.15, 2,500 Events) A <= 6 Isotope Yields ===")
    for key, lbl in [("c1_fo4_a6", "Eckart rho0/4 (A<=6)"),
                     ("c1_fo8_a6", "Eckart rho0/8 (A<=6)"),
                     ("c1_mic_a6", "Microscopic   (A<=6)")]:
        yd = load_yields(merged[key])
        print(f"  {lbl:22s}: p={yd['p'][0]:6.2f} | d={yd['d'][0]:5.2f} | t={yd['t'][0]:5.2f} | "
              f"3He={yd['3He'][0]:4.2f} | 4He={yd['4He'][0]:4.2f} | 6He={yd['6He'][0]:.4f} | 6Li={yd['6Li'][0]:.4f}")


if __name__ == "__main__":
    main()
