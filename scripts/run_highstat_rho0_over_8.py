#!/usr/bin/env python3
"""
High-statistics multi-core FOPI Au+Au @ 250 A MeV benchmark for Baryon-Only
Eckart-Frame Freeze-Out at n_fr = rho_0 / 8 = 0.020 fm^-3 (plus rho_0/3, rho_0/4, rho_0/6)
vs. Full Microscopic Dynamical Cluster Transport and Published FOPI Data.

Ensemble statistics (27,500 Au+Au events = 10.84 million test nucleons):
  - Semicentral 0.25 < b0 < 0.45 (10 b*db points x N_test=500 = 5,000 events each):
      * Eckart Freeze-Out n_fr = rho_0 / 8 (0.020 fm^-3, backflow=remelt)
      * Eckart Freeze-Out n_fr = rho_0 / 4 (0.040 fm^-3, backflow=remelt)
      * Microscopic Dynamical Cluster Transport
  - Central b0 < 0.15 (5 b*db points x N_test=500 = 2,500 events each):
      * Eckart Freeze-Out n_fr in {rho_0/3, rho_0/4, rho_0/6, rho_0/8} + Microscopic BUU
"""

import csv
import json
import math
import os
import struct
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIN_FP32 = ROOT / "build" / "cluster_transport_fp32"
FOPI_JSON = ROOT / "scripts" / "fopi_exp_auau250.json"
OUT_DIR = ROOT / "output" / "highstat_rho08"
RUNS_DIR = OUT_DIR / "runs"
PNG_COMP = ROOT / "assets" / "hcgf_vs_microscopic_comparison.png"
PNG_RHO8 = ROOT / "assets" / "fopi_eckart_rho0_over_8.png"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))  # 13.380 fm
RHO0 = 0.160


def run_job(job):
    tag, b_fm, ntest, cmode, nfr, seed, dump_extra = job
    fl_csv = RUNS_DIR / f"{tag}.csv"
    if (Path(str(fl_csv) + ".yields.csv")).exists() and not dump_extra:
        return tag, 0.0, "", fl_csv

    cmd = [
        str(BIN_FP32), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--ntest", str(ntest), "--box", "72.0", "--grid", "30",
        "--steps", "280", "--dt", "0.30", "--substeps", "2",
        "--eos", "md-soft", "--in-medium-ec", "on",
        "--cluster-mode", cmode, "--backflow", "remelt",
        "--nfr", f"{nfr:.5f}", "--freezeout-every", "4",
        "--seed", str(seed), "--flow-csv", str(fl_csv),
    ]
    if dump_extra:
        cmd += [
            "--dump-tmunu-dir", str(OUT_DIR / f"{tag}_tmunu"),
            "--dump-tmunu-every", "20",
            "--freezeout-cells-csv", str(OUT_DIR / f"{tag}_cells.csv"),
        ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    fo_ln = [ln.strip() for ln in res.stdout.splitlines() if "Freeze-Out Grid:" in ln]
    return tag, dt_s, (fo_ln[-1] if fo_ln else "microscopic"), fl_csv


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


def read_hcgf_tmunu(path):
    with open(path, "rb") as f:
        hdr = f.read(88)
        magic, _, step, nx, ny, nz, nf, _, t_fm, xmin, xmax, ymin, ymax, zmin, zmax = struct.unpack("<Iiiiiiiiddddddd", hdr)
        assert magic == 0x46474348 and nf == 24
        arr = np.frombuffer(f.read(nf * nx * ny * nz * 4), dtype=np.float32).reshape((nf, nz, ny, nx))
    return {"step": step, "t_fm": t_fm, "nx": nx, "ny": ny, "nz": nz, "ext": (xmin, xmax, zmin, zmax), "f": arr}


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


def run_all_ensembles():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Semicentral 0.25 < b0 < 0.45: 10 b*db points x N_test=500 = 5,000 events per model
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        b_fm = B_MAX_AUAU * math.sqrt(b0_sq)
        seed = 410000 + k * 7919
        jobs.append((f"c2_mic_{k:02d}",  b_fm, 500, "microscopic", RHO0 / 8.0, seed, False))
        jobs.append((f"c2_fo4_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 4.0, seed, False))
        jobs.append((f"c2_fo8_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 8.0, seed, (k == 5)))

    # Central b0 < 0.15: 5 b*db points x N_test=500 = 2,500 events per model
    for k in range(5):
        b_fm = B_MAX_AUAU * 0.15 * math.sqrt((k + 0.5) / 5.0)
        seed = 520000 + k * 7919
        jobs.append((f"c1_mic_{k:02d}",  b_fm, 500, "microscopic", RHO0 / 8.0, seed, False))
        jobs.append((f"c1_fo3_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 3.0, seed, False))
        jobs.append((f"c1_fo4_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 4.0, seed, False))
        jobs.append((f"c1_fo6_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 6.0, seed, False))
        jobs.append((f"c1_fo8_{k:02d}",  b_fm, 500, "freezeout",   RHO0 / 8.0, seed, False))

    workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(jobs)} Au+Au jobs across {workers} workers (27,500 events = 10.84M test nucleons)...", flush=True)
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
                print(f"  [{done:2d}/{len(jobs)}] {tag:11s} ({dt_s:5.1f}s) | {info}", flush=True)

    merged = {}
    for key, n_k in [("c2_mic", 10), ("c2_fo4", 10), ("c2_fo8", 10),
                     ("c1_mic", 5),  ("c1_fo3", 5),  ("c1_fo4", 5), ("c1_fo6", 5), ("c1_fo8", 5)]:
        dst = OUT_DIR / f"{key}.csv"
        merge_ensemble([out[f"{key}_{k:02d}"] for k in range(n_k)], dst)
        merged[key] = dst
    print(f"All 27,500 events merged in {time.time() - t0:.1f}s!", flush=True)
    return merged


def plot_panel_slice_and_cells(ax_slice, ax_cells, tag_extra="c2_fo8_05"):
    tm = read_hcgf_tmunu(OUT_DIR / f"{tag_extra}_tmunu" / "tmunu_grid_0005.bin")
    flds = tm["f"]
    nym = tm["ny"] // 2
    eps_e = 0.5 * (flds[18, :, nym - 1, :] + flds[18, :, nym, :])
    nb_e  = 0.5 * (flds[20, :, nym - 1, :] + flds[20, :, nym, :])
    vx_e  = 0.5 * (flds[21, :, nym - 1, :] + flds[21, :, nym, :])
    vz_e  = 0.5 * (flds[23, :, nym - 1, :] + flds[23, :, nym, :])
    e_exc = np.maximum(0.0, (eps_e - 0.9389 * nb_e) * 1e3)

    xmin, xmax, zmin, zmax = tm["ext"]
    im = ax_slice.imshow(e_exc.T, origin="lower", extent=[zmin, zmax, xmin, xmax],
                         cmap="inferno", aspect="auto", interpolation="bilinear")
    zs = np.linspace(zmin + 1.2, zmax - 1.2, tm["nz"])
    xs = np.linspace(xmin + 1.2, xmax - 1.2, tm["nx"])
    ZZ, XX = np.meshgrid(zs, xs)
    for lv, col, lw, ls in [(RHO0 / 8.0, "#00ff88", 1.8, "-"),
                            (RHO0 / 4.0, "#00ffff", 1.3, "--"),
                            (RHO0,       "#ffffff", 1.2, "-")]:
        if float(nb_e.max()) > lv * 1.02:
            ax_slice.contour(ZZ, XX, nb_e.T, levels=[lv], colors=[col], linewidths=[lw], linestyles=[ls])
    st = 2
    mq = nb_e.T[::st, ::st] > 0.006
    ax_slice.quiver(ZZ[::st, ::st][mq], XX[::st, ::st][mq],
                    vz_e.T[::st, ::st][mq], vx_e.T[::st, ::st][mq],
                    color="white", alpha=0.82, scale=2.7, width=0.004)
    cb = plt.colorbar(im, ax=ax_slice, fraction=0.046, pad=0.03)
    cb.set_label("$\\varepsilon_{\\mathrm{Eckart}} - m_N n_B$ [$\\mathrm{MeV/fm^3}$]")
    ax_slice.set_xlim(-22, 22); ax_slice.set_ylim(-20, 20)
    ax_slice.set_xlabel("Beam Axis $z$ [fm]"); ax_slice.set_ylabel("Reaction Plane $x$ [fm]")
    ax_slice.set_title(f"(a) Eckart $T_B^{{\\mu\\nu}}$ ($t={tm['t_fm']:.0f}$ fm/$c$; green: $\\rho_0/8$, cyan: $\\rho_0/4$)")

    fo_T, fo_muB, fo_t, fo_b = [], [], [], []
    for r in csv.DictReader(open(OUT_DIR / f"{tag_extra}_cells.csv")):
        if int(r["baryons_converted"]) >= 6 and float(r["muB_mev"]) > 300.0:
            fo_T.append(float(r["T_mev"])); fo_muB.append(float(r["muB_mev"]))
            fo_t.append(float(r["t_fm"]));  fo_b.append(int(r["baryons_converted"]))
    sc = ax_cells.scatter(fo_muB, fo_T, c=fo_t, s=np.clip(np.array(fo_b) * 0.55, 8, 75),
                          cmap="viridis", alpha=0.78, edgecolors="none")
    cb2 = plt.colorbar(sc, ax=ax_cells, fraction=0.046, pad=0.03)
    cb2.set_label("Freeze-Out Time $t_{\\mathrm{fr}}$ [fm/$c$]")
    ax_cells.set_xlabel("Cell Baryon Chemical Potential $\\mu_B$ [MeV]")
    ax_cells.set_ylabel("Eckart Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax_cells.grid(alpha=0.25)
    ax_cells.set_title("(b) Eckart $u^0 E - \\vec{P}\\cdot\\vec{u}$ Cells at $n_{\\mathrm{fr}} = \\rho_0/8$ ($0.02\\,\\mathrm{fm^{-3}}$)")


def render_figure(merged, fopi, out_png, show_rho4_overlay=False):
    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.3, "axes.labelsize": 9.4,
        "legend.fontsize": 7.6, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.0), dpi=185)
    title_str = (
        "High-Statistics FOPI Au+Au @ 250 $A$ MeV: Microscopic BUU vs. Eckart $n_{\\mathrm{fr}} = \\rho_0/8$ & $\\rho_0/4$ (5,000 Events/Curve)"
        if show_rho4_overlay else
        "High-Statistics Eckart-Frame Freeze-Out at $n_{\\mathrm{fr}} = \\rho_0 / 8$ ($0.020\\;\\mathrm{fm^{-3}}$) vs. Microscopic BUU & FOPI (5,000 Events)"
    )
    fig.suptitle(title_str, fontsize=12.6, fontweight="bold", y=0.98)

    plot_panel_slice_and_cells(axes[0, 0], axes[0, 1])

    # (c) Central b0 < 0.15 cluster yields (2,500 events each): rho0/3, rho0/4, rho0/6, rho0/8 vs Micro & FOPI
    ax = axes[0, 2]
    sp_list = ["p", "d", "t", "3He", "4He"]
    sp_lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$"]
    x_pos = np.arange(len(sp_list))
    w = 0.145
    bars = [
        ("c1_mic", "#1f77b4", "Microscopic BUU"),
        ("c1_fo3", "#f39c12", "Eckart $\\rho_0/3$"),
        ("c1_fo4", "#27ae60", "Eckart $\\rho_0/4$"),
        ("c1_fo6", "#8e44ad", "Eckart $\\rho_0/6$"),
        ("c1_fo8", "#c0392b", "Eckart $\\mathbf{\\rho_0/8}$"),
    ]
    for bi, (key, col, lbl) in enumerate(bars):
        yd = load_yields(merged[key])
        vals = [yd[s][0] for s in sp_list]
        errs = [yd[s][1] for s in sp_list]
        ax.bar(x_pos + (bi - 2.0) * w, vals, yerr=errs, width=w, color=col, capsize=2, label=lbl, alpha=0.92)
    f_vals = [fopi["central_yields_b0_lt_015"][s]["val"] for s in sp_list]
    f_errs = [fopi["central_yields_b0_lt_015"][s]["err"] for s in sp_list]
    ax.errorbar(x_pos, f_vals, yerr=f_errs, fmt="ko", ms=6.0, capsize=3.5, lw=1.6, zorder=7, label="FOPI ($b_0<0.15$)")
    ax.set_yscale("log"); ax.set_ylim(0.5, 180.0)
    ax.set_xticks(x_pos); ax.set_xticklabels(sp_lbls)
    ax.set_ylabel("Central Multiplicity / Event (2,500 ev)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=7.0, ncol=2)
    ax.set_title("(c) Central Cluster Yields: $\\rho_0/3 \\to \\rho_0/8$ vs. FOPI")

    # (d) High-stat Semicentral (0.25 < b0 < 0.45, 5,000 events) Rapidity Spectra dN/dy0
    ax = axes[1, 0]
    for sp, col, mult, lbl in [("p", "#2c3e50", 1.0, "$p$"), ("d", "#1f77b4", 1.0, "$d$"),
                               ("t", "#27ae60", 1.0, "$t$"), ("3He", "#8e44ad", 1.0, "$^3\\mathrm{He}$"),
                               ("4He", "#d62728", 8.0, "$^4\\mathrm{He}\\times 8$")]:
        ym, dm, dme, _, _, _ = load_rap(merged["c2_mic"], sp)
        y8, d8, d8e, _, _, _ = load_rap(merged["c2_fo8"], sp)
        ax.plot(ym, dm * mult, "-", color=col, lw=2.0, label=f"{lbl} Micro")
        ax.fill_between(ym, (dm - dme) * mult, (dm + dme) * mult, color=col, alpha=0.12)
        ax.plot(y8, d8 * mult, "--", color=col, lw=2.1, label=f"{lbl} $\\rho_0/8$")
        ax.fill_between(y8, (d8 - d8e) * mult, (d8 + d8e) * mult, color=col, alpha=0.12)
        if show_rho4_overlay and sp in ("d", "t"):
            y4, d4, _, _, _, _ = load_rap(merged["c2_fo4"], sp)
            ax.plot(y4, d4 * mult, ":", color=col, lw=1.6, alpha=0.85, label=f"{lbl} $\\rho_0/4$")
    ax.set_xlim(-1.40, 1.40)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$ (5,000 Events)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=6.7)
    ax.set_title("(d) High-Stat Rapidity Spectra $dN/dy_0$ ($0.25 < b_0 < 0.45$)")

    # (e) High-stat Directed Flow v1(y0) (u_t0 > 0.8) vs FOPI (5,000 events)
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"),
                             ("t", "#d62728", "^", "$t$"), ("3He", "#27ae60", "D", "$^3\\mathrm{He}$")]:
        ym, _, _, v1m, e1m, nm = load_rap(merged["c2_mic"], sp)
        y8, _, _, v18, e18, n8 = load_rap(merged["c2_fo8"], sp)
        sm = (np.abs(ym) <= 1.08) & (nm >= 25)
        s8 = (np.abs(y8) <= 1.08) & (n8 >= 25)
        ax.plot(ym[sm], v1m[sm], "-", color=col, lw=2.0, label=f"{lbl} Micro")
        ax.fill_between(ym[sm], v1m[sm] - e1m[sm], v1m[sm] + e1m[sm], color=col, alpha=0.14)
        ax.plot(y8[s8], v18[s8], "--", color=col, lw=2.1, label=f"{lbl} $\\rho_0/8$")
        ax.fill_between(y8[s8], v18[s8] - e18[s8], v18[s8] + e18[s8], color=col, alpha=0.14)
        if sp in fopi["v1_y0_ut0_gt_08"]:
            ep = fopi["v1_y0_ut0_gt_08"][sp]
            ax.errorbar([p["y0"] for p in ep[::2]], [p["v1"] for p in ep[::2]],
                        yerr=[p["err"] for p in ep[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.72, 0.72)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.8$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=6.5)
    ax.set_title("(e) High-Stat Directed Flow $v_1(y_0)$: $\\rho_0/8$ vs. Micro vs. FOPI")

    # (f) High-stat Differential Elliptic Flow v2(u_t0) (|y0| < 0.4, 5,000 events)
    ax = axes[1, 2]
    for sp, ekey, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                   ("d", "d", "#1f77b4", "s", "$d$"),
                                   ("A3", "A3", "#d62728", "^", "$A=3$")]:
        um, v2m, e2m, nm = load_ut0(merged["c2_mic"], sp)
        u8, v28, e28, n8 = load_ut0(merged["c2_fo8"], sp)
        sm = (um >= 0.40) & (um <= 1.80) & (nm >= 80)
        s8 = (u8 >= 0.40) & (u8 <= 1.80) & (n8 >= 80)
        ax.plot(um[sm], v2m[sm], "-", color=col, lw=2.0, label=f"{lbl} Micro")
        ax.fill_between(um[sm], v2m[sm] - e2m[sm], v2m[sm] + e2m[sm], color=col, alpha=0.14)
        ax.plot(u8[s8], v28[s8], "--", color=col, lw=2.1, label=f"{lbl} $\\rho_0/8$")
        ax.fill_between(u8[s8], v28[s8] - e28[s8], v28[s8] + e28[s8], color=col, alpha=0.14)
        if show_rho4_overlay and sp == "d":
            u4, v24, _, n4 = load_ut0(merged["c2_fo4"], sp)
            s4 = (u4 >= 0.40) & (u4 <= 1.80) & (n4 >= 80)
            ax.plot(u4[s4], v24[s4], ":", color=col, lw=1.7, label="$d$ $\\rho_0/4$")
        v2_dict = fopi["v2_ut0_y0_lt_04"]
        eu = v2_dict.get(ekey, v2_dict.get("t", []))
        if eu:
            ax.errorbar([p["ut0"] for p in eu[::2]], [p["v2"] for p in eu[::2]],
                        yerr=[p["err"] for p in eu[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.35, 1.85); ax.set_ylim(-0.155, 0.03)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncol=3, fontsize=6.5)
    ax.set_title("(f) High-Stat Squeeze-Out $v_2(u_{t0})$: $\\rho_0/8$ vs. Micro vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=185)
    plt.close(fig)
    print(f"Saved -> {out_png}", flush=True)


def main():
    fopi = json.load(open(FOPI_JSON))
    merged = run_all_ensembles()
    render_figure(merged, fopi, PNG_RHO8, show_rho4_overlay=False)
    render_figure(merged, fopi, PNG_COMP, show_rho4_overlay=True)

    print("\n=== Central Au+Au (b0 < 0.15, 2,500 Events/Model) Yields ===")
    for key, lbl in [("c1_mic", "Microscopic BUU"), ("c1_fo3", "Eckart rho0/3 (0.053)"),
                     ("c1_fo4", "Eckart rho0/4 (0.040)"), ("c1_fo6", "Eckart rho0/6 (0.027)"),
                     ("c1_fo8", "Eckart rho0/8 (0.020)")]:
        yd = load_yields(merged[key])
        print(f"  {lbl:22s}: p={yd['p'][0]:6.2f}+-{yd['p'][1]:.2f} | d={yd['d'][0]:5.2f}+-{yd['d'][1]:.2f} | "
              f"t={yd['t'][0]:5.2f}+-{yd['t'][1]:.2f} | 3He={yd['3He'][0]:4.2f}+-{yd['3He'][1]:.2f} | "
              f"4He={yd['4He'][0]:4.2f}+-{yd['4He'][1]:.2f}")


if __name__ == "__main__":
    main()
