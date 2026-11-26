#!/usr/bin/env python3
"""
Side-by-side benchmark comparing two cluster-production paradigms in
FOPI Au + Au @ 250 A MeV (MD-Soft EOS, identical seeds & centrality):
  1. Pure Hadronic (N + Delta + pi) Transport + HCGF T^munu Grid Export
     + Local Thermal Clusterization in cells reaching n_B < n_fr
     (scanning n_fr = rho_0/3 = 0.053, rho_0/4 = 0.040, rho_0/6 = 0.027 fm^-3)
  2. Full Microscopic Dynamical Cluster Transport (35 species, 57 2<->2 + 2<->1 channels)
  3. Published FOPI Collaboration Data (Reisdorf et al., NPA 848, 2010; NPA 876, 2012)
"""

import csv
import json
import math
import os
import struct
import subprocess
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_FP32 = os.path.join(ROOT, "build", "cluster_transport_fp32")
FOPI_JSON = os.path.join(ROOT, "scripts", "fopi_exp_auau250.json")
OUT_DIR = os.path.join(ROOT, "output", "hcgf_compare")
ASSET_PNG = os.path.join(ROOT, "assets", "hcgf_vs_microscopic_comparison.png")


def read_hcgf_tmunu_file(path):
    with open(path, "rb") as f:
        hdr_bytes = f.read(88)
        magic, ver, step, nx, ny, nz, n_fields, _, t_fm, xmin, xmax, ymin, ymax, zmin, zmax = struct.unpack(
            "<Iiiiiiiiddddddd", hdr_bytes
        )
        assert magic == 0x46474348 and n_fields == 24
        nc = nx * ny * nz
        raw = np.frombuffer(f.read(n_fields * nc * 4), dtype=np.float32)
        arr = raw.reshape((n_fields, nz, ny, nx))
    return {
        "step": step, "t_fm": t_fm, "nx": nx, "ny": ny, "nz": nz,
        "extent": (xmin, xmax, ymin, ymax, zmin, zmax), "fields": arr
    }


def run_sim(tag, b_fm, ntest, cluster_mode, nfr=0.040, dump_tmunu=False, fo_csv=False, seed=4200):
    os.makedirs(OUT_DIR, exist_ok=True)
    fl_csv = os.path.join(OUT_DIR, f"{tag}_flow.csv")
    ts_csv = os.path.join(OUT_DIR, f"{tag}_ts.csv")
    cmd = [
        BIN_FP32, "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.2f}",
        "--ntest", str(ntest), "--steps", "220", "--dt", "0.30", "--substeps", "2",
        "--box", "48.0", "--grid", "24", "--eos", "md-soft", "--in-medium-ec", "on",
        "--cluster-mode", cluster_mode, "--nfr", f"{nfr:.4f}", "--freezeout-every", "2",
        "--seed", str(seed), "--csv", ts_csv, "--flow-csv", fl_csv,
    ]
    tmunu_dir = os.path.join(OUT_DIR, f"{tag}_tmunu")
    cells_csv = os.path.join(OUT_DIR, f"{tag}_cells.csv")
    if dump_tmunu:
        cmd += ["--dump-tmunu-dir", tmunu_dir, "--dump-tmunu-every", "10"]
    if fo_csv:
        cmd += ["--freezeout-cells-csv", cells_csv]
    if not os.path.exists(fl_csv + ".yields.csv"):
        print(f"Running {tag} (mode={cluster_mode}, b={b_fm}, nfr={nfr}, N_test={ntest})...", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    yields = {}
    with open(fl_csv + ".yields.csv", "r") as f:
        for r in csv.DictReader(f):
            yields[r["species"]] = float(r["yield_per_event"])
    return {"fl": fl_csv, "ut": fl_csv + ".ut0.csv", "yields": yields, "tmunu_dir": tmunu_dir, "cells": cells_csv}


def load_flow_y0(fl_csv, sp):
    y0, dn, v1, v1e = [], [], [], []
    with open(fl_csv, "r") as f:
        for r in csv.DictReader(f):
            if r["species"] == sp:
                y0.append(float(r["y0"]))
                dn.append(float(r["dN_dy0"]))
                v1.append(float(r["v1_ut08"]))
                v1e.append(float(r["v1_ut08_err"]))
    return np.array(y0), np.array(dn), np.array(v1), np.array(v1e)


def load_flow_ut0(ut_csv, sp):
    u, v2, v2e, nst = [], [], [], []
    with open(ut_csv, "r") as f:
        for r in csv.DictReader(f):
            if r["species"] == sp:
                u.append(float(r["ut0"]))
                v2.append(float(r["v2"]))
                v2e.append(float(r["v2_err"]))
                nst.append(float(r["n_stat"]))
    return np.array(u), np.array(v2), np.array(v2e), np.array(nst)


def run_all_benchmarks():
    # 1. Central Au+Au @ 250 AMeV (b = 1.4 fm, FOPI b0 < 0.15) for cluster yield benchmark
    cen_mic = run_sim("cen_micro", 1.4, 300, "microscopic", seed=7101)
    cen_fo3 = run_sim("cen_nfr053", 1.4, 300, "freezeout", nfr=0.0533, seed=7101)
    cen_fo4 = run_sim("cen_nfr040", 1.4, 300, "freezeout", nfr=0.0400, seed=7101)
    cen_fo6 = run_sim("cen_nfr027", 1.4, 300, "freezeout", nfr=0.0267, seed=7101)

    # 2. Semicentral Au+Au @ 250 AMeV (b = 4.8 fm, FOPI 0.25 < b0 < 0.45):
    #    Compare Microscopic vs Freeze-Out (n_fr = rho_0/4 = 0.040 fm^-3) + export HCGF T^munu & cells!
    sem_mic = run_sim("sem_micro", 4.8, 500, "microscopic", seed=8202)
    sem_fo4 = run_sim("sem_nfr040", 4.8, 500, "freezeout", nfr=0.0400,
                      dump_tmunu=True, fo_csv=True, seed=8202)
    return cen_mic, cen_fo3, cen_fo4, cen_fo6, sem_mic, sem_fo4


def plot_all():
    with open(FOPI_JSON, "r") as f:
        fopi = json.load(f)

    cen_mic, cen_fo3, cen_fo4, cen_fo6, sem_mic, sem_fo4 = run_all_benchmarks()

    plt.rcParams.update({
        "font.size": 9.3, "axes.titlesize": 10.3, "axes.labelsize": 9.3,
        "legend.fontsize": 7.8, "figure.facecolor": "white"
    })
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), dpi=180)
    fig.suptitle(
        "Microscopic Dynamical Clusters vs. Pure $N+\\Delta+\\pi$ + HCGF $T^{\\mu\\nu}$ Grid Freeze-Out ($n_B < n_{\\mathrm{fr}}$) — FOPI Au+Au @ 250 $A$ MeV",
        fontsize=12.6, fontweight="bold", y=0.98
    )

    # (a) HCGF T^munu reaction-plane slice (y ~ 0, t = 27 fm/c): thermal energy density & Landau flow
    ax = axes[0, 0]
    tm_snap = read_hcgf_tmunu_file(os.path.join(sem_fo4["tmunu_dir"], "tmunu_grid_0009.bin"))
    flds = tm_snap["fields"]
    ny_mid = tm_snap["ny"] // 2
    eps_lrf = 0.5 * (flds[18, :, ny_mid - 1, :] + flds[18, :, ny_mid, :])  # (nz, nx)
    nb_lrf  = 0.5 * (flds[20, :, ny_mid - 1, :] + flds[20, :, ny_mid, :])
    vx_l    = 0.5 * (flds[21, :, ny_mid - 1, :] + flds[21, :, ny_mid, :])
    vz_l    = 0.5 * (flds[23, :, ny_mid - 1, :] + flds[23, :, ny_mid, :])
    e_exc_mev = np.maximum(0.0, (eps_lrf - 0.9389 * nb_lrf) * 1e3)

    xmin, xmax, _, _, zmin, zmax = tm_snap["extent"]
    im = ax.imshow(e_exc_mev.T, origin="lower", extent=[zmin, zmax, xmin, xmax],
                   cmap="inferno", aspect="auto", interpolation="bilinear")
    zs = np.linspace(zmin + 1.0, zmax - 1.0, tm_snap["nz"])
    xs = np.linspace(xmin + 1.0, xmax - 1.0, tm_snap["nx"])
    ZZ, XX = np.meshgrid(zs, xs)
    ax.contour(ZZ, XX, nb_lrf.T, levels=[0.040, 0.160], colors=["#00ffff", "#ffffff"], linewidths=[1.6, 1.2], linestyles=["--", "-"])
    st = 2
    mask_q = nb_lrf.T[::st, ::st] > 0.008
    ax.quiver(ZZ[::st, ::st][mask_q], XX[::st, ::st][mask_q],
              vz_l.T[::st, ::st][mask_q], vx_l.T[::st, ::st][mask_q],
              color="white", alpha=0.80, scale=2.6, width=0.004)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("$\\varepsilon_{\\mathrm{LRF}} - m_N n_B$ [$\\mathrm{MeV/fm^3}$]")
    ax.set_xlim(-18, 18); ax.set_ylim(-16, 16)
    ax.set_xlabel("Beam Axis $z$ [fm]"); ax.set_ylabel("Reaction Plane $x$ [fm]")
    ax.set_title(f"(a) HCGF $T^{{\\mu\\nu}}$ Slice ($t={tm_snap['t_fm']:.0f}$ fm/$c$; cyan: $n_{{\\mathrm{{fr}}}}$)")

    # (b) Freeze-out hypersurface cells (T vs n_B and T vs mu_B) recorded at n_B < n_fr
    ax = axes[0, 1]
    fo_T, fo_muB, fo_t, fo_b = [], [], [], []
    with open(sem_fo4["cells"], "r") as f:
        for r in csv.DictReader(f):
            if int(r["baryons_converted"]) >= 4:
                fo_T.append(float(r["T_mev"]))
                fo_muB.append(float(r["muB_mev"]))
                fo_t.append(float(r["t_fm"]))
                fo_b.append(int(r["baryons_converted"]))
    sc = ax.scatter(fo_muB, fo_T, c=fo_t, s=np.clip(np.array(fo_b) * 0.8, 10, 90),
                    cmap="viridis", alpha=0.78, edgecolors="none")
    cb2 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb2.set_label("Freeze-Out Time $t_{\\mathrm{fr}}$ [fm/$c$]")
    ax.set_xlabel("Cell Baryon Chemical Potential $\\mu_B$ [MeV]")
    ax.set_ylabel("Cell Freeze-Out Temperature $T_{\\mathrm{fr}}$ [MeV]")
    ax.grid(alpha=0.25)
    ax.set_title("(b) HCGF $n_B < n_{\\mathrm{fr}}$ Freeze-Out Hypersurface $(T, \\mu_B)$")

    # (c) Central Au+Au (b0 < 0.15) cluster yields: Microscopic vs Thermal n_fr scan vs FOPI
    ax = axes[0, 2]
    sp_list = ["p", "d", "t", "3He", "4He"]
    sp_lbls = ["$p$", "$d$", "$t$", "$^3\\mathrm{He}$", "$^4\\mathrm{He}$"]
    x_pos = np.arange(len(sp_list))
    w = 0.17
    f_vals = [fopi["central_yields_b0_lt_015"][s]["val"] for s in sp_list]
    f_errs = [fopi["central_yields_b0_lt_015"][s]["err"] for s in sp_list]

    ax.bar(x_pos - 1.5 * w, [cen_mic["yields"][s] for s in sp_list], width=w, color="#1f77b4", label="Microscopic BUU")
    ax.bar(x_pos - 0.5 * w, [cen_fo3["yields"][s] for s in sp_list], width=w, color="#f39c12", label="Thermal $n_{\\mathrm{fr}}=\\rho_0/3$")
    ax.bar(x_pos + 0.5 * w, [cen_fo4["yields"][s] for s in sp_list], width=w, color="#27ae60", label="Thermal $n_{\\mathrm{fr}}=\\rho_0/4$")
    ax.bar(x_pos + 1.5 * w, [cen_fo6["yields"][s] for s in sp_list], width=w, color="#8e44ad", label="Thermal $n_{\\mathrm{fr}}=\\rho_0/6$")
    ax.errorbar(x_pos, f_vals, yerr=f_errs, fmt="ko", ms=6.0, capsize=3.5, lw=1.6, zorder=6, label="FOPI Data ($b_0<0.15$)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 180.0)
    ax.set_xticks(x_pos); ax.set_xticklabels(sp_lbls)
    ax.set_ylabel("Freeze-Out Multiplicity / Event")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=7.3)
    ax.set_title("(c) Central Cluster Yields: Microscopic vs. $n<n_{\\mathrm{fr}}$ vs. FOPI")

    # (d) Semicentral dN/dy0 comparison: Microscopic (solid) vs Thermal n_fr = rho_0/4 (dashed)
    ax = axes[1, 0]
    for sp, col, lbl in [("p", "#2c3e50", "$p$"), ("d", "#1f77b4", "$d$"),
                         ("A3", "#27ae60", "$t + {}^3\\mathrm{He}$"), ("4He", "#d62728", "$^4\\mathrm{He} \\times 5$")]:
        mult = 5.0 if sp == "4He" else 1.0
        y_m, dn_m, _, _ = load_flow_y0(sem_mic["fl"], sp)
        y_f, dn_f, _, _ = load_flow_y0(sem_fo4["fl"], sp)
        ax.plot(y_m, dn_m * mult, "-", color=col, lw=2.1, label=f"{lbl} (Micro)")
        ax.plot(y_f, dn_f * mult, "--", color=col, lw=2.0, alpha=0.85, label=f"{lbl} ($n<n_{{\\mathrm{{fr}}}}$)")
    ax.set_xlim(-1.40, 1.40)
    ax.set_xlabel("Normalized CM Rapidity $y_0 = y / y_{\\mathrm{proj}}$")
    ax.set_ylabel("$dN / dy_0$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=7.0)
    ax.set_title("(d) Rapidity Spectra $dN/dy_0$: Microscopic vs. $n<n_{\\mathrm{fr}}$")

    # (e) Directed flow v1(y0) (u_t0 > 0.8) vs FOPI data (NPA 876, 2012)
    ax = axes[1, 1]
    for sp, col, mk, lbl in [("p", "#2c3e50", "o", "$p$"), ("d", "#1f77b4", "s", "$d$"), ("t", "#d62728", "^", "$t$")]:
        y_m, _, v1_m, _ = load_flow_y0(sem_mic["fl"], sp)
        y_f, _, v1_f, _ = load_flow_y0(sem_fo4["fl"], sp)
        sel = np.abs(y_m) <= 1.08
        ax.plot(y_m[sel], v1_m[sel], "-", color=col, lw=2.0, label=f"{lbl} Micro")
        ax.plot(y_f[sel], v1_f[sel], "--", color=col, lw=2.0, label=f"{lbl} $n<n_{{\\mathrm{{fr}}}}$")
        exp_pts = fopi["v1_y0_ut0_gt_08"][sp]
        ax.errorbar([p["y0"] for p in exp_pts[::2]], [p["v1"] for p in exp_pts[::2]],
                    yerr=[p["err"] for p in exp_pts[::2]], fmt=mk, color=col, ms=4.8,
                    mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9); ax.axvline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.72, 0.72)
    ax.set_xlabel("Normalized CM Rapidity $y_0$ ($u_{t0} > 0.8$)")
    ax.set_ylabel("Directed Flow $v_1(y_0)$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=6.7)
    ax.set_title("(e) Directed Flow $v_1(y_0)$: Microscopic vs. $n<n_{\\mathrm{fr}}$ vs. FOPI")

    # (f) Differential elliptic flow v2(u_t0) (|y0| < 0.4) vs FOPI data
    ax = axes[1, 2]
    for sp, exp_key, col, mk, lbl in [("p", "p", "#2c3e50", "o", "$p$"),
                                      ("d", "d", "#1f77b4", "s", "$d$"),
                                      ("A3", "A3", "#d62728", "^", "$A=3$")]:
        u_m, v2_m, e_m, n_m = load_flow_ut0(sem_mic["ut"], sp)
        u_f, v2_f, e_f, n_f = load_flow_ut0(sem_fo4["ut"], sp)
        sm = (u_m >= 0.40) & (u_m <= 1.70) & (n_m >= 35)
        sf = (u_f >= 0.40) & (u_f <= 1.70) & (n_f >= 35)
        ax.plot(u_m[sm], v2_m[sm], "-", color=col, lw=2.0, label=f"{lbl} Micro")
        ax.plot(u_f[sf], v2_f[sf], "--", color=col, lw=2.0, label=f"{lbl} $n<n_{{\\mathrm{{fr}}}}$")
        v2_dict = fopi["v2_ut0_y0_lt_04"]
        exp_u = v2_dict.get(exp_key, v2_dict.get("t", []))
        if exp_u:
            ax.errorbar([p["ut0"] for p in exp_u[::2]], [p["v2"] for p in exp_u[::2]],
                        yerr=[p["err"] for p in exp_u[::2]], fmt=mk, color=col, ms=4.8,
                        mfc="none", mew=1.3, capsize=2, label=f"{lbl} FOPI")
    ax.axhline(0, color="#777", ls=":", lw=0.9)
    ax.set_xlim(0.30, 1.85); ax.set_ylim(-0.16, 0.05)
    ax.set_xlabel("Scaled Transverse 4-Velocity $u_{t0}$ ($|y_0| < 0.4$)")
    ax.set_ylabel("Elliptic Squeeze-Out $v_2(u_{t0})$")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", ncol=3, fontsize=6.7)
    ax.set_title("(f) Elliptic Flow $v_2(u_{t0})$: Microscopic vs. $n<n_{\\mathrm{fr}}$ vs. FOPI")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(ASSET_PNG), exist_ok=True)
    fig.savefig(ASSET_PNG, dpi=180)
    print(f"\nSaved comparison figure -> {ASSET_PNG}")
    for name, res in [("Microscopic", cen_mic), ("Thermal rho0/3", cen_fo3),
                      ("Thermal rho0/4", cen_fo4), ("Thermal rho0/6", cen_fo6)]:
        y = res["yields"]
        print(f"  {name:16s} : p={y['p']:6.2f}, d={y['d']:6.2f}, t={y['t']:6.2f}, "
              f"3He={y['3He']:5.2f}, 4He={y['4He']:5.2f}, pi-={y['pi-']:5.2f}")


if __name__ == "__main__":
    plot_all()
