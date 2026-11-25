#!/usr/bin/env python3
"""
Read binary 3D hydro grids (CHYD), particle trajectory snapshots (CTR2), and FOPI
flow CSV from Au + Au @ 250 A MeV (b = 5.5 fm, MD-Soft) to generate:
  1. assets/auau_250amev_collision.gif (.mp4) — 3D HIC + n_B/rho_0 + eps_LRF slices
  2. assets/fopi_auau250_flow.png             — dN/dy^(0), v1(y^(0)), and v2(y^(0))
"""

import csv
import glob
import os
import shutil
import struct
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_cluster_transport")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output"
ASSETS_DIR = REPO_ROOT / "assets"
TMP_PNG = Path("/tmp/auau250_png_work")

BG_DARK = "#0b0f19"
PANEL_BG = "#111827"
GRID_COL = "#243049"
TEXT_MAIN = "#e5e7eb"
TEXT_DIM = "#9ca3af"

COL_NUC = "#38bdf8"
COL_DEUT = "#fbbf24"
COL_DSTAR = "#e879f9"
COL_A3 = "#34d399"
COL_HE4 = "#fb923c"
COL_A4STAR = "#fb7185"

HYDRO_HDR_FMT = "<Iiiiiiii d dddddd"
HYDRO_HDR_SIZE = struct.calcsize(HYDRO_HDR_FMT)

FRAME_HDR_FMT = "<Iiii dddd QQQQ iiiiii dddddd"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)
PARTICLE_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"),
    ("E", "<f4"), ("mass", "<f4"),
    ("species", "<i2"), ("collided", "<i2"),
])


def read_hydro_grid(path: str):
    """Read a self-describing binary CHYD 3D local energy & baryon density snapshot."""
    with open(path, "rb") as f:
        v = struct.unpack(HYDRO_HDR_FMT, f.read(HYDRO_HDR_SIZE))
        nx, ny, nz = v[3], v[4], v[5]
        nc = nx * ny * nz
        raw = np.frombuffer(f.read(), dtype="<f4").reshape((8, nz, ny, nx))
    hdr = {
        "step": v[2], "nx": nx, "ny": ny, "nz": nz, "t_fm": v[8],
        "xmin": v[9], "xmax": v[10], "ymin": v[11], "ymax": v[12], "zmin": v[13], "zmax": v[14],
    }
    fields = {
        "rho_B_lab": raw[0], "rho_B_lrf": raw[1], "rho_Q_lab": raw[2],
        "e_tot_lab": raw[3], "eps_lrf": raw[4],
        "vx": raw[5], "vy": raw[6], "vz": raw[7],
    }
    return hdr, fields


def read_particle_frame(path: str):
    with open(path, "rb") as f:
        vals = struct.unpack(FRAME_HDR_FMT, f.read(FRAME_HDR_SIZE))
        p = np.frombuffer(f.read(), dtype=PARTICLE_DTYPE)
    hdr = {
        "step": vals[1], "n_saved": vals[2], "n_total": vals[3],
        "t_fm": vals[4], "E_tot": vals[5], "aniso": vals[6], "rms_r": vals[7],
        "el_cum": vals[8], "inel_cum": vals[9], "form_cum": vals[10], "decay_cum": vals[11],
        "c_nuc": vals[12], "c_d": vals[13], "c_dstar": vals[14],
        "c_a3": vals[15], "c_he4": vals[16], "c_a4star": vals[17],
    }
    return hdr, p


def style_dark():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COL,
        "axes.labelcolor": TEXT_MAIN,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "text.color": TEXT_MAIN,
        "grid.color": GRID_COL,
        "grid.alpha": 0.45,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    })


def encode_gif_and_mp4(png_dir: Path, out_stem: Path, fps: int = 14):
    mp4_path = out_stem.with_suffix(".mp4")
    gif_path = out_stem.with_suffix(".gif")
    palette_path = png_dir / "palette.png"

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4_path)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"),
        "-vf", "palettegen=stats_mode=diff:max_colors=192", str(palette_path)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"), "-i", str(palette_path),
        "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0", str(gif_path)
    ], check=True)
    print(f"  -> Wrote {gif_path.name} ({gif_path.stat().st_size / 1e6:.2f} MB) & {mp4_path.name}")


def make_auau_collision_animation(hydro_dir: Path, frame_dir: Path):
    print("[1/2] Rendering Au+Au @ 250 AMeV 3D Collision + Binary Hydro Density Animation...")
    shutil.rmtree(TMP_PNG, ignore_errors=True)
    TMP_PNG.mkdir(parents=True, exist_ok=True)

    h_files = sorted(glob.glob(str(hydro_dir / "hydro_grid_*.bin")))
    p_files = sorted(glob.glob(str(frame_dir / "frame_*.bin")))
    n_fr = min(len(h_files), len(p_files))

    # Pre-scan peak proper baryon density n_B(t) / rho_0 & central excitation energy density
    t_arr, peak_nb_over_rho0, peak_eps_ex_mev = [], [], []
    mN = 0.9389187
    for k in range(n_fr):
        hh, hf = read_hydro_grid(h_files[k])
        t_arr.append(hh["t_fm"])
        peak_nb_over_rho0.append(float(np.max(hf["rho_B_lrf"]) / 0.16))
        eps_ex = np.maximum(0.0, (hf["eps_lrf"] - mN * hf["rho_B_lrf"]) * 1e3)
        peak_eps_ex_mev.append(float(np.max(eps_ex)))
    t_arr = np.array(t_arr)
    peak_nb_over_rho0 = np.array(peak_nb_over_rho0)
    peak_eps_ex_mev = np.array(peak_eps_ex_mev)

    for idx in range(n_fr):
        hh, hf = read_hydro_grid(h_files[idx])
        ph, p = read_particle_frame(p_files[idx])

        fig = plt.figure(figsize=(13.8, 5.9), dpi=110)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.16, 1.0, 0.96], hspace=0.40, wspace=0.33,
                              left=0.015, right=0.96, top=0.87, bottom=0.09)
        fig.suptitle(
            f"FOPI Au + Au @ 250 A MeV (b = 5.5 fm, MD-Soft)   |   t = {hh['t_fm']:5.1f} fm/c   |   max(n_B/ρ_0) = {peak_nb_over_rho0[idx]:4.2f}",
            fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        # Panel 1 (Left): 3D colliding Au+Au nuclei in reaction-plane perspective
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.set_facecolor(BG_DARK)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor(GRID_COL)
        ax3d.grid(False)
        lim = 20.0
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_xlabel("z_beam [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_ylabel("y_out [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_zlabel("x_react [fm]", fontsize=8.5, labelpad=1)
        ax3d.tick_params(labelsize=7, pad=1)

        # Plot (z, y, x) so beam axis z is horizontal and reaction-plane deflection x is vertical!
        sp = p["species"]
        p_nuc = p[(sp == 0) | (sp == 1)][::2]
        p_d = p[sp == 2]
        p_ds = p[(sp >= 3) & (sp <= 5)]
        p_a3 = p[(sp == 6) | (sp == 7)]
        p_a4 = p[sp >= 8]

        if len(p_nuc) > 0:
            ax3d.scatter(p_nuc["z"], p_nuc["y"], p_nuc["x"], s=4.0, c=COL_NUC, alpha=0.40, label="p, n")
        if len(p_d) > 0:
            ax3d.scatter(p_d["z"], p_d["y"], p_d["x"], s=13.0, c=COL_DEUT, alpha=0.88, label="d")
        if len(p_ds) > 0:
            ax3d.scatter(p_ds["z"], p_ds["y"], p_ds["x"], s=10.0, c=COL_DSTAR, alpha=0.75, label="d*")
        if len(p_a3) > 0:
            ax3d.scatter(p_a3["z"], p_a3["y"], p_a3["x"], s=22.0, c=COL_A3, alpha=0.94, label="t, 3He")
        if len(p_a4) > 0:
            ax3d.scatter(p_a4["z"], p_a4["y"], p_a4["x"], s=32.0, c=COL_HE4, marker="*", alpha=0.96, label="4He, 4He*")
        ax3d.view_init(elev=16.0, azim=-78.0 + 0.30 * idx)
        ax3d.set_title("3D Au + Au Collision & Spectator Bounce", pad=4)
        ax3d.legend(loc="upper left", fontsize=7.2, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=3)

        # Panel 2 (Top-Center): Reaction-plane (z, x) slice at y ~ 0 of Proper Baryon Density n_B / rho_0
        ax_nb = fig.add_subplot(gs[0, 1])
        ny_mid = hh["ny"] // 2
        nb_zx = 0.5 * (hf["rho_B_lrf"][:, ny_mid - 1, :] + hf["rho_B_lrf"][:, ny_mid, :]) / 0.16
        vz_zx = 0.5 * (hf["vz"][:, ny_mid - 1, :] + hf["vz"][:, ny_mid, :])
        vx_zx = 0.5 * (hf["vx"][:, ny_mid - 1, :] + hf["vx"][:, ny_mid, :])
        # nb_zx has shape (nz, nx); transpose -> (nx, nz) so horizontal=z, vertical=x!
        im1 = ax_nb.imshow(nb_zx.T, origin="lower",
                           extent=[hh["zmin"], hh["zmax"], hh["xmin"], hh["xmax"]],
                           cmap="magma", vmin=0.0, vmax=1.75, aspect="equal", interpolation="bilinear")
        cb1 = fig.colorbar(im1, ax=ax_nb, fraction=0.046, pad=0.03)
        cb1.set_label(r"$n_B / \rho_0$", fontsize=8)
        cb1.ax.tick_params(labelsize=7)

        zc = np.linspace(hh["zmin"] + 1.0, hh["zmax"] - 1.0, hh["nz"])[::2]
        xc = np.linspace(hh["xmin"] + 1.0, hh["xmax"] - 1.0, hh["nx"])[::2]
        ZZ, XX = np.meshgrid(zc, xc)
        sub_nb = nb_zx.T[::2, ::2]
        sub_vz = vz_zx.T[::2, ::2]
        sub_vx = vx_zx.T[::2, ::2]
        mask_q = sub_nb > 0.08
        if np.any(mask_q):
            ax_nb.quiver(ZZ[mask_q], XX[mask_q], sub_vz[mask_q], sub_vx[mask_q],
                         color="#e2e8f0", alpha=0.80, scale=2.6, width=0.005)
        ax_nb.set_xlabel("Beam z [fm]", fontsize=8)
        ax_nb.set_ylabel("Reaction Plane x [fm]", fontsize=8)
        ax_nb.set_title("Reaction-Plane Proper Baryon Density n_B/ρ_0")

        # Panel 3 (Bot-Center): Transverse midplane (x, y) at z ~ 0 of Local Energy Density eps_ex [MeV/fm^3]
        ax_ep = fig.add_subplot(gs[1, 1])
        nz_mid = hh["nz"] // 2
        eps_ex_xy = 0.5 * ((hf["eps_lrf"][nz_mid - 1, :, :] - mN * hf["rho_B_lrf"][nz_mid - 1, :, :]) +
                           (hf["eps_lrf"][nz_mid, :, :]     - mN * hf["rho_B_lrf"][nz_mid, :, :])) * 1e3
        eps_ex_xy = np.maximum(0.0, eps_ex_xy)  # shape (ny, nx): horizontal=x, vertical=y!
        im2 = ax_ep.imshow(eps_ex_xy, origin="lower",
                           extent=[hh["xmin"], hh["xmax"], hh["ymin"], hh["ymax"]],
                           cmap="inferno", vmin=0.0, vmax=65.0, aspect="equal", interpolation="bilinear")
        cb2 = fig.colorbar(im2, ax=ax_ep, fraction=0.046, pad=0.03)
        cb2.set_label(r"$\varepsilon_{\rm ex}$ [MeV/fm³]", fontsize=8.0)
        cb2.ax.tick_params(labelsize=7)
        ax_ep.set_xlabel("In-Plane x [fm]", fontsize=8)
        ax_ep.set_ylabel("Out-of-Plane y [fm] (Squeeze-Out)", fontsize=8)
        ax_ep.set_title("Midplane Local Excitation Energy Density")

        # Panel 4 (Top-Right): Peak compression n_B/rho_0(t) & excitation energy density
        ax_pk = fig.add_subplot(gs[0, 2])
        ax_pk.plot(t_arr, peak_nb_over_rho0, color="#fbbf24", lw=1.2, alpha=0.25)
        ax_pk.plot(t_arr[:idx+1], peak_nb_over_rho0[:idx+1], color="#fbbf24", lw=2.2)
        ax_pk.axvline(hh["t_fm"], color=TEXT_MAIN, ls=":", lw=0.9, alpha=0.7)
        ax_pk.set_xlim(t_arr[0], t_arr[-1])
        ax_pk.set_ylim(0.0, max(2.0, 1.12 * np.max(peak_nb_over_rho0)))
        ax_pk.set_ylabel(r"Peak $n_B / \rho_0$", color="#fbbf24", fontsize=8.5)
        ax_pk.set_title("Baryon Compression Trajectory")
        ax_pk.grid(True)

        # Panel 5 (Bot-Right): Live cluster formation counts per Au+Au event
        ax_cl = fig.add_subplot(gs[1, 2])
        cats = ["d", "d*", "t+3He", "4He+4He*"]
        vals = np.array([ph["c_d"], ph["c_dstar"], ph["c_a3"], ph["c_he4"] + ph["c_a4star"]]) / 120.0
        bars = ax_cl.bar(cats, vals, color=[COL_DEUT, COL_DSTAR, COL_A3, COL_HE4], alpha=0.88)
        ax_cl.set_ylim(0.0, 62.0)
        ax_cl.set_ylabel("Multiplicity / Event", fontsize=8.5)
        ax_cl.set_title("Dynamic Cluster Yields / Event")
        ax_cl.grid(True, axis="y")
        for b_rect, v_num in zip(bars, vals):
            ax_cl.text(b_rect.get_x() + b_rect.get_width() * 0.5, v_num + 1.2,
                       f"{v_num:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT_MAIN)

        fig.savefig(TMP_PNG / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(TMP_PNG, ASSETS_DIR / "auau_250amev_collision", fps=12)
    shutil.rmtree(TMP_PNG, ignore_errors=True)


def make_fopi_flow_plot(flow_csv: Path):
    rows_by_sp = {}
    with open(flow_csv, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows_by_sp.setdefault(r["species"], []).append({
                "y0": float(r["y0"]), "dN_dy": float(r["dN_dy"]), "dN_dy0": float(r["dN_dy0"]),
                "v1": float(r["v1"]), "v1_err": float(r["v1_err"]),
                "v2": float(r["v2"]), "v2_err": float(r["v2_err"]), "n": int(r["n_stat"]),
            })

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.14, wspace=0.25)
    fig.suptitle(
        "FOPI Benchmark: Au + Au @ 250 A MeV (b = 5.5 fm, MD-Soft Skyrme + Lorentzian MD, Freeze-Out t = 72 fm/c)",
        fontsize=12.2, fontweight="bold", color=TEXT_MAIN, y=0.95
    )

    # Panel 1: dN / dy^(0)
    ax = axes[0]
    spec_styles = [
        ("n",   "n",          "#93c5fd", "--", "s", 1.0),
        ("p",   "p",          COL_NUC,   "-",  "o", 1.0),
        ("d",   "d",          COL_DEUT,  "-",  "^", 1.0),
        ("t",   "t",          COL_A3,    "-",  "v", 1.0),
        ("3He", "³He",        "#2dd4bf", "-.", "D", 1.0),
        ("4He", "⁴He (×5)",   COL_HE4,   "-",  "*", 5.0),
    ]
    for key, lbl, col, ls, mk, scale in spec_styles:
        arr = rows_by_sp.get(key, [])
        if not arr:
            continue
        y0 = np.array([x["y0"] for x in arr])
        dn = np.array([x["dN_dy0"] for x in arr]) * scale
        ax.plot(y0, dn, ls=ls, marker=mk, ms=4.5, lw=1.8, color=col, label=lbl)
    for y_beam in (-1.0, 1.0):
        ax.axvline(y_beam, color=TEXT_DIM, ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"Normalized CM Rapidity $y^{(0)} = y / y_{\rm beam}$")
    ax.set_ylabel(r"$dN / dy^{(0)}$")
    ax.set_title("Rapidity Distributions (Post-Decay)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.8, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=2)

    # Panel 2: Directed flow v1(y^(0)) — FOPI S-curve & cluster mass ordering
    ax = axes[1]
    v1_styles = [
        ("p",   "p (A=1)",        COL_NUC,  "o", 25),
        ("d",   "d (A=2)",        COL_DEUT, "^", 20),
        ("A3",  "t + ³He (A=3)",  COL_A3,   "s", 20),
        ("4He", "⁴He (A=4)",      COL_HE4,  "*", 12),
    ]
    ax.axhline(0.0, color=TEXT_DIM, lw=0.9, alpha=0.7)
    for y_beam in (-1.0, 1.0):
        ax.axvline(y_beam, color=TEXT_DIM, ls=":", lw=1.0, alpha=0.6)
    for key, lbl, col, mk, nmin in v1_styles:
        arr = [x for x in rows_by_sp.get(key, []) if x["n"] >= nmin]
        if not arr:
            continue
        ax.errorbar([x["y0"] for x in arr], [x["v1"] for x in arr], yerr=[x["v1_err"] for x in arr],
                    fmt=mk + "-", ms=4.8, lw=1.6, capsize=2.2, color=col, label=lbl, alpha=0.92)
    ax.set_ylim(-0.56, 0.56)
    ax.set_xlabel(r"Normalized CM Rapidity $y^{(0)} = y / y_{\rm beam}$")
    ax.set_ylabel(r"$v_1 = \langle p_x / p_T \rangle$")
    ax.set_title(r"Directed Flow $v_1(y^{(0)})$ Mass Ordering")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.0, facecolor=PANEL_BG, edgecolor=GRID_COL)

    # Panel 3: Elliptic flow v2(y^(0)) — Midrapidity out-of-plane squeeze-out (v2 < 0)
    ax = axes[2]
    ax.axhline(0.0, color=TEXT_DIM, lw=0.9, alpha=0.7)
    ax.axvspan(-0.45, 0.45, color="#38bdf8", alpha=0.08, label=r"Midrapidity $|y^{(0)}|<0.45$")
    for key, lbl, col, mk, nmin in v1_styles[:3]:
        arr = [x for x in rows_by_sp.get(key, []) if x["n"] >= 35 and abs(x["y0"]) <= 1.25]
        if not arr:
            continue
        ax.errorbar([x["y0"] for x in arr], [x["v2"] for x in arr], yerr=[x["v2_err"] for x in arr],
                    fmt=mk + "-", ms=4.8, lw=1.6, capsize=2.2, color=col, label=lbl, alpha=0.92)
    ax.set_ylim(-0.18, 0.16)
    ax.set_xlabel(r"Normalized CM Rapidity $y^{(0)} = y / y_{\rm beam}$")
    ax.set_ylabel(r"$v_2 = \langle (p_x^2 - p_y^2) / p_T^2 \rangle$")
    ax.set_title(r"Elliptic Flow $v_2(y^{(0)})$ (Squeeze-Out $v_2 < 0$)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.8, facecolor=PANEL_BG, edgecolor=GRID_COL)

    out_png = ASSETS_DIR / "fopi_auau250_flow.png"
    fig.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)
    print(f"[Generated] {out_png}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()
    make_auau_collision_animation(OUT_DIR / "auau250_hydro", OUT_DIR / "auau250_frames")
    make_fopi_flow_plot(OUT_DIR / "fopi_auau250_flow.csv")
