#!/usr/bin/env python3
"""
Generate animations & EOS/optical-potential figures for feat/skyrme-momentum-meanfield:
  1. assets/spinodal_meanfield_vs_free.gif (.mp4) — Spinodal clumping & cluster droplets
  2. assets/eos_and_optical_potential.png         — E/A(rho/rho0) & U_opt(rho0, p)
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
BUILD_BIN = REPO_ROOT / "build" / "cluster_transport"
ASSETS_DIR = REPO_ROOT / "assets"
TMP_DIR = Path("/tmp/meanfield_anim_work")

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

HEADER_FMT = "<Iiii dddd QQQQ iiiiii dddddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PARTICLE_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"),
    ("E", "<f4"), ("mass", "<f4"),
    ("species", "<i2"), ("collided", "<i2"),
])


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


def load_frames(dump_dir: Path):
    files = sorted(glob.glob(str(dump_dir / "frame_*.bin")))
    frames = []
    for fp in files:
        with open(fp, "rb") as f:
            vals = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
            hdr = {
                "step": vals[1], "n_saved": vals[2], "n_total": vals[3],
                "t_fm": vals[4], "E_tot": vals[5], "aniso": vals[6], "rms_r": vals[7],
                "el_cum": vals[8], "inel_cum": vals[9],
                "form_cum": vals[10], "decay_cum": vals[11],
                "c_nuc": vals[12], "c_d": vals[13], "c_dstar": vals[14],
                "c_a3": vals[15], "c_he4": vals[16], "c_a4star": vals[17],
                "xmin": vals[18], "xmax": vals[19],
                "ymin": vals[20], "ymax": vals[21],
                "zmin": vals[22], "zmax": vals[23],
            }
            pdata = np.frombuffer(f.read(), dtype=PARTICLE_DTYPE)
            frames.append((hdr, pdata))
    return frames


def draw_wireframe_box(ax, L):
    for x in [0.0, L]:
        for y in [0.0, L]:
            ax.plot([x, x], [y, y], [0.0, L], color="#334155", lw=0.85, alpha=0.65)
        for z in [0.0, L]:
            ax.plot([x, x], [0.0, L], [z, z], color="#334155", lw=0.85, alpha=0.65)
    for y in [0.0, L]:
        for z in [0.0, L]:
            ax.plot([0.0, L], [y, y], [z, z], color="#334155", lw=0.85, alpha=0.65)


def encode_gif_and_mp4(png_dir: Path, out_stem: Path, fps: int = 18):
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


def render_3d_particles(ax, p, L, title_str, elev, azim):
    ax.set_facecolor(BG_DARK)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRID_COL)
    ax.grid(False)
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_zlim(0.0, L)
    ax.tick_params(labelsize=7, pad=1)
    ax.set_xlabel("x [fm]", fontsize=8, labelpad=1)
    ax.set_ylabel("y [fm]", fontsize=8, labelpad=1)
    ax.set_zlabel("z [fm]", fontsize=8, labelpad=1)
    draw_wireframe_box(ax, L)

    sp = p["species"]
    p_nuc = p[(sp == 0) | (sp == 1)][::4]
    p_d = p[sp == 2][::2]
    p_ds = p[(sp >= 3) & (sp <= 5)][::2]
    p_a3 = p[(sp == 6) | (sp == 7)]
    p_he4 = p[sp == 8]
    p_a4s = p[sp >= 9]

    if len(p_nuc) > 0:
        ax.scatter(p_nuc["x"], p_nuc["y"], p_nuc["z"], s=4.5, c=COL_NUC, alpha=0.45, label="p, n")
    if len(p_d) > 0:
        ax.scatter(p_d["x"], p_d["y"], p_d["z"], s=14.0, c=COL_DEUT, alpha=0.88, label="d")
    if len(p_ds) > 0:
        ax.scatter(p_ds["x"], p_ds["y"], p_ds["z"], s=11.0, c=COL_DSTAR, alpha=0.78, label="d*")
    if len(p_a3) > 0:
        ax.scatter(p_a3["x"], p_a3["y"], p_a3["z"], s=24.0, c=COL_A3, alpha=0.95, label="t, 3He")
    if len(p_he4) > 0:
        ax.scatter(p_he4["x"], p_he4["y"], p_he4["z"], s=40.0, c=COL_HE4, edgecolors="#ffffff",
                   linewidths=0.4, alpha=0.98, label="4He")
    if len(p_a4s) > 0:
        ax.scatter(p_a4s["x"], p_a4s["y"], p_a4s["z"], s=28.0, c=COL_A4STAR, marker="*", alpha=0.95, label="4He*")

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title_str, pad=4, fontsize=10)


def make_spinodal_comparison_animation():
    print("[1/2] Simulating Free Cascade vs MD-Soft Skyrme Spinodal Clumping + Nucleosynthesis...")
    free_dir = TMP_DIR / "free_dump"
    md_dir = TMP_DIR / "md_dump"
    png_dir = TMP_DIR / "spin_png"
    free_csv = TMP_DIR / "free.csv"
    md_csv = TMP_DIR / "md.csv"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--mode", "nucleosynthesis", "--reactions", "all",
        "--npart", "18000", "--ntest", "30",
        "--box", "24.0", "--grid", "12", "--temp", "6.0",
        "--steps", "380", "--dt", "0.38", "--substeps", "2",
        "--dump-every", "5", "--dump-limit", "20000",
    ]
    subprocess.run([str(BUILD_BIN), "--eos", "none", "--csv", str(free_csv),
                    "--dump-dir", str(free_dir)] + common_args, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(BUILD_BIN), "--eos", "md-soft", "--csv", str(md_csv),
                    "--dump-dir", str(md_dir)] + common_args, check=True, stdout=subprocess.DEVNULL)

    f_frames = load_frames(free_dir)
    m_frames = load_frames(md_dir)
    n_fr = min(len(f_frames), len(m_frames))

    # Compute baryon density contrast & A=4 multiplicity directly on every frame
    t_arr, con_free, con_md, a4_free, a4_md = [], [], [], [], []
    L = 24.0
    bins_3d = np.linspace(0.0, L, 13)
    bins_2d = np.linspace(0.0, L, 25)
    baryon_map = np.array([1, 1, 2, 2, 2, 2, 3, 3, 4] + [4] * 19)

    for k in range(n_fr):
        hf, pf = f_frames[k]
        hm, pm = m_frames[k]
        t_arr.append(hm["t_fm"])
        wf = baryon_map[pf["species"]]
        wm = baryon_map[pm["species"]]
        gf, _ = np.histogramdd((pf["x"], pf["y"], pf["z"]), bins=(bins_3d, bins_3d, bins_3d), weights=wf)
        gm, _ = np.histogramdd((pm["x"], pm["y"], pm["z"]), bins=(bins_3d, bins_3d, bins_3d), weights=wm)
        con_free.append(float(np.std(gf) / max(np.mean(gf), 1e-9)))
        con_md.append(float(np.std(gm) / max(np.mean(gm), 1e-9)))
        a4_free.append(hf["c_he4"] + hf["c_a4star"])
        a4_md.append(hm["c_he4"] + hm["c_a4star"])

    t_arr = np.array(t_arr)
    con_free = np.array(con_free)
    con_md = np.array(con_md)
    a4_free = np.array(a4_free)
    a4_md = np.array(a4_md)

    print(f"  Rendering {n_fr} comparison frames (contrast: free={con_free[-1]:.2f} vs MD-Skyrme={con_md[-1]:.2f})...")
    for idx in range(n_fr):
        hf, pf = f_frames[idx]
        hm, pm = m_frames[idx]
        fig = plt.figure(figsize=(13.6, 5.8), dpi=110)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 0.95], hspace=0.38, wspace=0.22,
                              left=0.015, right=0.93, top=0.87, bottom=0.09)
        fig.suptitle(
            f"Spinodal Liquid–Gas Clumping & Cluster Formation (ρ_B = 0.27 ρ_0, T_0 = 6 MeV)   |   t = {hm['t_fm']:5.1f} fm/c",
            fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        ax_free = fig.add_subplot(gs[:, 0], projection="3d")
        render_3d_particles(ax_free, pf, L, "Free Cascade (U = 0) — Uniform Vapor",
                            elev=20.0, azim=28.0 + 0.55 * idx)
        ax_free.legend(loc="upper left", fontsize=7.0, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=3)

        ax_md = fig.add_subplot(gs[:, 1], projection="3d")
        render_3d_particles(ax_md, pm, L, "MD-Skyrme (K=215, m*/m=0.69) — Spinodal Droplets",
                            elev=20.0, azim=28.0 + 0.55 * idx)

        # Top-Right: Column-averaged baryon density projection rho_B(x, y) / rho_0 of MD-Skyrme
        ax_sl = fig.add_subplot(gs[0, 2])
        wm = baryon_map[pm["species"]]
        h2d, _, _ = np.histogram2d(pm["x"], pm["y"], bins=(bins_2d, bins_2d), weights=wm)
        rho_proj = h2d.T / (30.0 * (L / 24.0) * (L / 24.0) * L * 0.16)
        im = ax_sl.imshow(rho_proj, origin="lower", extent=[0, L, 0, L],
                          cmap="inferno", vmin=0.05, vmax=0.65, aspect="equal", interpolation="bilinear")
        cb = fig.colorbar(im, ax=ax_sl, fraction=0.046, pad=0.03)
        cb.set_label(r"$\bar{\rho}_B(x,y) / \rho_0$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax_sl.set_xlabel("x [fm]", fontsize=8)
        ax_sl.set_ylabel("y [fm]", fontsize=8)
        ax_sl.set_title("MD-Skyrme Column Density Map", fontsize=9.8)

        # Bottom-Right: Density contrast sigma(rho)/<rho> & A=4 cluster multiplicity comparison
        ax_c = fig.add_subplot(gs[1, 2])
        ax_c.plot(t_arr, con_free, color="#94a3b8", lw=1.2, alpha=0.28)
        ax_c.plot(t_arr[:idx+1], con_free[:idx+1], color="#94a3b8", lw=1.8, ls="--", label="Contrast (Free)")
        ax_c.plot(t_arr, con_md, color="#fde047", lw=1.2, alpha=0.25)
        ax_c.plot(t_arr[:idx+1], con_md[:idx+1], color="#fde047", lw=2.3, label="Contrast (MD-Skyrme)")
        ax_c.axvline(hm["t_fm"], color=TEXT_MAIN, ls=":", lw=0.9, alpha=0.7)
        ax_c.set_xlim(t_arr[0], t_arr[-1])
        ax_c.set_ylim(0.0, max(0.95, 1.12 * np.max(con_md)))
        ax_c.set_xlabel("Time t [fm/c]", fontsize=8.5)
        ax_c.set_ylabel(r"$\sigma(\rho_B) / \langle\rho_B\rangle$", color="#fde047", fontsize=8.5)
        ax_c.grid(True)
        ax_c.legend(loc="upper left", fontsize=7.2, facecolor=PANEL_BG, edgecolor=GRID_COL)
        ax_c.set_title(f"A=4 Clusters: MD={a4_md[idx]} vs Free={a4_free[idx]}", fontsize=9.8)

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "spinodal_meanfield_vs_free", fps=18)


def make_eos_and_optical_plot():
    print("[2/2] Generating Cold Nuclear EOS & Optical Potential Plot (eos_and_optical_potential.png)...")
    eos_csv = TMP_DIR / "eos_curves.csv"
    subprocess.run([str(BUILD_BIN), "--dump-eos", str(eos_csv)], check=True)

    cols = {k: [] for k in [
        "u_rho_over_rho0", "ea_skyrme_soft_mev", "ea_md_soft_mev", "ea_md_stiff_mev",
        "p_gev", "uopt_skyrme_soft_mev", "uopt_md_soft_mev", "uopt_md_stiff_mev"
    ]}
    with open(eos_csv, "r") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    for k in cols:
        cols[k] = np.array(cols[k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.13, wspace=0.22)
    fig.suptitle(
        "Skyrme + Lorentzian Momentum-Dependent Mean Field: Nuclear EOS & Optical Potential",
        fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
    )

    u = cols["u_rho_over_rho0"]
    ax1.axvspan(0.05, 0.62, color="#fb7185", alpha=0.12, label="Spinodal Zone (dP/dρ < 0)")
    ax1.plot(u, cols["ea_skyrme_soft_mev"], color="#94a3b8", lw=2.0, ls="--", label="Skyrme-Soft (K₀ = 200 MeV, m*/m = 1.0)")
    ax1.plot(u, cols["ea_md_soft_mev"], color="#38bdf8", lw=2.3, label="MD-Soft (K₀ = 215 MeV, m*/m = 0.69)")
    ax1.plot(u, cols["ea_md_stiff_mev"], color="#fbbf24", lw=2.3, label="MD-Stiff (K₀ = 380 MeV, m*/m = 0.69)")
    ax1.scatter([1.0], [-16.0], c="#4ade80", s=55, zorder=5, label="Saturation (ρ₀ = 0.16 fm⁻³, -16 MeV)")
    ax1.axhline(0.0, color=TEXT_DIM, ls=":", lw=0.9)
    ax1.set_xlim(0.0, 3.2)
    ax1.set_ylim(-22.0, 75.0)
    ax1.set_xlabel(r"Reduced Baryon Density $\rho_B / \rho_0$")
    ax1.set_ylabel(r"Energy per Nucleon $E/A - m_N$ [MeV]")
    ax1.set_title("Cold Symmetric Nuclear Matter EOS")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.0, facecolor=PANEL_BG, edgecolor=GRID_COL)

    p = cols["p_gev"]
    ax2.plot(p, cols["uopt_skyrme_soft_mev"], color="#94a3b8", lw=2.0, ls="--", label="Local Skyrme-Soft (momentum-indep.)")
    ax2.plot(p, cols["uopt_md_soft_mev"], color="#e879f9", lw=2.4, label="GBD / Danielewicz MD Potential (ρ = ρ₀)")
    ax2.axhline(0.0, color="#4ade80", ls=":", lw=1.1, label="Zero Crossing (p ≈ 0.78 GeV/c, T ≈ 285 MeV)")
    ax2.axvline(0.263, color="#38bdf8", ls="--", lw=1.0, alpha=0.75, label="Fermi Surface p_F = 0.263 GeV/c")
    ax2.set_xlim(0.0, 1.2)
    ax2.set_ylim(-80.0, 30.0)
    ax2.set_xlabel(r"Nucleon Momentum $p$ [GeV/$c$] at $\rho_B = \rho_0$")
    ax2.set_ylabel(r"Single-Nucleon Optical Potential $U_N(\rho_0, p)$ [MeV]")
    ax2.set_title("Lorentzian Momentum-Dependent Optical Potential")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8.0, facecolor=PANEL_BG, edgecolor=GRID_COL)

    out_png = ASSETS_DIR / "eos_and_optical_potential.png"
    fig.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)
    print(f"  -> Wrote {out_png.name} ({out_png.stat().st_size / 1e3:.1f} KB)")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()
    make_spinodal_comparison_animation()
    make_eos_and_optical_plot()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Generated mean-field assets in:", ASSETS_DIR)


if __name__ == "__main__":
    main()
