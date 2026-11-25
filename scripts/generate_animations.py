#!/usr/bin/env python3
"""
Generate publication-grade animations and physics overview plots for
cuda_cluster_transport from live simulation snapshots.
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
from matplotlib.patches import Ellipse
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO_ROOT / "build" / "cluster_transport"
ASSETS_DIR = REPO_ROOT / "assets"
TMP_DIR = Path("/tmp/cluster_transport_anim_work")

# Dark scientific palette
BG_DARK = "#0b0f19"
PANEL_BG = "#111827"
GRID_COL = "#243049"
TEXT_MAIN = "#e5e7eb"
TEXT_DIM = "#9ca3af"

COL_NUC = "#38bdf8"       # p, n (sky cyan)
COL_GS_CLUST = "#fbbf24"  # d, t, 3He, 4He (warm gold)
COL_DSTAR = "#e879f9"     # d*, nn*, pp* (neon magenta)
COL_HE4STAR = "#fb7185"   # 4He* (17 levels) + 4H*, 4Li* (coral crimson)
COL_SPARK = "#ffffff"     # Active collision vertex flash

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
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def load_frames(dump_dir: Path):
    files = sorted(glob.glob(str(dump_dir / "frame_*.bin")))
    frames = []
    for fp in files:
        with open(fp, "rb") as f:
            raw_hdr = f.read(HEADER_SIZE)
            vals = struct.unpack(HEADER_FMT, raw_hdr)
            hdr = {
                "magic": vals[0],
                "step": vals[1],
                "n_saved": vals[2],
                "n_total": vals[3],
                "t_fm": vals[4],
                "E_tot": vals[5],
                "aniso": vals[6],
                "rms_r": vals[7],
                "collisions": vals[8] + vals[9] + vals[10],
                "xmin": vals[18], "xmax": vals[19],
                "ymin": vals[20], "ymax": vals[21],
                "zmin": vals[22], "zmax": vals[23],
            }
            pdata = np.frombuffer(f.read(), dtype=PARTICLE_DTYPE)
            frames.append((hdr, pdata))
    return frames


def split_families(sp):
    m_nuc = (sp == 0) | (sp == 1)
    m_gs = (sp == 2) | (sp == 6) | (sp == 7) | (sp == 8)
    m_dstar = (sp >= 3) & (sp <= 5)
    m_a4star = sp >= 9
    return m_nuc, m_gs, m_dstar, m_a4star


def draw_wireframe_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, color="#334155", lw=0.9):
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    zs = [zmin, zmax]
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], zs, color=color, lw=lw, alpha=0.65)
    for x in xs:
        for z in zs:
            ax.plot([x, x], ys, [z, z], color=color, lw=lw, alpha=0.65)
    for y in ys:
        for z in zs:
            ax.plot(xs, [y, y], [z, z], color=color, lw=lw, alpha=0.65)


def encode_gif_and_mp4(png_dir: Path, out_stem: Path, fps: int = 20):
    mp4_path = out_stem.with_suffix(".mp4")
    gif_path = out_stem.with_suffix(".gif")
    palette_path = png_dir / "palette.png"

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(mp4_path)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"),
        "-vf", "palettegen=stats_mode=diff:max_colors=192",
        str(palette_path)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(png_dir / "img_%04d.png"),
        "-i", str(palette_path),
        "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0",
        str(gif_path)
    ], check=True)
    print(f"  -> Wrote {gif_path.name} ({gif_path.stat().st_size / 1e6:.2f} MB) & {mp4_path.name}")


def render_3d_species(ax, p, tail_dt=0.45):
    sp = p["species"]
    m_nuc, m_gs, m_dstar, m_a4star = split_families(sp)

    # Short velocity streaks for subset of particles to convey flying motion
    vx = p["px"] / p["E"]
    vy = p["py"] / p["E"]
    vz = p["pz"] / p["E"]

    stride_tail = max(1, len(p) // 350)
    idx_t = np.arange(0, len(p), stride_tail)
    for i in idx_t:
        c = COL_NUC if m_nuc[i] else (COL_GS_CLUST if m_gs[i] else (COL_DSTAR if m_dstar[i] else COL_HE4STAR))
        ax.plot(
            [p["x"][i] - vx[i] * tail_dt, p["x"][i]],
            [p["y"][i] - vy[i] * tail_dt, p["y"][i]],
            [p["z"][i] - vz[i] * tail_dt, p["z"][i]],
            color=c, lw=0.65, alpha=0.35
        )

    if np.any(m_nuc):
        ax.scatter(p["x"][m_nuc], p["y"][m_nuc], p["z"][m_nuc],
                   s=5.5, c=COL_NUC, alpha=0.58, edgecolors="none", label="p, n", depthshade=True)
    if np.any(m_gs):
        ax.scatter(p["x"][m_gs], p["y"][m_gs], p["z"][m_gs],
                   s=18.0, c=COL_GS_CLUST, alpha=0.90, edgecolors="none", label="d, t, 3He, 4He", depthshade=True)
    if np.any(m_dstar):
        ax.scatter(p["x"][m_dstar], p["y"][m_dstar], p["z"][m_dstar],
                   s=16.0, c=COL_DSTAR, alpha=0.88, edgecolors="none", label="d*(0+) triplet", depthshade=True)
    if np.any(m_a4star):
        ax.scatter(p["x"][m_a4star], p["y"][m_a4star], p["z"][m_a4star],
                   s=28.0, c=COL_HE4STAR, alpha=0.95, marker="*", edgecolors="none", label="4He* (19 states)", depthshade=True)

    # Highlight particles that underwent an elastic 2->2 collision in this step!
    m_col = p["collided"] > 0
    if np.any(m_col):
        ax.scatter(p["x"][m_col], p["y"][m_col], p["z"][m_col],
                   s=32.0, c="#fef08a", alpha=0.92, edgecolors=COL_SPARK, linewidths=0.5,
                   label="2->2 scatter", depthshade=False)


def style_3d_axis(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    ax.set_facecolor(BG_DARK)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COL)
    ax.yaxis.pane.set_edgecolor(GRID_COL)
    ax.zaxis.pane.set_edgecolor(GRID_COL)
    ax.grid(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x [fm]", labelpad=2, fontsize=9)
    ax.set_ylabel("y [fm]", labelpad=2, fontsize=9)
    ax.set_zlabel("z [fm]", labelpad=2, fontsize=9)
    ax.tick_params(labelsize=7, pad=1)


def make_colliding_slabs_animation():
    print("[1/4] Simulating & rendering Counter-Streaming Cluster Beams (colliding_slabs.gif)...")
    dump_dir = TMP_DIR / "slabs_dump"
    png_dir = TMP_DIR / "slabs_png"
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        str(BUILD_BIN),
        "--mode", "slabs",
        "--npart", "18000",
        "--ntest", "22",
        "--box", "24.0",
        "--grid", "12",
        "--temp", "32.0",
        "--cluster-boost", "5.0",
        "--excited-boost", "10.0",
        "--steps", "72",
        "--dt", "0.22",
        "--substeps", "2",
        "--ang", "iso",
        "--dump-dir", str(dump_dir),
        "--dump-every", "1",
        "--dump-limit", "1800",
    ], check=True, stdout=subprocess.DEVNULL)

    frames = load_frames(dump_dir)
    t_arr = np.array([h["t_fm"] for h, _ in frames])
    col_cum = np.array([h["collisions"] for h, _ in frames], dtype=float)
    dt = t_arr[1] - t_arr[0]
    col_rate = np.gradient(col_cum, dt)
    # Compute beam anisotropy P_x / P_perp(y,z) along beam axis x!
    aniso_beam = []
    for _, p_f in frames:
        pxx = np.sum(p_f["px"]**2 / p_f["E"])
        pperp = 0.5 * np.sum((p_f["py"]**2 + p_f["pz"]**2) / p_f["E"])
        aniso_beam.append(pxx / max(pperp, 1e-12))
    aniso_arr = np.array(aniso_beam)

    for idx, (hdr, p) in enumerate(frames):
        fig = plt.figure(figsize=(12.4, 5.6), dpi=110)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.24,
                              left=0.02, right=0.92, top=0.88, bottom=0.10)

        fig.suptitle(
            f"CUDA Relativistic Cluster Transport — Counter-Streaming Nuclear Slabs   |   t = {hdr['t_fm']:5.2f} fm/c",
            fontsize=13, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        # Left 3D view with beam axis x horizontal across screen
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        style_3d_axis(ax3d, hdr["xmin"], hdr["xmax"], hdr["ymin"], hdr["ymax"], hdr["zmin"], hdr["zmax"])
        draw_wireframe_box(ax3d, hdr["xmin"], hdr["xmax"], hdr["ymin"], hdr["ymax"], hdr["zmin"], hdr["zmax"])
        render_3d_species(ax3d, p, tail_dt=0.65)
        ax3d.view_init(elev=17.0, azim=-72.0 + 0.35 * idx)
        ax3d.set_title("3D Coordinate Space + Active 2→2 Collision Vertices", pad=4)
        ax3d.legend(loc="upper left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL,
                    markerscale=1.3, ncol=2)

        # Top-right: Momentum phase portrait (ux vs u_perp)
        ax_p = fig.add_subplot(gs[0, 1])
        sp = p["species"]
        m_nuc, m_gs, m_dstar, m_a4star = split_families(sp)
        u_beam = p["px"] / p["mass"]
        u_perp = np.sqrt(p["py"]**2 + p["pz"]**2) / p["mass"]

        ax_p.scatter(u_beam[m_nuc], u_perp[m_nuc], s=4, c=COL_NUC, alpha=0.35)
        ax_p.scatter(u_beam[m_gs], u_perp[m_gs], s=10, c=COL_GS_CLUST, alpha=0.78)
        ax_p.scatter(u_beam[m_dstar], u_perp[m_dstar], s=9, c=COL_DSTAR, alpha=0.75)
        ax_p.scatter(u_beam[m_a4star], u_perp[m_a4star], s=18, c=COL_HE4STAR, marker="*", alpha=0.92)
        ax_p.axvline(0.0, color=TEXT_DIM, ls="--", lw=0.8, alpha=0.5)
        ax_p.set_xlim(-1.15, 1.15)
        ax_p.set_ylim(0.0, 1.05)
        ax_p.grid(True)
        ax_p.set_xlabel(r"Beam 4-velocity $u_x = p_x / m$", fontsize=9)
        ax_p.set_ylabel(r"Transverse $u_\perp = \sqrt{p_y^2+p_z^2} / m$", fontsize=9)
        ax_p.set_title("Beam Stopping & Transverse Splash in 4-Velocity Space")

        # Bottom-right: Collision rate spike & anisotropy relaxation
        ax_t = fig.add_subplot(gs[1, 1])
        ax_t.plot(t_arr, col_rate / 1e3, color="#fde047", lw=1.6, alpha=0.30)
        ax_t.plot(t_arr[:idx+1], col_rate[:idx+1] / 1e3, color="#fde047", lw=2.2)
        ax_t.axvline(hdr["t_fm"], color=TEXT_MAIN, ls=":", lw=1.0, alpha=0.7)
        ax_t.set_xlim(t_arr[0], t_arr[-1])
        ax_t.set_ylim(0, max(0.5, 1.18 * np.max(col_rate / 1e3)))
        ax_t.set_xlabel("Time t [fm/c]", fontsize=9)
        ax_t.set_ylabel(r"dN$_{\rm coll}$/dt [$10^3$ c/fm]", color="#fde047", fontsize=9)
        ax_t.grid(True)

        ax_t2 = ax_t.twinx()
        ax_t2.plot(t_arr, aniso_arr, color="#38bdf8", lw=1.4, ls="--", alpha=0.28)
        ax_t2.plot(t_arr[:idx+1], aniso_arr[:idx+1], color="#38bdf8", lw=2.0)
        ax_t2.axhline(1.0, color="#4ade80", ls="--", lw=0.9, alpha=0.7)
        ax_t2.set_ylim(0.0, max(2.0, 1.08 * np.max(aniso_arr)))
        ax_t2.set_ylabel(r"Beam Anisotropy $P_x / P_\perp$", color="#38bdf8", fontsize=9)
        ax_t.set_title(f"Cumulative Elastic Scatters: {hdr['collisions']:,}")

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "colliding_slabs", fps=18)


def make_fireball_animation():
    print("[2/4] Simulating & rendering 3D Expanding Cluster Fireball (fireball_expansion.gif)...")
    dump_dir = TMP_DIR / "fireball_dump"
    png_dir = TMP_DIR / "fireball_png"
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        str(BUILD_BIN),
        "--mode", "fireball",
        "--npart", "24000",
        "--ntest", "12",
        "--box", "18.0",
        "--grid", "14",
        "--temp", "50.0",
        "--cluster-boost", "4.0",
        "--excited-boost", "9.0",
        "--steps", "68",
        "--dt", "0.24",
        "--substeps", "2",
        "--dump-dir", str(dump_dir),
        "--dump-every", "1",
        "--dump-limit", "2000",
    ], check=True, stdout=subprocess.DEVNULL)

    frames = load_frames(dump_dir)
    t_arr = np.array([h["t_fm"] for h, _ in frames])
    rms_arr = np.array([h["rms_r"] for h, _ in frames])
    col_arr = np.array([h["collisions"] for h, _ in frames], dtype=float) / 1e3

    r_bins = np.linspace(0.0, 22.0, 36)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    for idx, (hdr, p) in enumerate(frames):
        fig = plt.figure(figsize=(12.4, 5.6), dpi=110)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.24,
                              left=0.02, right=0.92, top=0.88, bottom=0.10)

        fig.suptitle(
            f"3D Expanding Heavy-Ion Hadronic & Cluster Fireball   |   t = {hdr['t_fm']:5.2f} fm/c   |   R_rms = {hdr['rms_r']:4.2f} fm",
            fontsize=13, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        lim = 16.5
        style_3d_axis(ax3d, -lim, lim, -lim, lim, -lim, lim)
        render_3d_species(ax3d, p, tail_dt=0.65)
        ax3d.view_init(elev=22.0, azim=30.0 + 0.9 * idx)
        ax3d.set_title("Hubble Expansion + Kinetic Freeze-Out in 3D", pad=4)
        ax3d.legend(loc="upper left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL,
                    markerscale=1.3, ncol=2)

        # Top-right: Radial distribution dN/dr by family
        ax_r = fig.add_subplot(gs[0, 1])
        r = np.sqrt(p["x"]**2 + p["y"]**2 + p["z"]**2)
        m_nuc, m_gs, m_dstar, m_a4star = split_families(p["species"])
        h_nuc, _ = np.histogram(r[m_nuc], bins=r_bins)
        h_clu, _ = np.histogram(r[~m_nuc], bins=r_bins)
        dr = r_bins[1] - r_bins[0]

        ax_r.fill_between(r_centers, h_nuc / dr, color=COL_NUC, alpha=0.28, label="Nucleons (p, n)")
        ax_r.plot(r_centers, h_nuc / dr, color=COL_NUC, lw=1.8)
        ax_r.fill_between(r_centers, h_clu / dr, color=COL_HE4STAR, alpha=0.32, label="Clusters (d, d*, t, 3He, 4He, 4He*)")
        ax_r.plot(r_centers, h_clu / dr, color=COL_HE4STAR, lw=1.8)
        ax_r.set_xlim(0.0, 22.0)
        ax_r.set_ylim(0.0, 240.0)
        ax_r.set_xlabel("Radius r [fm]", fontsize=9)
        ax_r.set_ylabel("Sampled dN / dr [fm$^{-1}$]", fontsize=9)
        ax_r.set_title("Outward Radial Wave & Cluster Flow")
        ax_r.legend(loc="upper right", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL)
        ax_r.grid(True)

        # Bottom-right: R_rms(t) & cumulative collisions freeze-out curve
        ax_c = fig.add_subplot(gs[1, 1])
        ax_c.plot(t_arr, rms_arr, color="#4ade80", lw=1.5, alpha=0.28)
        ax_c.plot(t_arr[:idx+1], rms_arr[:idx+1], color="#4ade80", lw=2.3, label=r"$R_{\rm rms}(t)$ [fm]")
        ax_c.axvline(hdr["t_fm"], color=TEXT_MAIN, ls=":", lw=1.0, alpha=0.7)
        ax_c.set_xlim(t_arr[0], t_arr[-1])
        ax_c.set_ylim(4.0, max(12.0, 1.08 * np.max(rms_arr)))
        ax_c.set_xlabel("Time t [fm/c]", fontsize=9)
        ax_c.set_ylabel(r"$R_{\rm rms}$ [fm]", color="#4ade80", fontsize=9)
        ax_c.grid(True)

        ax_c2 = ax_c.twinx()
        ax_c2.plot(t_arr, col_arr, color="#fbbf24", lw=1.5, ls="--", alpha=0.28)
        ax_c2.plot(t_arr[:idx+1], col_arr[:idx+1], color="#fbbf24", lw=2.1, label="Collisions")
        ax_c2.set_ylim(0.0, max(1.0, 1.12 * np.max(col_arr)))
        ax_c2.set_ylabel(r"Cumulative Scatters [$\times 10^3$]", color="#fbbf24", fontsize=9)
        ax_c.set_title(f"Freeze-Out Progress — Total Collisions: {hdr['collisions']:,}")

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "fireball_expansion", fps=18)


def make_box_isotropization_animation():
    print("[3/4] Simulating & rendering 3D Box Pressure Isotropization (box_isotropization.gif)...")
    dump_dir = TMP_DIR / "aniso_dump"
    png_dir = TMP_DIR / "aniso_png"
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        str(BUILD_BIN),
        "--mode", "aniso",
        "--npart", "20000",
        "--ntest", "38",
        "--box", "20.0",
        "--grid", "10",
        "--temp", "55.0",
        "--cluster-boost", "4.0",
        "--excited-boost", "8.0",
        "--steps", "70",
        "--dt", "0.25",
        "--substeps", "2",
        "--ang", "iso",
        "--dump-dir", str(dump_dir),
        "--dump-every", "1",
        "--dump-limit", "1500",
    ], check=True, stdout=subprocess.DEVNULL)

    frames = load_frames(dump_dir)
    t_arr = np.array([h["t_fm"] for h, _ in frames])
    aniso_arr = np.array([h["aniso"] for h, _ in frames])
    E0 = frames[0][0]["E_tot"]
    rel_dE_arr = np.abs(np.array([h["E_tot"] for h, _ in frames]) - E0) / E0

    # Isotropic reference RMS radius in 4-velocity space
    p_last = frames[-1][1]
    u_iso_rms = np.sqrt(0.5 * (np.mean((p_last["px"] / p_last["mass"])**2) +
                               np.mean((p_last["pz"] / p_last["mass"])**2)))

    for idx, (hdr, p) in enumerate(frames):
        fig = plt.figure(figsize=(12.6, 4.9), dpi=110)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 1.0, 1.0], wspace=0.27,
                              left=0.02, right=0.98, top=0.84, bottom=0.13)

        fig.suptitle(
            f"3D Periodic Box Boltzmann Isotropization   |   t = {hdr['t_fm']:5.2f} fm/c   |   Pz / P_perp = {hdr['aniso']:5.3f}",
            fontsize=13, fontweight="bold", color=TEXT_MAIN, y=0.95
        )

        # Panel 1: 3D periodic torus box
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        style_3d_axis(ax3d, hdr["xmin"], hdr["xmax"], hdr["ymin"], hdr["ymax"], hdr["zmin"], hdr["zmax"])
        draw_wireframe_box(ax3d, hdr["xmin"], hdr["xmax"], hdr["ymin"], hdr["ymax"], hdr["zmin"], hdr["zmax"])
        render_3d_species(ax3d, p, tail_dt=0.50)
        ax3d.view_init(elev=20.0, azim=35.0 + 0.5 * idx)
        ax3d.set_title("Periodic 3D Torus [0, 20 fm]^3", pad=4)

        # Panel 2: (ux, uz) momentum ellipse -> circle
        ax_m = fig.add_subplot(gs[0, 1])
        ux = p["px"] / p["mass"]
        uz = p["pz"] / p["mass"]
        m_nuc, m_gs, m_dstar, m_a4star = split_families(p["species"])

        ax_m.scatter(ux[m_nuc], uz[m_nuc], s=4, c=COL_NUC, alpha=0.35, label="p, n")
        ax_m.scatter(ux[m_gs], uz[m_gs], s=10, c=COL_GS_CLUST, alpha=0.75, label="d, t, 3He, 4He")
        ax_m.scatter(ux[m_dstar], uz[m_dstar], s=9, c=COL_DSTAR, alpha=0.70, label="d*(0+)")
        ax_m.scatter(ux[m_a4star], uz[m_a4star], s=18, c=COL_HE4STAR, marker="*", alpha=0.90, label="4He*")

        sig_x = 2.0 * np.std(ux)
        sig_z = 2.0 * np.std(uz)
        ell_curr = Ellipse((0, 0), width=2 * sig_x, height=2 * sig_z,
                           fill=False, edgecolor="#fde047", lw=2.0, label="Current 2σ contour")
        ell_iso = Ellipse((0, 0), width=4 * u_iso_rms, height=4 * u_iso_rms,
                          fill=False, edgecolor="#4ade80", ls="--", lw=1.4, label="Isotropic target")
        ax_m.add_patch(ell_iso)
        ax_m.add_patch(ell_curr)
        ax_m.set_xlim(-1.05, 1.05)
        ax_m.set_ylim(-1.05, 1.05)
        ax_m.set_aspect("equal")
        ax_m.grid(True)
        ax_m.set_xlabel(r"$u_x = p_x / m$", fontsize=9)
        ax_m.set_ylabel(r"$u_z = p_z / m$", fontsize=9)
        ax_m.set_title("Momentum-Space Relaxation")
        ax_m.legend(loc="upper right", fontsize=7.0, facecolor=PANEL_BG, edgecolor=GRID_COL)

        # Panel 3: Pz / P_perp relaxation + |dE|/E conservation badge
        ax_r = fig.add_subplot(gs[0, 2])
        ax_r.plot(t_arr, aniso_arr, color=COL_NUC, lw=1.5, alpha=0.25)
        ax_r.plot(t_arr[:idx+1], aniso_arr[:idx+1], color=COL_NUC, lw=2.4, label=r"$P_z / P_\perp(t)$")
        ax_r.axhline(1.0, color="#4ade80", ls="--", lw=1.4, label="Equilibrium (1.0)")
        ax_r.scatter([hdr["t_fm"]], [hdr["aniso"]], c="#fde047", s=45, zorder=5)
        ax_r.set_xlim(t_arr[0], t_arr[-1])
        ax_r.set_ylim(0.80, max(3.5, 1.06 * np.max(aniso_arr)))
        ax_r.set_xlabel("Time t [fm/c]", fontsize=9)
        ax_r.set_ylabel(r"Pressure Anisotropy $P_z / P_\perp$", fontsize=9)
        ax_r.set_title("Shear Stress Relaxation")
        ax_r.grid(True)
        ax_r.legend(loc="upper right", fontsize=8.5, facecolor=PANEL_BG, edgecolor=GRID_COL)
        ax_r.text(
            0.36, 0.50,
            f"Collisions: {hdr['collisions']:,}\n|ΔE|/E = {rel_dE_arr[idx]:.2e}\nΔB = 0, ΔQ = 0",
            transform=ax_r.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=GRID_COL, alpha=0.9)
        )

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "box_isotropization", fps=18)


def make_cross_section_plot():
    print("[4/4] Generating Elastic Cross-Section Overview Plot (cross_sections_overview.png)...")
    xs_csv = TMP_DIR / "xs_curves.csv"
    subprocess.run([str(BUILD_BIN), "--dump-xs", str(xs_csv)], check=True)

    cols = {k: [] for k in [
        "p_lab_gev", "pp_mb", "pn_mb", "pd_mb", "pdstar_mb",
        "pt_mb", "p4He_mb", "p4He2021_mb", "4He4He_mb"
    ]}
    with open(xs_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    for k in cols:
        cols[k] = np.array(cols[k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.13, wspace=0.22)
    fig.suptitle(
        "Parametric NN (ERE + Kikuchi–Kawai + Cugnon) & Gaussian Glauber Cluster Elastic Cross Sections",
        fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
    )

    p = cols["p_lab_gev"]
    ax1.plot(p, cols["pp_mb"], color="#38bdf8", lw=2.1, label=r"$p + p$")
    ax1.plot(p, cols["pn_mb"], color="#818cf8", lw=2.1, label=r"$p + n$")
    ax1.plot(p, cols["pd_mb"], color="#fbbf24", lw=2.1, label=r"$p + d(1^+)$")
    ax1.plot(p, cols["pt_mb"], color="#34d399", lw=2.1, label=r"$p + t(1/2^+)$")
    ax1.plot(p, cols["p4He_mb"], color="#fb923c", lw=2.1, label=r"$p + {}^4\mathrm{He}_{\rm g.s.}$")
    ax1.plot(p, cols["4He4He_mb"], color="#f43f5e", lw=2.1, ls="--", label=r"${}^4\mathrm{He} + {}^4\mathrm{He}$ (same $p_{\rm lab}/A$)")
    ax1.set_xscale("log")
    ax1.set_xlim(0.04, 3.0)
    ax1.set_ylim(0, 280)
    ax1.set_xlabel(r"Projectile Lab Momentum per Nucleon $p_{\rm lab}/A$ [GeV/$c$]")
    ax1.set_ylabel(r"Elastic Cross Section $\sigma_{\rm el}$ [mb]")
    ax1.set_title("Mass & Isospin Hierarchy Across Energy Regimes")
    ax1.grid(True, which="both")
    ax1.legend(loc="upper right", fontsize=8.5, facecolor=PANEL_BG, edgecolor=GRID_COL)

    ax2.plot(p, cols["pd_mb"], color=COL_GS_CLUST, lw=2.2, label=r"Ground $p + d(1^+)$ ($R_{\rm rms}=2.14$ fm)")
    ax2.plot(p, cols["pdstar_mb"], color=COL_DSTAR, lw=2.2, ls="-.", label=r"Excited $p + d^*(0^+)$ ($R_{\rm rms}=3.80$ fm)")
    ax2.fill_between(p, cols["pd_mb"], cols["pdstar_mb"], color=COL_DSTAR, alpha=0.14)

    ax2.plot(p, cols["p4He_mb"], color="#fb923c", lw=2.2, label=r"Ground $p + {}^4\mathrm{He}(0^+)$ ($R_{\rm rms}=1.68$ fm)")
    ax2.plot(p, cols["p4He2021_mb"], color=COL_HE4STAR, lw=2.2, ls="-.",
             label=r"Breathing $p + {}^4\mathrm{He}^*(20.21, 0^+)$ ($R_{\rm rms}=2.52$ fm)")
    ax2.fill_between(p, cols["p4He_mb"], cols["p4He2021_mb"], color=COL_HE4STAR, alpha=0.16)

    ax2.set_xscale("log")
    ax2.set_xlim(0.04, 3.0)
    ax2.set_ylim(0, 355)
    ax2.set_xlabel(r"Proton Beam Momentum $p_{\rm lab}$ [GeV/$c$]")
    ax2.set_ylabel(r"Elastic Cross Section $\sigma_{\rm el}$ [mb]")
    ax2.set_title("Excited-State Spatial Swelling Enhancement")
    ax2.grid(True, which="both")
    ax2.legend(loc="upper right", fontsize=8.2, facecolor=PANEL_BG, edgecolor=GRID_COL)

    out_png = ASSETS_DIR / "cross_sections_overview.png"
    fig.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)
    print(f"  -> Wrote {out_png.name} ({out_png.stat().st_size / 1e3:.1f} KB)")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()

    make_colliding_slabs_animation()
    make_fireball_animation()
    make_box_isotropization_animation()
    make_cross_section_plot()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("All animations and plots generated successfully in:", ASSETS_DIR)


if __name__ == "__main__":
    main()
