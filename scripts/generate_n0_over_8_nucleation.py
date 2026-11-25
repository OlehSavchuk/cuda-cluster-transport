#!/usr/bin/env python3
"""
Simulate & animate dynamic light-cluster nucleation at sub-saturation nuclear density
  rho_B = n_0 / 8 = 0.02 fm^-3  (with n_0 = 0.16 fm^-3)
  T_start = 20.0 MeV
starting from 100% pure protons & neutrons in a [0, 30 fm]^3 periodic box.
"""

import csv
import glob
import math
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
TMP_DIR = Path("/tmp/nucleation_n0_8_work")

BG_DARK = "#0b0f19"
PANEL_BG = "#111827"
GRID_COL = "#243049"
TEXT_MAIN = "#e5e7eb"
TEXT_DIM = "#9ca3af"

COL_NUC = "#38bdf8"     # p, n
COL_DEUT = "#fbbf24"    # d (1+)
COL_DSTAR = "#e879f9"   # d*(0+) triplet
COL_A3 = "#34d399"      # t, 3He
COL_HE4 = "#fb923c"     # 4He g.s.
COL_A4STAR = "#fb7185"  # 4He* (19 states)

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


def draw_wireframe_box(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            ax.plot([x, x], [y, y], [zmin, zmax], color="#334155", lw=0.9, alpha=0.65)
        for z in [zmin, zmax]:
            ax.plot([x, x], [ymin, ymax], [z, z], color="#334155", lw=0.9, alpha=0.65)
    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            ax.plot([xmin, xmax], [y, y], [z, z], color="#334155", lw=0.9, alpha=0.65)


def n_mj(m: float, g: int, T_gev: float) -> float:
    hbarc = 0.1973269804
    ds = 8.0 / 128.0
    s = (np.arange(128) + 0.5) * ds
    u = s * s
    K = T_gev * u
    E = m + K
    pm = np.sqrt(K * (K + 2.0 * m))
    integ = float(np.sum(2.0 * s * pm * E * np.exp(-u) * ds))
    return (g / (2.0 * math.pi**2 * hbarc**3)) * T_gev * integ * math.exp(-m / T_gev)


def compute_saha_targets(p_final, vol_fm3: float, n_test: float):
    sp = p_final["species"]
    n_p = float(np.sum(sp == 0))
    n_n = float(np.sum(sp == 1))
    p2_e = (p_final["px"].astype(float)**2 + p_final["py"].astype(float)**2 +
            p_final["pz"].astype(float)**2) / p_final["E"].astype(float)
    T_eq = float(np.mean(p2_e) / 3.0)

    mp, mn = 0.938272088, 0.939565420
    md, mt, m3he, m4he = 1.875612943, 2.808921133, 2.808391607, 3.727379407
    zp = (n_p / (n_test * vol_fm3)) / n_mj(mp, 2, T_eq)
    zn = (n_n / (n_test * vol_fm3)) / n_mj(mn, 2, T_eq)

    return {
        "T_eq_mev": T_eq * 1e3,
        "d": n_test * vol_fm3 * (zp * zn) * n_mj(md, 3, T_eq),
        "a3": n_test * vol_fm3 * ((zp * zn**2) * n_mj(mt, 2, T_eq) + (zp**2 * zn) * n_mj(m3he, 2, T_eq)),
        "he4": n_test * vol_fm3 * (zp**2 * zn**2) * n_mj(m4he, 1, T_eq),
    }


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


def run_and_render():
    dump_dir = TMP_DIR / "dump"
    png_dir = TMP_DIR / "png"
    bw_csv = TMP_DIR / "bw.csv"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(BUILD_BIN), "--dump-bw", str(bw_csv)], check=True)
    bw_dstar_dm, bw_dstar_A = [], []
    with open(bw_csv, "r") as f:
        for row in csv.DictReader(f):
            if int(row["species_id"]) == 4:
                bw_dstar_dm.append(float(row["dm_above_th_mev"]))
                bw_dstar_A.append(float(row["A_gev_inv"]))
    bw_dstar_dm = np.array(bw_dstar_dm)
    bw_dstar_A = np.array(bw_dstar_A)

    # Exact rho_B = n_0 / 8 = 0.02 fm^-3:
    #   L = 30.0 fm -> V = 27,000 fm^3, N_test = 60 -> N_part(0) = 60 * 0.02 * 27,000 = 32,400!
    print("Simulating n_0/8 (0.02 fm^-3), T_start = 20.0 MeV nucleation (640 steps, dt=0.40 fm/c)...")
    subprocess.run([
        str(BUILD_BIN),
        "--mode", "nucleosynthesis",
        "--npart", "32400",
        "--ntest", "60",
        "--box", "30.0",
        "--grid", "10",
        "--temp", "20.0",
        "--inelastic-scale", "1.2",
        "--steps", "640",
        "--dt", "0.40",
        "--substeps", "1",
        "--dump-dir", str(dump_dir),
        "--dump-every", "8",
        "--dump-limit", "34000",
    ], check=True)

    frames = load_frames(dump_dir)
    saha = compute_saha_targets(frames[-1][1], vol_fm3=30.0**3, n_test=60.0)

    t_arr = np.array([h["t_fm"] for h, _ in frames])
    y_nuc = np.array([h["c_nuc"] for h, _ in frames])
    y_d = np.array([h["c_d"] for h, _ in frames])
    y_ds = np.array([h["c_dstar"] for h, _ in frames])
    y_a3 = np.array([h["c_a3"] for h, _ in frames])
    y_he4 = np.array([h["c_he4"] for h, _ in frames])
    y_a4s = np.array([h["c_a4star"] for h, _ in frames])

    T_mev_arr = []
    X_clust_arr = []
    b_tot = float(frames[0][0]["c_nuc"])
    for hdr, p in frames:
        p2_e = (p["px"].astype(float)**2 + p["py"].astype(float)**2 +
                p["pz"].astype(float)**2) / p["E"].astype(float)
        T_mev_arr.append(float(np.mean(p2_e) / 3.0) * 1e3)
        b_clust = 2.0 * (hdr["c_d"] + hdr["c_dstar"]) + 3.0 * hdr["c_a3"] + 4.0 * (hdr["c_he4"] + hdr["c_a4star"])
        X_clust_arr.append(100.0 * b_clust / b_tot)
    T_mev_arr = np.array(T_mev_arr)
    X_clust_arr = np.array(X_clust_arr)

    m_th_dstar = 0.938272088 + 0.939565420
    dm_bins = np.linspace(0.0, 35.0, 26)

    print(f"Rendering {len(frames)} frames (T_start={T_mev_arr[0]:.1f} MeV -> T_eq={T_mev_arr[-1]:.1f} MeV, X_clust={X_clust_arr[-1]:.1f}%)...")
    for idx, (hdr, p) in enumerate(frames):
        fig = plt.figure(figsize=(13.2, 6.4), dpi=110)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.28, 1.0], hspace=0.44, wspace=0.23,
                              left=0.02, right=0.92, top=0.89, bottom=0.08)
        fig.suptitle(
            f"Light-Cluster Nucleation at ρ_B = n_0/8 (0.02 fm⁻³), T_start = 20 MeV   |   t = {hdr['t_fm']:5.1f} fm/c   |   T(t) = {T_mev_arr[idx]:4.1f} MeV",
            fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.set_facecolor(BG_DARK)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor(GRID_COL)
        ax3d.grid(False)
        ax3d.set_xlim(0.0, 30.0)
        ax3d.set_ylim(0.0, 30.0)
        ax3d.set_zlim(0.0, 30.0)
        ax3d.tick_params(labelsize=7.5, pad=1)
        ax3d.set_xlabel("x [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_ylabel("y [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_zlabel("z [fm]", fontsize=8.5, labelpad=1)
        draw_wireframe_box(ax3d, 0.0, 30.0, 0.0, 30.0, 0.0, 30.0)

        sp = p["species"]
        p_nuc = p[(sp == 0) | (sp == 1)][::18]
        p_d = p[sp == 2][::2]
        p_ds = p[(sp >= 3) & (sp <= 5)][::2]
        p_a3 = p[(sp == 6) | (sp == 7)]
        p_he4 = p[sp == 8]
        p_a4s = p[sp >= 9]

        # Short flying velocity streaks on clusters & nucleons
        for sub_p, col in [(p_nuc[::5], COL_NUC), (p_d[::4], COL_DEUT), (p_a3[::2], COL_A3), (p_he4, COL_HE4)]:
            for row in sub_p:
                vx, vy, vz = row["px"] / row["E"], row["py"] / row["E"], row["pz"] / row["E"]
                ax3d.plot([row["x"] - 1.6 * vx, row["x"]],
                          [row["y"] - 1.6 * vy, row["y"]],
                          [row["z"] - 1.6 * vz, row["z"]], color=col, lw=0.7, alpha=0.40)

        if len(p_nuc) > 0:
            ax3d.scatter(p_nuc["x"], p_nuc["y"], p_nuc["z"], s=4.5, c=COL_NUC, alpha=0.35, label="p, n")
        if len(p_d) > 0:
            ax3d.scatter(p_d["x"], p_d["y"], p_d["z"], s=13.0, c=COL_DEUT, alpha=0.88, label="d (1+)")
        if len(p_ds) > 0:
            ax3d.scatter(p_ds["x"], p_ds["y"], p_ds["z"], s=11.0, c=COL_DSTAR, alpha=0.78, label="d*(0+) BW")
        if len(p_a3) > 0:
            ax3d.scatter(p_a3["x"], p_a3["y"], p_a3["z"], s=24.0, c=COL_A3, alpha=0.94, label="t, 3He")
        if len(p_he4) > 0:
            ax3d.scatter(p_he4["x"], p_he4["y"], p_he4["z"], s=44.0, c=COL_HE4, edgecolors="#ffffff",
                         linewidths=0.4, alpha=0.98, label="4He g.s.")
        if len(p_a4s) > 0:
            ax3d.scatter(p_a4s["x"], p_a4s["y"], p_a4s["z"], s=28.0, c=COL_A4STAR, marker="*", alpha=0.94, label="4He*")

        ax3d.view_init(elev=19.0, azim=28.0 + 0.55 * idx)
        ax3d.set_title("3D Periodic Box [0, 30 fm]³ (N_test = 60, Overflows = 0)", pad=4)
        ax3d.legend(loc="upper left", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=2)

        # Panel 1 (Top-Right): Multiplicities + Saha dashed equilibrium targets
        ax_y = fig.add_subplot(gs[0, 1])
        for arr, col, lbl in [
            (y_nuc, COL_NUC, "p+n"), (y_d, COL_DEUT, "d"), (y_ds, COL_DSTAR, "d*"),
            (y_a4s, COL_A4STAR, "4He*"), (y_a3, COL_A3, "t+3He"), (y_he4, COL_HE4, "4He")
        ]:
            ax_y.plot(t_arr, np.maximum(arr, 0.7), color=col, lw=1.1, alpha=0.20)
            ax_y.plot(t_arr[:idx+1], np.maximum(arr[:idx+1], 0.7), color=col, lw=2.0, label=lbl)
        for val, col in [(saha["d"], COL_DEUT), (saha["a3"], COL_A3), (saha["he4"], COL_HE4)]:
            ax_y.axhline(val, color=col, ls="--", lw=0.95, alpha=0.75)
        ax_y.axvline(hdr["t_fm"], color=TEXT_MAIN, ls=":", lw=0.9, alpha=0.7)
        ax_y.set_yscale("log")
        ax_y.set_xlim(t_arr[0], t_arr[-1])
        ax_y.set_ylim(0.7, 45000)
        ax_y.set_ylabel("N(t)  [dashed=Saha]", fontsize=8.5)
        ax_y.set_title("Cluster Multiplicities & BBN-like d Overshoot -> Saha")
        ax_y.grid(True, which="both")
        ax_y.legend(loc="right", fontsize=7.0, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=3)

        # Panel 2 (Mid-Right): Latent-heat fusion temperature T(t) & Bound Baryon Fraction X_clust(t)
        ax_t = fig.add_subplot(gs[1, 1])
        ax_t.plot(t_arr, T_mev_arr, color="#fde047", lw=1.2, alpha=0.22)
        ax_t.plot(t_arr[:idx+1], T_mev_arr[:idx+1], color="#fde047", lw=2.1)
        ax_t.axvline(hdr["t_fm"], color=TEXT_MAIN, ls=":", lw=0.9, alpha=0.7)
        ax_t.set_xlim(t_arr[0], t_arr[-1])
        ax_t.set_ylim(18.5, max(34.0, 1.08 * np.max(T_mev_arr)))
        ax_t.set_ylabel("T_kin [MeV]", color="#fde047", fontsize=8.5)
        ax_t.grid(True)

        ax_t2 = ax_t.twinx()
        ax_t2.plot(t_arr, X_clust_arr, color="#4ade80", lw=1.2, ls="--", alpha=0.22)
        ax_t2.plot(t_arr[:idx+1], X_clust_arr[:idx+1], color="#4ade80", lw=2.0)
        ax_t2.set_ylim(0.0, 52.0)
        ax_t2.set_ylabel("Bound Baryons [%]", color="#4ade80", fontsize=8.5)
        ax_t.set_title(f"Self-Heating from Cluster Formation: 20.0 -> {T_mev_arr[idx]:.1f} MeV ({X_clust_arr[idx]:.1f}% Bound)")

        # Panel 3 (Bot-Right): Live d*(0+) off-shell Breit-Wigner spectrum at instantaneous T(t)
        ax_bw = fig.add_subplot(gs[2, 1])
        m_ds = (p[sp == 4]["mass"] - m_th_dstar) * 1e3
        if len(m_ds) >= 10:
            ax_bw.hist(m_ds, bins=dm_bins, density=True, color=COL_DSTAR, alpha=0.45,
                       edgecolor=COL_DSTAR, lw=0.8, label=f"Sim d*(0+) (N={len(m_ds)})")
        w_th = bw_dstar_A * np.exp(-bw_dstar_dm / max(T_mev_arr[idx], 1.0))
        w_th /= np.trapezoid(w_th, bw_dstar_dm) if hasattr(np, "trapezoid") else np.trapz(w_th, bw_dstar_dm)
        ax_bw.plot(bw_dstar_dm, w_th, color="#fde047", lw=1.9, label=f"BW × exp(-m/T), T={T_mev_arr[idx]:.1f} MeV")
        ax_bw.set_xlim(0.0, 35.0)
        ax_bw.set_ylim(0.0, 0.17)
        ax_bw.set_xlabel("Off-Shell Mass m - (m_p + m_n) [MeV]     |     Time t [fm/c] (upper panels)", fontsize=8.2)
        ax_bw.set_ylabel("Density [MeV⁻¹]", fontsize=8.5)
        ax_bw.set_title("Live Off-Shell Breit–Wigner Mass Distribution")
        ax_bw.grid(True)
        ax_bw.legend(loc="upper right", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL)

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "nucleation_n0_over_8_T20MeV", fps=18)
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()
    run_and_render()
