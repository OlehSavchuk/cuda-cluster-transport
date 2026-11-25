#!/usr/bin/env python3
"""
Generate animations & spectral plots for the feat/inelastic-breit-wigner branch:
  1. assets/cluster_nucleosynthesis.gif (.mp4) — Dynamic nucleosynthesis from pure p+n
  2. assets/breit_wigner_spectra.png          — Relativistic spectral functions & widths
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
TMP_DIR = Path("/tmp/cluster_inel_anim_work")

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
        "font.size": 10,
        "axes.titlesize": 11,
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


def load_bw_curves(bw_csv: Path):
    by_id = {}
    with open(bw_csv, "r") as f:
        for row in csv.DictReader(f):
            sid = int(row["species_id"])
            if sid not in by_id:
                by_id[sid] = {"name": row["name"], "m": [], "dm": [], "A": [], "gam": []}
            by_id[sid]["m"].append(float(row["m_gev"]))
            by_id[sid]["dm"].append(float(row["dm_above_th_mev"]))
            by_id[sid]["A"].append(float(row["A_gev_inv"]))
            by_id[sid]["gam"].append(float(row["gamma_tot_mev"]))
    for sid in by_id:
        for k in ["m", "dm", "A", "gam"]:
            by_id[sid][k] = np.array(by_id[sid][k])
    return by_id


def make_nucleosynthesis_animation(bw_data):
    print("[1/2] Simulating & rendering Dynamic Cluster Nucleosynthesis (cluster_nucleosynthesis.gif)...")
    dump_dir = TMP_DIR / "nucleo_dump"
    png_dir = TMP_DIR / "nucleo_png"
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        str(BUILD_BIN),
        "--mode", "nucleosynthesis",
        "--npart", "24000",
        "--ntest", "14",
        "--box", "15.5",
        "--grid", "7",
        "--temp", "62.0",
        "--inelastic-scale", "2.6",
        "--steps", "72",
        "--dt", "0.16",
        "--substeps", "2",
        "--dump-dir", str(dump_dir),
        "--dump-every", "1",
        "--dump-limit", "24000",
    ], check=True, stdout=subprocess.DEVNULL)

    frames = load_frames(dump_dir)
    t_arr = np.array([h["t_fm"] for h, _ in frames])
    y_nuc = np.array([h["c_nuc"] for h, _ in frames])
    y_d = np.array([h["c_d"] for h, _ in frames])
    y_ds = np.array([h["c_dstar"] for h, _ in frames])
    y_a3 = np.array([h["c_a3"] for h, _ in frames])
    y_he4 = np.array([h["c_he4"] for h, _ in frames])
    y_a4s = np.array([h["c_a4star"] for h, _ in frames])

    m_th_dstar = 0.938272088 + 0.939565420
    dm_bins = np.linspace(0.0, 45.0, 28)

    for idx, (hdr, p) in enumerate(frames):
        fig = plt.figure(figsize=(12.6, 5.7), dpi=110)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], hspace=0.38, wspace=0.22,
                              left=0.02, right=0.97, top=0.88, bottom=0.10)
        fig.suptitle(
            f"Dynamic Cluster Nucleosynthesis from Pure p + n   |   t = {hdr['t_fm']:5.2f} fm/c   |   Inel+Form+Dec = {hdr['inel_cum']+hdr['form_cum']+hdr['decay_cum']:,}",
            fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
        )

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.set_facecolor(BG_DARK)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor(GRID_COL)
        ax3d.grid(False)
        ax3d.set_xlim(hdr["xmin"], hdr["xmax"])
        ax3d.set_ylim(hdr["ymin"], hdr["ymax"])
        ax3d.set_zlim(hdr["zmin"], hdr["zmax"])
        ax3d.tick_params(labelsize=7, pad=1)
        ax3d.set_xlabel("x [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_ylabel("y [fm]", fontsize=8.5, labelpad=1)
        ax3d.set_zlabel("z [fm]", fontsize=8.5, labelpad=1)
        draw_wireframe_box(ax3d, hdr["xmin"], hdr["xmax"], hdr["ymin"], hdr["ymax"], hdr["zmin"], hdr["zmax"])

        sp = p["species"]
        p_nuc = p[(sp == 0) | (sp == 1)][::14]
        p_d = p[sp == 2][::2]
        p_ds = p[(sp >= 3) & (sp <= 5)][::2]
        p_a3 = p[(sp == 6) | (sp == 7)]
        p_he4 = p[sp == 8]
        p_a4s = p[sp >= 9]
        p_rxn = p[p["collided"] >= 2][::3]

        if len(p_nuc) > 0:
            ax3d.scatter(p_nuc["x"], p_nuc["y"], p_nuc["z"], s=4.5, c=COL_NUC, alpha=0.35, label="p, n (subsampled)")
        if len(p_d) > 0:
            ax3d.scatter(p_d["x"], p_d["y"], p_d["z"], s=14.0, c=COL_DEUT, alpha=0.88, label="d (1+)")
        if len(p_ds) > 0:
            ax3d.scatter(p_ds["x"], p_ds["y"], p_ds["z"], s=12.0, c=COL_DSTAR, alpha=0.80, label="d*(0+) BW")
        if len(p_a3) > 0:
            ax3d.scatter(p_a3["x"], p_a3["y"], p_a3["z"], s=22.0, c=COL_A3, alpha=0.92, label="t, 3He")
        if len(p_he4) > 0:
            ax3d.scatter(p_he4["x"], p_he4["y"], p_he4["z"], s=34.0, c=COL_HE4, alpha=0.98, label="4He g.s.")
        if len(p_a4s) > 0:
            ax3d.scatter(p_a4s["x"], p_a4s["y"], p_a4s["z"], s=30.0, c=COL_A4STAR, marker="*", alpha=0.95, label="4He* (19 BW states)")
        if len(p_rxn) > 0:
            ax3d.scatter(p_rxn["x"], p_rxn["y"], p_rxn["z"], s=28.0, c="#fef08a", edgecolors="#ffffff",
                         linewidths=0.4, alpha=0.90, depthshade=False, label="Inelastic/BW vertex")

        ax3d.view_init(elev=20.0, azim=32.0 + 0.65 * idx)
        ax3d.set_title("3D Nucleation of d, d*, (t, 3He), 4He & 4He*", pad=4)
        ax3d.legend(loc="upper left", fontsize=7.2, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=2)

        # Top-right: Chemical yields N_i(t) on log scale
        ax_y = fig.add_subplot(gs[0, 1])
        for arr, col, lbl in [
            (y_nuc, COL_NUC, "p + n"), (y_d, COL_DEUT, "d"), (y_ds, COL_DSTAR, "d*(0+)"),
            (y_a3, COL_A3, "t + 3He"), (y_a4s, COL_A4STAR, "4He*"), (y_he4, COL_HE4, "4He g.s.")
        ]:
            ax_y.plot(t_arr, np.maximum(arr, 0.8), color=col, lw=1.2, alpha=0.22)
            ax_y.plot(t_arr[:idx+1], np.maximum(arr[:idx+1], 0.8), color=col, lw=2.1, label=lbl)
        ax_y.axvline(hdr["t_fm"], color=TEXT_MAIN, ls=":", lw=1.0, alpha=0.7)
        ax_y.set_yscale("log")
        ax_y.set_xlim(t_arr[0], t_arr[-1])
        ax_y.set_ylim(0.8, 35000)
        ax_y.set_xlabel("Time t [fm/c]", fontsize=8.5)
        ax_y.set_ylabel("Multiplicity N(t)", fontsize=8.5)
        ax_y.set_title("Sequential Chemical Nucleosynthesis Hierarchy")
        ax_y.grid(True, which="both")
        ax_y.legend(loc="right", fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL, ncol=2)

        # Bottom-right: Live off-shell Breit-Wigner mass spectrum of d*(0+) vs theory!
        ax_bw = fig.add_subplot(gs[1, 1])
        m_dstar_live = (p[sp == 4]["mass"] - m_th_dstar) * 1e3
        if len(m_dstar_live) >= 10:
            ax_bw.hist(m_dstar_live, bins=dm_bins, density=True, color=COL_DSTAR, alpha=0.45,
                       edgecolor=COL_DSTAR, lw=0.8, label=f"Live d*(0+) masses (N={len(m_dstar_live)})")
        if 4 in bw_data:
            dm_th = bw_data[4]["dm"]
            w_th = bw_data[4]["A"] * np.exp(-dm_th / 62.0)
            w_th /= np.trapezoid(w_th, dm_th) if hasattr(np, "trapezoid") else np.trapz(w_th, dm_th)
            ax_bw.plot(dm_th, w_th, color="#fde047", lw=2.0, label=r"Thermal BW $\mathcal{A}_{d^*}(m)e^{-m/T}$")

        ax_bw.set_xlim(0.0, 45.0)
        ax_bw.set_ylim(0.0, 0.14)
        ax_bw.set_xlabel(r"Off-Shell Mass Above Threshold $m - (m_p + m_n)$ [MeV]", fontsize=8.5)
        ax_bw.set_ylabel(r"Probability Density [MeV$^{-1}$]", fontsize=8.5)
        ax_bw.set_title("Emergent Off-Shell Breit–Wigner Mass Spectrum")
        ax_bw.grid(True)
        ax_bw.legend(loc="upper right", fontsize=8.0, facecolor=PANEL_BG, edgecolor=GRID_COL)

        fig.savefig(png_dir / f"img_{idx:04d}.png", facecolor=BG_DARK)
        plt.close(fig)

    encode_gif_and_mp4(png_dir, ASSETS_DIR / "cluster_nucleosynthesis", fps=18)


def make_breit_wigner_plot(bw_data):
    print("[2/2] Generating Breit-Wigner Spectral Functions & Widths Plot (breit_wigner_spectra.png)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.13, wspace=0.22)
    fig.suptitle(
        "Relativistic Breit–Wigner Spectral Functions A_R(m) & Threshold-Dependent Partial Widths Γ(m)",
        fontsize=12.5, fontweight="bold", color=TEXT_MAIN, y=0.96
    )

    palette = [
        (4,  "#e879f9", r"$d^*(0^+)$ [$s$-wave $p+n$]"),
        (9,  "#38bdf8", r"${}^4\mathrm{He}^*(20.21, 0^+)$ [$s$-wave $p+t$]"),
        (11, "#fbbf24", r"${}^4\mathrm{He}^*(21.84, 2^-)$ [$p$-wave $N+A_3$]"),
        (14, "#34d399", r"${}^4\mathrm{He}^*(24.25, 1^-)$ [$p+t, n+{}^3\mathrm{He}, d+d$]"),
        (16, "#fb7185", r"${}^4\mathrm{He}^*(25.95, 1^-)$ [broad $\Gamma_0=12.7$ MeV]"),
        (18, "#818cf8", r"${}^4\mathrm{He}^*(27.42, 2^+)$ [$d$-wave $N+A_3$, $s$-wave $dd$]"),
        (26, "#fb923c", r"${}^4\mathrm{H}^*(2^-)$ [$p$-wave $n+t$]"),
    ]

    for sid, col, lbl in palette:
        if sid not in bw_data:
            continue
        d = bw_data[sid]
        ax1.plot(d["dm"], d["A"] * 1e-3, color=col, lw=2.0, label=lbl)
        ax2.plot(d["dm"], d["gam"], color=col, lw=2.0, label=lbl)

    ax1.set_yscale("log")
    ax1.set_xlim(0.0, 42.0)
    ax1.set_ylim(2e-4, 2.0)
    ax1.set_xlabel(r"Mass Above Lowest Decay Threshold $m - m_{\rm th}$ [MeV]")
    ax1.set_ylabel(r"Normalized Spectral Density $\mathcal{A}_R(m)$ [MeV$^{-1}$]")
    ax1.set_title(r"Unit-Normalized Spectral Shapes ($\int \mathcal{A}_R(m)\,dm = 1$)")
    ax1.grid(True, which="both")
    ax1.legend(loc="upper right", fontsize=7.8, facecolor=PANEL_BG, edgecolor=GRID_COL)

    ax2.set_xlim(0.0, 42.0)
    ax2.set_ylim(0.0, 26.0)
    ax2.set_xlabel(r"Mass Above Lowest Decay Threshold $m - m_{\rm th}$ [MeV]")
    ax2.set_ylabel(r"Total Width $\Gamma_{\rm tot}(m)$ [MeV]")
    ax2.set_title(r"Blatt–Weisskopf Threshold Opening ($q^{2L+1}$ & Multi-Channel Kinks)")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=7.8, facecolor=PANEL_BG, edgecolor=GRID_COL)

    out_png = ASSETS_DIR / "breit_wigner_spectra.png"
    fig.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)
    print(f"  -> Wrote {out_png.name} ({out_png.stat().st_size / 1e3:.1f} KB)")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    style_dark()

    bw_csv = TMP_DIR / "bw_curves.csv"
    subprocess.run([str(BUILD_BIN), "--dump-bw", str(bw_csv)], check=True)
    bw_data = load_bw_curves(bw_csv)

    make_nucleosynthesis_animation(bw_data)
    make_breit_wigner_plot(bw_data)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Generated inelastic & Breit-Wigner assets in:", ASSETS_DIR)


if __name__ == "__main__":
    main()
