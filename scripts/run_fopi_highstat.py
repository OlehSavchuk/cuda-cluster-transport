#!/usr/bin/env python3
"""
High-statistics multi-core FOPI Au+Au @ 0.25A GeV (250 A MeV) ensemble runner.
Samples FOPI's exact geometric b*db centrality windows with --in-medium-ec exact:
  - 0.25 < b0 < 0.45 (MD-Soft):  10 b-points x N_test=500 = 5,000 events (1.97M nucleons)
  - 0.25 < b0 < 0.45 (MD-Stiff): 10 b-points x N_test=500 = 5,000 events (1.97M nucleons)
  - b0 < 0.15        (MD-Soft):   5 b-points x N_test=500 = 2,500 events (0.99M nucleons)
Pools raw bin counts & exact standard errors across all runs.
"""

import csv
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = REPO_ROOT / "build" / "cluster_transport"
OUT_DIR = REPO_ROOT / "output"
TMP_RUNS = OUT_DIR / "highstat_runs"

B_MAX_AUAU = 1.15 * 2.0 * (197.0 ** (1.0 / 3.0))  # 13.380 fm (Reisdorf et al., NPA 876)


def run_one_job(job):
    tag, eos, b_fm, ntest, seed = job
    out_csv = TMP_RUNS / f"{tag}.csv"
    cmd = [
        str(BIN_PATH), "--mode", "hic", "--A-nuc", "197", "--Z-nuc", "79",
        "--elab-amev", "250.0", "--impact-b", f"{b_fm:.4f}",
        "--eos", eos, "--in-medium-ec", "exact",
        "--ntest", str(ntest), "--box", "48.0", "--grid", "24",
        "--steps", "240", "--dt", "0.30", "--seed", str(seed),
        "--flow-csv", str(out_csv),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dt_s = time.time() - t0
    cons_line = [ln.strip() for ln in res.stdout.splitlines() if "Conservation" in ln]
    return tag, b_fm, dt_s, (cons_line[-1] if cons_line else ""), out_csv


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


def merge_ensemble_csvs(csv_paths, dst_csv: Path):
    rap_tables = [list(csv.DictReader(open(p))) for p in csv_paths]
    ut0_tables = [list(csv.DictReader(open(str(p) + ".ut0.csv"))) for p in csv_paths]
    yld_tables = [list(csv.DictReader(open(str(p) + ".yields.csv"))) for p in csv_paths]
    m_runs = len(csv_paths)

    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "bin", "y", "y0", "dN_dy", "dN_dy0",
                    "v1", "v1_err", "v2", "v2_err", "n_stat",
                    "v1_ut08", "v1_ut08_err", "v2_ut08", "v2_ut08_err", "n_ut08"])
        for idx in range(len(rap_tables[0])):
            r0 = rap_tables[0][idx]
            dn_dy = sum(float(t[idx]["dN_dy"]) for t in rap_tables) / m_runs
            dn_dy0 = sum(float(t[idx]["dN_dy0"]) for t in rap_tables) / m_runs
            v1, e1, n_st = pool_mean_err(
                [(float(t[idx]["v1"]), float(t[idx]["v1_err"]), float(t[idx]["n_stat"])) for t in rap_tables])
            v2, e2, _ = pool_mean_err(
                [(float(t[idx]["v2"]), float(t[idx]["v2_err"]), float(t[idx]["n_stat"])) for t in rap_tables])
            v1u, e1u, n_u8 = pool_mean_err(
                [(float(t[idx]["v1_ut08"]), float(t[idx]["v1_ut08_err"]), float(t[idx]["n_ut08"])) for t in rap_tables])
            v2u, e2u, _ = pool_mean_err(
                [(float(t[idx]["v2_ut08"]), float(t[idx]["v2_ut08_err"]), float(t[idx]["n_ut08"])) for t in rap_tables])
            w.writerow([r0["species"], r0["bin"], r0["y"], r0["y0"],
                        f"{dn_dy:.6g}", f"{dn_dy0:.6g}",
                        f"{v1:.6g}", f"{e1:.6g}", f"{v2:.6g}", f"{e2:.6g}", f"{n_st:.0f}",
                        f"{v1u:.6g}", f"{e1u:.6g}", f"{v2u:.6g}", f"{e2u:.6g}", f"{n_u8:.0f}"])

    with open(str(dst_csv) + ".ut0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "bin", "ut0", "v2", "v2_err", "n_stat"])
        for idx in range(len(ut0_tables[0])):
            r0 = ut0_tables[0][idx]
            v2, e2, n_u = pool_mean_err(
                [(float(t[idx]["v2"]), float(t[idx]["v2_err"]), float(t[idx]["n_stat"])) for t in ut0_tables])
            w.writerow([r0["species"], r0["bin"], r0["ut0"], f"{v2:.6g}", f"{e2:.6g}", f"{n_u:.0f}"])

    with open(str(dst_csv) + ".yields.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "yield_per_event"])
        for idx in range(len(yld_tables[0])):
            sp = yld_tables[0][idx]["species"]
            yv = sum(float(t[idx]["yield_per_event"]) for t in yld_tables) / m_runs
            w.writerow([sp, f"{yv:.6g}"])


def main():
    TMP_RUNS.mkdir(parents=True, exist_ok=True)
    jobs = []
    # 10 b*db Gauss-midpoint samples in 0.25 < b0 < 0.45 for MD-Soft & MD-Stiff (5,000 events each)
    for k in range(10):
        b0_sq = 0.25 ** 2 + ((k + 0.5) / 10.0) * (0.45 ** 2 - 0.25 ** 2)
        b_fm = B_MAX_AUAU * math.sqrt(b0_sq)
        jobs.append((f"c2_soft_{k:02d}",  "md-soft",  b_fm, 500, 100000 + k * 7919))
        jobs.append((f"c2_stiff_{k:02d}", "md-stiff", b_fm, 500, 200000 + k * 7919))
    # 5 b*db Gauss-midpoint samples in b0 < 0.15 for Central MD-Soft (2,500 events)
    for k in range(5):
        b_fm = B_MAX_AUAU * 0.15 * math.sqrt((k + 0.5) / 5.0)
        jobs.append((f"c1_soft_{k:02d}", "md-soft", b_fm, 500, 300000 + k * 7919))

    workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(jobs)} FOPI runs across {workers} workers (12,500 Au+Au events, 4.93M test nucleons)...")
    t_start = time.time()
    out_map = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_job, j): j for j in jobs}
        for fut in as_completed(futs):
            tag, b_fm, dt_s, cons, path = fut.result()
            out_map[tag] = path
            print(f"  [DONE] {tag:12s} (b={b_fm:5.2f} fm) in {dt_s:5.1f}s | {cons}", flush=True)

    merge_ensemble_csvs([out_map[f"c2_soft_{k:02d}"] for k in range(10)],  OUT_DIR / "fopi_c2_mdsoft.csv")
    merge_ensemble_csvs([out_map[f"c2_stiff_{k:02d}"] for k in range(10)], OUT_DIR / "fopi_c2_mdstiff.csv")
    merge_ensemble_csvs([out_map[f"c1_soft_{k:02d}"] for k in range(5)],   OUT_DIR / "fopi_c1_mdsoft.csv")
    shutil.copyfile(OUT_DIR / "fopi_c2_mdsoft.csv", OUT_DIR / "fopi_auau250_flow.csv")
    shutil.rmtree(TMP_RUNS, ignore_errors=True)
    print(f"All 12,500 FOPI events merged in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
