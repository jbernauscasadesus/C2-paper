#!/usr/bin/env python3
"""Experiment 2: Horn-Schunck vs TV-L1 vs RAFT on Yosemite, the sequence used in the paper.

Same frames (8-9) and same metric (average angular error, with and without the sky) as Table 2 of
Zach et al. (2007), so our TV-L1 can be compared directly with the published numbers. We also
corrupt the second frame with outlier patches and noise to test the robustness claim of the L1
data term.

Run:  python exp2_yosemite.py           (writes results/exp2_*.png and results/exp2_*.csv)
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import flowtools
from compare_flow import HornSchunck
from tvl1_demo import TVL1

PAPER_TABLE2 = {50: (2.85, 5.06), 150: (2.88, 3.70), 250: (2.89, 3.27)}   # (without sky, with sky)
COLORS = {"TV-L1 (ours)": "#dd6b20", "Horn-Schunck": "#2b6cb0", "RAFT": "#2f855a"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    s = flowtools.fetch_yosemite(args.data)
    gt, every, no_sky = s["gt"], s["valid"], ~s["sky"]
    tvl1 = dict(lam=0.15, theta=0.3, tau=0.125, warps=1, levels=6, device=args.device)

    def aae(flow, mask):
        return flowtools.angular_error(flow, gt, mask)

    # --- 1. Reproduce Table 2 of the paper with our TV-L1
    rows = []
    for iters, (paper_ns, paper_s) in PAPER_TABLE2.items():
        flow = flowtools.Runner("TV-L1", TVL1(iters=iters, **tvl1))(s["img0"], s["img1"])
        rows.append({"iterations": iters, "paper, without sky": paper_ns, "ours, without sky": aae(flow, no_sky),
                     "paper, with sky": paper_s, "ours, with sky": aae(flow, every)})
    table2 = pd.DataFrame(rows).set_index("iterations").round(2)
    table2.to_csv(out / "exp2_table2_reproduction.csv")
    print(table2.to_string(), "\n")

    # --- 2. Three methods, 50 iterations per level for TV-L1 (as in the paper's real-time setting)
    runners = {
        "TV-L1 (ours)": flowtools.Runner("TV-L1 (ours)", TVL1(iters=50, **tvl1)),
        "Horn-Schunck": flowtools.Runner("Horn-Schunck", HornSchunck(alpha=15.0, iters=200, warps=1, levels=6,
                                                                      device=args.device)),
        "RAFT": flowtools.Runner("RAFT", flowtools.RAFTFlow("C_T_V2", device=args.device), kind="bgr"),
    }
    flows, rows = {}, []
    for name, r in runners.items():
        r.warmup(s["img0"], s["img1"])
        for _ in range(5):
            flows[name] = r(s["img0"], s["img1"])
        f = flows[name]
        rows.append({"method": name, "AAE without sky (deg)": aae(f, no_sky), "AAE with sky (deg)": aae(f, every),
                     "EPE without sky (px)": flowtools.evaluate(f, gt, no_sky)["EPE (px)"],
                     "time per pair (ms, CPU)": r.ms})
    rows.append({"method": "TV-L1, paper (Table 2)", "AAE without sky (deg)": 2.85, "AAE with sky (deg)": 5.06})
    rows.append({"method": "Zero flow (baseline)", "AAE without sky (deg)": aae(np.zeros_like(gt), no_sky),
                 "AAE with sky (deg)": aae(np.zeros_like(gt), every),
                 "EPE without sky (px)": flowtools.evaluate(np.zeros_like(gt), gt, no_sky)["EPE (px)"]})
    summary = pd.DataFrame(rows).set_index("method").round(3)
    summary.to_csv(out / "exp2_yosemite.csv")
    print(summary.to_string(), "\n")

    # --- 3. Flow images: frame, ground truth and the three methods
    mag = float(np.percentile(np.hypot(gt[..., 0], gt[..., 1]), 99))
    panels = [("Frame 8", s["img0"], None), ("Ground truth", flowtools.flow_to_bgr(gt, mag), None)]
    for name, f in flows.items():
        panels.append((name, flowtools.flow_to_bgr(f, mag), f"AAE {aae(f, no_sky):.2f} deg (without sky)"))
    panels.append(("Colour code", flowtools.color_wheel(240, mag), None))
    grid = flowtools.make_grid(panels, cols=3, panel_size=(474, 378))
    cv2.imwrite(str(out / "exp2_yosemite_flows.png"), grid)

    # --- 4. Robustness: corrupt frame 9 and measure the end-point error (without sky)
    tests = {
        "Outlier patches (fraction of pixels)": ([0.0, 0.05, 0.1, 0.2], flowtools.add_outliers),
        "Gaussian noise (std, gray levels)": ([0, 5, 10, 20], flowtools.add_noise),
    }
    fig, axes = plt.subplots(1, len(tests), figsize=(10, 3.6))
    rob = []
    for ax, (title, (levels, corrupt)) in zip(axes, tests.items()):
        for name, r in runners.items():
            errs = [flowtools.evaluate(r(s["img0"], corrupt(s["img1"], v), record=False), gt, no_sky)["EPE (px)"]
                    for v in levels]
            rob += [{"test": title, "level": v, "method": name, "EPE (px)": e} for v, e in zip(levels, errs)]
            ax.plot(levels, errs, marker="o", label=name, color=COLORS[name])
        ax.set(xlabel=title, ylabel="EPE without sky (px)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out / "exp2_robustness.png", dpi=200)
    pd.DataFrame(rob).round(3).to_csv(out / "exp2_robustness.csv", index=False)
    print(pd.DataFrame(rob).pivot_table(index=["test", "level"], columns="method", values="EPE (px)").round(3))
    print("saved", out / "exp2_yosemite_flows.png", out / "exp2_robustness.png")


if __name__ == "__main__":
    main()
