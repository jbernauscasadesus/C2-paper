#!/usr/bin/env python3
"""Experiment 1: same TV-L1 energy, two algorithms.

The paper's contribution is an algorithm, not a model: TV-L1 already existed, and the usual way
to minimise it was to round the corner of the absolute value, |x| ~ sqrt(x^2 + eps^2), and run
gradient descent. Here both algorithms minimise the same energy on the same image pair, and we
track the true (not rounded) TV-L1 energy against time.

To compare like with like we use a single scale and one linearisation around u = 0 (no pyramid,
no warping), so both algorithms minimise exactly the same convex energy

    E(u) = sum_x  lambda |rho(u)| + |grad u1| + |grad u2|,   rho(u) = I1 + Ix u1 + Iy u2 - I0

Run:  python exp1_solvers.py            (writes results/exp1_*.png and results/exp1_*.csv)
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import flowtools
from tvl1_demo import backward_div, central_grad, forward_grad


def linearise(img0, img1):
    """Image terms of the linearised data term around u = 0."""
    i0 = torch.from_numpy(flowtools.to_gray(img0))[None, None]
    i1 = torch.from_numpy(flowtools.to_gray(img1))[None, None]
    ix, iy = central_grad(i1)
    return i0, i1, ix, iy


def energy(ux, uy, i0, i1, ix, iy, lam):
    """True TV-L1 energy (isotropic TV on each component, as in Eq. 12 of the paper)."""
    rho = i1 + ix * ux + iy * uy - i0
    gxx, gxy = forward_grad(ux)
    gyx, gyy = forward_grad(uy)
    tv = torch.sqrt(gxx ** 2 + gxy ** 2) + torch.sqrt(gyx ** 2 + gyy ** 2)
    return float((lam * rho.abs() + tv).sum())


def run_dual(i0, i1, ix, iy, lam, theta, tau, iters, every=1):
    """The paper's algorithm: thresholding (v-step) + Chambolle's dual update (u-step), Eq. 17
    with the consistent sign (u = v + theta * div p, see tvl1_demo.TVL1)."""
    ux, uy = torch.zeros_like(i0), torch.zeros_like(i0)
    p1x, p1y, p2x, p2y = (torch.zeros_like(i0) for _ in range(4))
    rho0 = i1 - i0
    grad2 = ix * ix + iy * iy
    lt = lam * theta
    step = tau / theta
    history, elapsed = [(0, 0.0, energy(ux, uy, i0, i1, ix, iy, lam))], 0.0
    for k in range(1, iters + 1):
        t = time.perf_counter()
        rho = rho0 + ix * ux + iy * uy
        factor = torch.where(rho < -lt * grad2, torch.full_like(rho, lt),
                             torch.where(rho > lt * grad2, torch.full_like(rho, -lt), -rho / (grad2 + 1e-6)))
        vx, vy = ux + factor * ix, uy + factor * iy
        ux = vx + theta * backward_div(p1x, p1y)
        uy = vy + theta * backward_div(p2x, p2y)
        gxx, gxy = forward_grad(ux)
        gyx, gyy = forward_grad(uy)
        nx = 1 + step * torch.sqrt(gxx ** 2 + gxy ** 2)
        ny = 1 + step * torch.sqrt(gyx ** 2 + gyy ** 2)
        p1x, p1y = (p1x + step * gxx) / nx, (p1y + step * gxy) / nx
        p2x, p2y = (p2x + step * gyx) / ny, (p2y + step * gyy) / ny
        elapsed += time.perf_counter() - t
        if k % every == 0:
            history.append((k, elapsed, energy(ux, uy, i0, i1, ix, iy, lam)))
    return (ux, uy), history


def run_smoothed(i0, i1, ix, iy, lam, eps, iters, every=1):
    """The usual workaround: replace |x| by sqrt(x^2 + eps^2) everywhere and run gradient descent.
    The step is 1 / L, with L the Lipschitz constant of the smoothed gradient; the smaller eps,
    the larger L and the smaller the step."""
    ux, uy = torch.zeros_like(i0), torch.zeros_like(i0)
    rho0 = i1 - i0
    lipschitz = lam * float((ix * ix + iy * iy).max()) / eps + 8.0 / eps
    lr = 1.0 / lipschitz
    history, elapsed = [(0, 0.0, energy(ux, uy, i0, i1, ix, iy, lam))], 0.0
    for k in range(1, iters + 1):
        t = time.perf_counter()
        rho = rho0 + ix * ux + iy * uy
        data = lam * rho / torch.sqrt(rho * rho + eps * eps)
        gxx, gxy = forward_grad(ux)
        gyx, gyy = forward_grad(uy)
        nx = torch.sqrt(gxx ** 2 + gxy ** 2 + eps * eps)
        ny = torch.sqrt(gyx ** 2 + gyy ** 2 + eps * eps)
        # backward_div is minus the adjoint of forward_grad, so grad of sum |grad u|_eps = -div(...)
        ux = ux - lr * (data * ix - backward_div(gxx / nx, gxy / nx))
        uy = uy - lr * (data * iy - backward_div(gyx / ny, gyy / ny))
        elapsed += time.perf_counter() - t
        if k % every == 0:
            history.append((k, elapsed, energy(ux, uy, i0, i1, ix, iy, lam)))
    return (ux, uy), history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--theta", type=float, default=0.1,
                    help="coupling of u and v; smaller = closer to the true TV-L1 energy")
    ap.add_argument("--dual-iters", type=int, default=2000)
    ap.add_argument("--gd-iters", type=int, default=20000)
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_num_threads(4)

    scene = flowtools.fetch_yosemite(args.data)
    i0, i1, ix, iy = linearise(scene["img0"], scene["img1"])

    runs = {"Paper's algorithm (duality + thresholding)": run_dual(i0, i1, ix, iy, args.lam, args.theta, 0.125, args.dual_iters)}
    for eps in args.eps:
        runs[f"Rounded |x| + gradient descent, ε = {eps:g}"] = run_smoothed(i0, i1, ix, iy, args.lam, eps, args.gd_iters, every=10)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    best = min(h[-1][2] for _, h in runs.values())
    rows = []
    for name, ((ux, uy), hist) in runs.items():
        e_final = hist[-1][2]
        # time to get within 5 % of the lowest energy reached by any algorithm
        reach = next((t for _, t, e in hist if e <= 1.05 * best), np.nan)
        rows.append({"algorithm": name, "iterations": hist[-1][0], "time (s)": hist[-1][1],
                     "final energy": e_final, "energy / best": e_final / best,
                     "time to 5% of best (s)": reach})
    table = pd.DataFrame(rows).set_index("algorithm")
    table.round(4).to_csv(out / "exp1_solvers.csv")
    print(table.round(3).to_string())

    colors = ["#dd6b20", "#2b6cb0", "#4a90c8", "#9ec3e6"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (name, (_, hist)), c in zip(runs.items(), colors):
        t = np.array([h[1] for h in hist[1:]])
        e = np.array([h[2] for h in hist[1:]])
        ax.plot(t, e / best, label=name, color=c, lw=2.4 if "Paper" in name else 1.8)
    ax.axhline(1.05, color="grey", linestyle=":", lw=1)
    ax.text(1.2e-3, 1.06, "5 % above the best", color="grey", fontsize=8)
    ax.set(xscale="log", xlabel="time (s, CPU)", ylabel="TV-L1 energy / lowest energy reached",
           title="Same TV-L1 energy, two algorithms (Yosemite 8-9, single scale)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exp1_energy_vs_time.png", dpi=200)
    print("saved", out / "exp1_energy_vs_time.png")


if __name__ == "__main__":
    main()
