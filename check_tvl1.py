import time
import cv2
import numpy as np
import torch
from tvl1_demo import TVL1, Pong, flow_to_bgr, vertical_motion, backward_div, forward_grad


def texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for s, a in ((2, 1.0), (5, 1.0), (11, 1.2), (25, 1.5)):
        n = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), s)
        out += a * n / n.std()
    out = (out - out.min()) / (out.max() - out.min()) * 255
    return out


def make_pair(kind, h=240, w=320):
    i1 = texture(h, w)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "translacio":
        u = np.full((h, w), 6.3, np.float32); v = np.full((h, w), -3.7, np.float32)
    elif kind == "suau":
        u = 7 * np.sin(2 * np.pi * ys / h).astype(np.float32)
        v = 5 * np.cos(2 * np.pi * xs / w).astype(np.float32)
    else:  # discontinu: fons quiet, quadrat que es mou
        u = np.zeros((h, w), np.float32); v = np.zeros((h, w), np.float32)
        m = (xs > 100) & (xs < 200) & (ys > 70) & (ys < 170)
        u[m] = 5; v[m] = 3
    i0 = cv2.remap(i1, xs + u, ys + v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return i0, i1, np.dstack((u, v))


def epe(f, gt, b=16):
    d = np.hypot(*(f - gt).transpose(2, 0, 1))[b:-b, b:-b]
    return float(d.mean())


if __name__ == "__main__":
    # adjunció de gradient i divergència
    torch.manual_seed(0)
    a = torch.rand(1, 1, 40, 50, dtype=torch.float64)
    px, py = torch.rand_like(a), torch.rand_like(a)
    gx, gy = forward_grad(a)
    print("adjunt:", float((gx * px + gy * py).sum() + (a * backward_div(px, py)).sum()))

    try:
        ocv = cv2.optflow.DualTVL1OpticalFlow_create(warps=1, scaleStep=0.5, nscales=4)
    except Exception as e:
        ocv = None
        print("sense opencv-contrib:", e)

    for kind in ("translacio", "suau", "discontinu"):
        i0, i1, gt = make_pair(kind)
        print("==", kind, "EPE zero:", round(epe(np.zeros_like(gt), gt), 3))
        for label, kw in (
            ("paper 1 warp 50 it", dict(iters=50, warps=1)),
            ("5 warps 30 it", dict(iters=30, warps=5)),
            ("tau 1/4", dict(iters=50, warps=1, tau=0.25)),
            ("Eq17 literal", dict(iters=50, warps=1, literal_eq17=True)),
        ):
            s = TVL1(**kw)
            t = time.perf_counter(); f = s(i0, i1); dt = time.perf_counter() - t
            print(f"{label:22s} EPE {epe(f, gt):8.3f}  max|u| {np.abs(f).max():9.2f}  {dt*1000:7.1f} ms")
        if ocv is not None:
            f = ocv.calc(i0.astype(np.uint8), i1.astype(np.uint8), None)
            print(f"{'OpenCV':22s} EPE {epe(f, gt):8.3f}")

    # Temps en mode demo i fum del Pong
    i0, i1, gt = make_pair("suau")
    s = TVL1(iters=30, levels=4)
    s(i0, i1)
    t = time.perf_counter()
    for _ in range(5):
        f = s(i0, i1)
    print("demo CPU ms/frame:", (time.perf_counter() - t) / 5 * 1000, "threads", torch.get_num_threads())

    frame = cv2.cvtColor(i0.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    g = Pong(seed=1)
    start = g.paddles[0]
    down = np.zeros((240, 320, 2), np.float32); down[..., 1] = 2.0
    g.update(down)
    print("pala baixa:", g.paddles[0] > start)
    for _ in range(600):
        g.update(np.zeros((240, 320, 2), np.float32))
    print("marcador:", g.score, "imatge", g.draw(frame, down).shape, flow_to_bgr(down).shape)
    print("quiet:", vertical_motion(np.zeros((10, 10, 2), np.float32)))
