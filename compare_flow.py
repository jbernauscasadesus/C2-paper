#!/usr/bin/env python3
"""Video comparatiu: webcam, Horn-Schunck i TV-L1 (Zach et al., 2007).

Instal·lacio:  pip install torch opencv-contrib-python numpy imageio-ffmpeg
Execucio:      python compare_flow.py video.mp4 --output comparacio.mp4

Tots dos metodes fan servir la mateixa piramide, el mateix warping, els mateixos
gradients d'imatge i la mateixa escala de colors.
"""
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tvl1_demo import central_grad, flow_to_bgr, pick_device, upscale_flow, warp


class HornSchunck:
    """Horn-Schunck amb iteracions de Jacobi, piramide i warping."""

    KERNEL = (
        (1 / 12, 1 / 6, 1 / 12),
        (1 / 6, 0.0, 1 / 6),
        (1 / 12, 1 / 6, 1 / 12),
    )

    def __init__(self, alpha=15.0, iters=200, warps=1, levels=5, device="cpu"):
        self.alpha2 = alpha * alpha
        self.iters = iters
        self.warps = warps
        self.levels = levels
        self.device = torch.device(device)
        self.kernel = torch.tensor(self.KERNEL, dtype=torch.float32, device=self.device)[None, None]

    def _tensor(self, image):
        arr = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(arr)[None, None].to(self.device)

    def _pyramid(self, image):
        pyramid = [image]
        while len(pyramid) < self.levels and min(pyramid[-1].shape[-2:]) >= 32:
            pyramid.append(F.avg_pool2d(pyramid[-1], 2))
        return pyramid[::-1]

    def _average(self, u):
        return F.conv2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), self.kernel)

    def _solve_level(self, i0, i1, ux, uy):
        for _ in range(self.warps):
            ux0, uy0 = ux, uy
            warped = warp(i1, ux0, uy0)
            ix, iy = central_grad(warped)
            rho0 = warped - i0 - ix * ux0 - iy * uy0
            denom = self.alpha2 + ix * ix + iy * iy
            for _ in range(self.iters):
                avg_x = self._average(ux)
                avg_y = self._average(uy)
                factor = (rho0 + ix * avg_x + iy * avg_y) / denom
                ux = avg_x - ix * factor
                uy = avg_y - iy * factor
        return ux, uy

    @torch.inference_mode()
    def __call__(self, image0, image1):
        pyramid0 = self._pyramid(self._tensor(image0))
        pyramid1 = self._pyramid(self._tensor(image1))
        ux = torch.zeros_like(pyramid0[0])
        uy = torch.zeros_like(pyramid0[0])
        for level, (i0, i1) in enumerate(zip(pyramid0, pyramid1)):
            if level > 0:
                ux, uy = upscale_flow(ux, uy, *i0.shape[-2:])
            ux, uy = self._solve_level(i0, i1, ux, uy)
        flow = torch.cat((ux, uy), dim=1)[0].permute(1, 2, 0)
        return flow.cpu().numpy()


PANEL_W, PANEL_H = 640, 360


def preprocess(frame, width, height):
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0.8)


def put_text(image, text, position, scale, thickness, outline=True):
    x, y = position
    if outline:
        cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0),
                    thickness, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def labelled(image, text):
    out = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    put_text(out, text, (16, 36), 0.85, 2)
    return out


def open_writer(path, width, height, fps):
    import subprocess

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    return subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", path],
        stdin=subprocess.PIPE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--output", default="comparacio.mp4")
    parser.add_argument("--still", default=None, help="desa aquest fotograma com a PNG")
    parser.add_argument("--still-frame", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--tv-iters", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--hs-iters", type=int, default=200)
    parser.add_argument("--max-magnitude", type=float, default=16.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from tvl1_demo import TVL1

    device = pick_device(args.device)
    print(f"Dispositiu: {device}")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"No s'ha pogut obrir el video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    source_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(round(source_h * args.width / source_w / 2)) * 2

    tvl1 = TVL1(lam=args.lam, theta=args.theta, iters=args.tv_iters, levels=args.levels,
                device=device)
    horn = HornSchunck(alpha=args.alpha, iters=args.hs_iters, levels=args.levels,
                       device=device)
    writer = open_writer(args.output, PANEL_W * 3, PANEL_H, fps)

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("El video no te fotogrames.")
    previous = preprocess(frame, args.width, height)
    index = 0
    times = {"hs": 0.0, "tv": 0.0}
    started = time.perf_counter()
    while True:
        ok, next_frame = capture.read()
        if not ok:
            break
        current = preprocess(next_frame, args.width, height)

        tick = time.perf_counter()
        flow_hs = horn(previous, current)
        times["hs"] += time.perf_counter() - tick
        tick = time.perf_counter()
        flow_tv = tvl1(previous, current)
        times["tv"] += time.perf_counter() - tick

        canvas = np.hstack((
            labelled(frame, "Webcam"),
            labelled(flow_to_bgr(flow_hs, args.max_magnitude), "Horn-Schunck"),
            labelled(flow_to_bgr(flow_tv, args.max_magnitude), "TV-L1"),
        ))
        writer.stdin.write(canvas.tobytes())
        if args.still and index == args.still_frame:
            cv2.imwrite(args.still, canvas)

        frame, previous = next_frame, current
        index += 1
        if index % 50 == 0:
            print(f"{index} fotogrames, {time.perf_counter() - started:.0f} s", flush=True)

    writer.stdin.close()
    writer.wait()
    capture.release()
    print(f"Fet: {args.output}, {index} fotogrames")
    print(f"Horn-Schunck: {index / times['hs']:.1f} fps  TV-L1: {index / times['tv']:.1f} fps")


if __name__ == "__main__":
    main()