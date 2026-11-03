#!/usr/bin/env python3
"""Replica de TV-L1 (Zach, Pock, Bischof, 2007) amb demos en temps real.

Instal·lacio:  pip install torch opencv-contrib-python numpy
Execucio:      python tvl1_demo.py mirror
               python tvl1_demo.py pong
               python tvl1_demo.py mirror --source video.mp4

Opcions utils: --width 240 --iters 20 --levels 3 --device cpu
Tecles:        q o Esc per sortir, r per reiniciar el Pong
"""
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_grad(u):
    """Gradient amb diferencies endavant."""
    ux = F.pad(u[..., :, 1:] - u[..., :, :-1], (0, 1, 0, 0))
    uy = F.pad(u[..., 1:, :] - u[..., :-1, :], (0, 0, 0, 1))
    return ux, uy


def backward_div(px, py):
    """Divergencia amb diferencies enrere, adjunta de -forward_grad."""
    dx = torch.cat(
        (px[..., :, :1], px[..., :, 1:-1] - px[..., :, :-2], -px[..., :, -2:-1]),
        dim=-1,
    )
    dy = torch.cat(
        (py[..., :1, :], py[..., 1:-1, :] - py[..., :-2, :], -py[..., -2:-1, :]),
        dim=-2,
    )
    return dx + dy


def central_grad(img):
    """Gradient d'imatge amb diferencies centrals."""
    p = F.pad(img, (1, 1, 1, 1), mode="replicate")
    ix = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / 2
    iy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / 2
    return ix, iy


def warp(img, ux, uy):
    """Mostreja I1(x + u) amb interpolacio bilineal."""
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    gx = 2 * (xs + ux[0, 0]) / max(w - 1, 1) - 1
    gy = 2 * (ys + uy[0, 0]) / max(h - 1, 1) - 1
    grid = torch.stack((gx, gy), dim=-1)[None]
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def upscale_flow(ux, uy, height, width):
    _, _, h, w = ux.shape
    size = (height, width)
    ux = F.interpolate(ux, size=size, mode="bilinear", align_corners=True) * (width / w)
    uy = F.interpolate(uy, size=size, mode="bilinear", align_corners=True) * (height / h)
    return ux, uy


class TVL1:
    """Flux optic TV-L1 per dualitat, amb piramide coarse-to-fine.

    Imatges en escala 0-255. Com al paper, un sol warp per nivell (warps=1)
    i tau <= 1/8 (Eq. 10). El paper no fixa lambda ni theta.

    Nota: l'Eq. 17 del paper combina u = v - theta*div(p) amb
    p + (tau/theta)*grad(u), que no es coherent amb les Eqs. 8 i 10.
    Aqui s'usa la forma equivalent i coherent u = v + theta*div(p).
    literal_eq17=True reprodueix el text tal com esta imprès.
    """

    def __init__(self, lam=0.15, theta=0.3, tau=0.125, iters=50, warps=1,
                 levels=4, device="cpu", literal_eq17=False):
        self.lam = lam
        self.theta = theta
        self.tau = tau
        self.iters = iters
        self.warps = warps
        self.levels = levels
        self.device = torch.device(device)
        self.literal_eq17 = literal_eq17

    def _tensor(self, image):
        arr = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(arr)[None, None].to(self.device)

    def _pyramid(self, image):
        pyramid = [image]
        while len(pyramid) < self.levels and min(pyramid[-1].shape[-2:]) >= 32:
            pyramid.append(F.avg_pool2d(pyramid[-1], 2))
        return pyramid[::-1]

    def _solve_level(self, i0, i1, ux, uy):
        lam_theta = self.lam * self.theta
        step = self.tau / self.theta
        div_sign = -1.0 if self.literal_eq17 else 1.0

        p1x = torch.zeros_like(ux)
        p1y = torch.zeros_like(ux)
        p2x = torch.zeros_like(ux)
        p2y = torch.zeros_like(ux)

        for _ in range(self.warps):
            ux0, uy0 = ux, uy
            warped = warp(i1, ux0, uy0)
            ix, iy = central_grad(warped)
            grad2 = ix * ix + iy * iy
            rho0 = warped - i0 - ix * ux0 - iy * uy0
            threshold = lam_theta * grad2
            inverse_grad2 = 1.0 / (grad2 + 1e-6)

            for _ in range(self.iters):
                rho = rho0 + ix * ux + iy * uy

                # Pas 1a: llindar puntual de l'Eq. 16.
                factor = torch.where(
                    rho < -threshold,
                    torch.full_like(rho, lam_theta),
                    torch.where(
                        rho > threshold,
                        torch.full_like(rho, -lam_theta),
                        -rho * inverse_grad2,
                    ),
                )
                vx = ux + factor * ix
                vy = uy + factor * iy

                # Pas 1b: recuperem u a partir de les variables duals.
                ux = vx + div_sign * self.theta * backward_div(p1x, p1y)
                uy = vy + div_sign * self.theta * backward_div(p2x, p2y)

                # Pas 2: iteracio de punt fix de Chambolle.
                gxx, gxy = forward_grad(ux)
                gyx, gyy = forward_grad(uy)
                norm_x = 1 + step * torch.sqrt(gxx * gxx + gxy * gxy + 1e-12)
                norm_y = 1 + step * torch.sqrt(gyx * gyx + gyy * gyy + 1e-12)
                p1x = (p1x + step * gxx) / norm_x
                p1y = (p1y + step * gxy) / norm_x
                p2x = (p2x + step * gyx) / norm_y
                p2y = (p2y + step * gyy) / norm_y
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


def flow_to_bgr(flow, max_magnitude=8.0):
    """Tonalitat = direccio, intensitat = magnitud, com a la Fig. 1."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * 255 / max_magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def vertical_motion(flow, threshold=0.7):
    """Mediana del moviment vertical dels pixels que es mouen."""
    moving = np.hypot(flow[..., 0], flow[..., 1]) > threshold
    if moving.mean() < 0.01:
        return 0.0
    return float(np.median(flow[..., 1][moving]))


class Pong:
    """Pong a dos jugadors: cada meitat de la imatge controla una pala."""

    def __init__(self, width=960, height=720, gain=5.0, seed=None):
        self.width = width
        self.height = height
        self.gain = gain
        self.paddle_height = height // 4
        self.rng = np.random.default_rng(seed)
        self.score = [0, 0]
        self.paddles = [height / 2, height / 2]
        self.reset_ball()

    def reset_ball(self):
        self.ball = np.array([self.width / 2, self.height / 2], dtype=float)
        direction = self.rng.choice([-1.0, 1.0])
        self.velocity = np.array(
            [direction * 9.0, self.rng.uniform(-5.0, 5.0)], dtype=float
        )

    def update(self, flow):
        half = flow.shape[1] // 2
        motions = (vertical_motion(flow[:, :half]), vertical_motion(flow[:, half:]))
        limit = self.paddle_height / 2
        for index, motion in enumerate(motions):
            target = self.paddles[index] + self.gain * motion
            self.paddles[index] = float(np.clip(target, limit, self.height - limit))

        self.ball += self.velocity
        if self.ball[1] < 12 or self.ball[1] > self.height - 12:
            self.ball[1] = np.clip(self.ball[1], 12, self.height - 12)
            self.velocity[1] *= -1

        margin = 42
        if self.ball[0] < margin and self.velocity[0] < 0:
            self._hit_or_score(0, margin)
        elif self.ball[0] > self.width - margin and self.velocity[0] > 0:
            self._hit_or_score(1, self.width - margin)

    def _hit_or_score(self, side, x):
        offset = (self.ball[1] - self.paddles[side]) / (self.paddle_height / 2)
        if abs(offset) <= 1:
            self.ball[0] = x
            self.velocity[0] *= -1.05
            self.velocity[1] += offset * 4
        else:
            self.score[1 - side] += 1
            self.reset_ball()

    def draw(self, frame, flow):
        canvas = cv2.resize(frame, (self.width, self.height))
        colors = cv2.resize(flow_to_bgr(flow), (self.width, self.height))
        canvas = cv2.addWeighted(canvas, 0.35, colors, 0.9, 0)
        cv2.line(
            canvas, (self.width // 2, 0), (self.width // 2, self.height),
            (220, 220, 220), 1,
        )
        half = self.paddle_height // 2
        for x, y in ((24, self.paddles[0]), (self.width - 24, self.paddles[1])):
            y = int(y)
            cv2.rectangle(canvas, (x - 9, y - half), (x + 9, y + half), (255, 255, 255), -1)
        ball = (int(self.ball[0]), int(self.ball[1]))
        cv2.circle(canvas, ball, 12, (0, 230, 255), -1)
        text = f"{self.score[0]}  :  {self.score[1]}"
        cv2.putText(
            canvas, text, (self.width // 2 - 85, 62),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA,
        )
        return canvas


def open_source(source):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"No s'ha pogut obrir la font de video: {source}")
    return capture


def read_gray(capture, width):
    ok, frame = capture.read()
    if not ok:
        return None, None
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    frame = cv2.resize(cv2.flip(frame, 1), (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0.8)
    return frame, gray


def draw_fps(image, fps):
    cv2.putText(
        image, f"{fps:.1f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )


def add_controls(window, solver):
    cv2.createTrackbar("lambda x100", window, int(solver.lam * 100), 100, lambda _: None)
    cv2.createTrackbar("theta x100", window, int(solver.theta * 100), 100, lambda _: None)
    cv2.createTrackbar("iteracions", window, solver.iters, 100, lambda _: None)


def read_controls(window, solver):
    solver.lam = max(cv2.getTrackbarPos("lambda x100", window), 1) / 100
    solver.theta = max(cv2.getTrackbarPos("theta x100", window), 5) / 100
    solver.iters = max(cv2.getTrackbarPos("iteracions", window), 1)


def run(args):
    device = pick_device(args.device)
    print(f"Dispositiu: {device}")
    solver = TVL1(
        lam=args.lam, theta=args.theta, iters=args.iters, warps=args.warps,
        levels=args.levels, device=device,
    )
    capture = open_source(args.source)
    window = "TV-L1"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    game = None
    if args.mode == "pong":
        game = Pong(gain=args.gain)
    else:
        add_controls(window, solver)

    _, previous = read_gray(capture, args.width)
    if previous is None:
        raise SystemExit("La font de video no ha donat cap imatge.")

    fps = 0.0
    while True:
        started = time.perf_counter()
        frame, current = read_gray(capture, args.width)
        if frame is None:
            break
        if game is None:
            read_controls(window, solver)

        flow = solver(previous, current)
        previous = current

        if game is None:
            output = np.hstack((frame, flow_to_bgr(flow)))
        else:
            game.update(flow)
            output = game.draw(frame, flow)

        elapsed = time.perf_counter() - started
        fps = 0.9 * fps + 0.1 / max(elapsed, 1e-6)
        draw_fps(output, fps)
        cv2.imshow(window, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and game is not None:
            game = Pong(gain=args.gain)

    capture.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("mirror", "pong"))
    parser.add_argument("--source", default="0", help="index de webcam o ruta de video")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--warps", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--gain", type=float, default=5.0, help="sensibilitat del Pong")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps o cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
