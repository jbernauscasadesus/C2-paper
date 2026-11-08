"""flowtools.py -- helpers for the optical-flow comparison notebook (KITTI 2015).

Methods compared: our TV-L1 (tvl1_demo.TVL1), our Horn-Schunck (compare_flow.HornSchunck) and RAFT (torchvision).
Contents: KITTI download and loading, metrics (EPE, Fl-all), method wrappers, colour coding,
panel grid and an H.264 video writer.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
YOSEMITE_URL = "https://cs.brown.edu/people/mjblack/Sequences/yosemite.tar.gz"
EXTRA_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/optflow/"
FLO_MAGIC = 202021.25


# ------------------------------------------------------------------ data
def download(url, dest):
    """Download url to dest with a progress bar (skipped if dest already exists)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name)
        except ImportError:
            bar = None
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            if bar:
                bar.update(len(chunk))
        if bar:
            bar.close()
    tmp.replace(dest)
    return dest


def read_kitti_flow(path):
    """KITTI flow PNG (16 bit, BGR = valid, v, u) -> flow (H, W, 2) in pixels and valid mask."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    flow = (np.stack([raw[..., 2], raw[..., 1]], -1).astype(np.float32) - 2 ** 15) / 64.0
    return flow, raw[..., 0] > 0


def fetch_kitti(data_dir, scene_ids, url=None):
    """Download KITTI 2015 (once, ~2 GB), extract the selected training scenes and load them.

    Returns a list of dicts with img0, img1 (BGR uint8), gt (H, W, 2), valid (H, W) and name.
    """
    data_dir = Path(data_dir)
    archive = download(url or KITTI_URL, data_dir / "data_scene_flow.zip")
    scenes = []
    with zipfile.ZipFile(archive) as z:
        for i in scene_ids:
            members = [f"training/image_2/{i:06d}_10.png", f"training/image_2/{i:06d}_11.png",
                       f"training/flow_occ/{i:06d}_10.png"]
            for m in members:
                if not (data_dir / "kitti" / m).exists():
                    z.extract(m, data_dir / "kitti")
            root = data_dir / "kitti" / "training"
            gt, valid = read_kitti_flow(root / "flow_occ" / f"{i:06d}_10.png")
            scenes.append(dict(name=f"{i:06d}",
                               img0=cv2.imread(str(root / "image_2" / f"{i:06d}_10.png")),
                               img1=cv2.imread(str(root / "image_2" / f"{i:06d}_11.png")),
                               gt=gt, valid=valid))
    return scenes


def read_flo(path):
    """Middlebury .flo file -> flow (H, W, 2) float32."""
    with open(path, "rb") as fh:
        if abs(np.fromfile(fh, np.float32, 1)[0] - FLO_MAGIC) > 1e-3:
            raise ValueError(f"{path} is not a valid .flo file")
        w, h = (int(v) for v in np.fromfile(fh, np.int32, 2))
        return np.fromfile(fh, np.float32, 2 * w * h).reshape(h, w, 2)


def fetch_rubberwhale(data_dir):
    """Middlebury RubberWhale pair with dense ground truth (hosted in opencv_extra)."""
    data_dir = Path(data_dir)
    for name in ("RubberWhale1.png", "RubberWhale2.png", "RubberWhale.flo"):
        download(EXTRA_URL + name, data_dir / name)
    gt = read_flo(data_dir / "RubberWhale.flo")
    valid = np.isfinite(gt).all(-1) & (np.abs(gt) < 1e9).all(-1)
    return dict(name="RubberWhale", img0=cv2.imread(str(data_dir / "RubberWhale1.png")),
                img1=cv2.imread(str(data_dir / "RubberWhale2.png")), gt=gt, valid=valid)


def fetch_yosemite(data_dir, frame=8):
    """Yosemite without clouds (Barron et al. 1994, distributed by M. Black), frames `frame` and `frame + 1`.

    The images are 316 x 252 gray levels with a uniform black sky. The ground truth is stored as bytes
    (value = byte * scale + pedestal), data0 = vertical and data1 = horizontal motion.
    Returns img0, img1 (BGR uint8), gt (H, W, 2), valid (all pixels) and sky (H, W), the sky mask
    used to report the error "without sky" as in Table 2 of the paper.
    """
    import re
    import tarfile

    data_dir = Path(data_dir)
    root = data_dir / "yosemite"
    if not root.exists():
        archive = download(YOSEMITE_URL, data_dir / "yosemite.tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
    h, w = 252, 316

    def image(k):
        gray = np.fromfile(root / "yos-images" / f"yos{k}" / "data", np.uint8).reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    folder = root / "yos-flows" / f"actual-flow{frame}"
    text = (folder / "descriptor").read_text()
    scale = float(re.search(r"\(scale ([-\d.e]+)\)", text).group(1))
    pedestal = float(re.search(r"\(pedestal ([-\d.e]+)\)", text).group(1))
    v = np.fromfile(folder / "data0", np.uint8).reshape(h, w) * scale + pedestal
    u = np.fromfile(folder / "data1", np.uint8).reshape(h, w) * scale + pedestal
    img0 = image(frame)
    # The sky is the black region above the horizon in each column.
    horizon = np.argmax(img0[..., 0] > 0, axis=0)
    sky = np.arange(h)[:, None] < horizon[None, :]
    return dict(name=f"Yosemite {frame}-{frame + 1}", img0=img0, img1=image(frame + 1),
                gt=np.dstack((u, v)).astype(np.float32), valid=np.ones((h, w), bool), sky=sky)


# ------------------------------------------------------------------ metrics
def evaluate(flow, gt, valid):
    """EPE (pixels) and Fl-all (% of outliers: EPE > 3 px and > 5 % of the GT magnitude),
    both over the pixels with ground truth (KITTI protocol)."""
    err = np.hypot(flow[..., 0] - gt[..., 0], flow[..., 1] - gt[..., 1])
    mag = np.hypot(gt[..., 0], gt[..., 1])
    outlier = (err > 3.0) & (err > 0.05 * mag)
    return {"EPE (px)": float(err[valid].mean()), "Fl-all (%)": float(100 * outlier[valid].mean())}


def angular_error(flow, gt, valid):
    """Average angular error in degrees (the metric of the paper, Barron et al. 1994):
    angle between the space-time vectors (u, v, 1) and (u_gt, v_gt, 1)."""
    u, v, ug, vg = flow[..., 0], flow[..., 1], gt[..., 0], gt[..., 1]
    cos = (1 + u * ug + v * vg) / (np.sqrt(1 + u * u + v * v) * np.sqrt(1 + ug * ug + vg * vg))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1)))[valid].mean())


def epe_by_motion(flow, gt, valid, bins):
    """Sum of EPE and pixel count per bin of ground-truth motion magnitude (to pool over scenes)."""
    err = np.hypot(flow[..., 0] - gt[..., 0], flow[..., 1] - gt[..., 1])
    mag = np.hypot(gt[..., 0], gt[..., 1])
    idx = np.digitize(mag, bins[1:-1])[valid]
    n = len(bins) - 1
    return np.bincount(idx, weights=err[valid], minlength=n), np.bincount(idx, minlength=n)


# ------------------------------------------------------------------ methods
def to_gray(img_bgr, blur=0.8):
    """Gray float32 in [0, 255], lightly blurred (same preprocessing as compare_flow.py)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (5, 5), blur) if blur else gray


class RAFTFlow:
    """RAFT (Teed & Deng, ECCV 2020) from torchvision. Input: two BGR uint8 frames.

    weights='C_T_V2': trained on FlyingChairs and FlyingThings3D only, i.e. not fine-tuned
    on KITTI, so the evaluation on KITTI is a fair generalisation test.
    """

    def __init__(self, weights="C_T_V2", device="auto", num_flow_updates=12):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = getattr(Raft_Large_Weights, weights)
        self.model = raft_large(weights=self.weights, progress=True).to(self.device).eval()
        self.preprocess = self.weights.transforms()
        self.updates = num_flow_updates

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)

    def __call__(self, frame0_bgr, frame1_bgr):
        torch = self.torch
        with torch.inference_mode():
            a, b = self._to_tensor(frame0_bgr), self._to_tensor(frame1_bgr)
            h, w = a.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8          # RAFT needs H and W divisible by 8
            if ph or pw:
                pad = lambda x: torch.nn.functional.pad(x.float(), (0, pw, 0, ph), mode="replicate").to(torch.uint8)
                a, b = pad(a), pad(b)
            a, b = self.preprocess(a, b)
            flow = self.model(a, b, num_flow_updates=self.updates)[-1]
            return flow[0, :, :h, :w].permute(1, 2, 0).float().cpu().numpy()


def sync():
    """Wait for the GPU so timings are honest; harmless without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Runner:
    """Preprocessing + timing around a flow method.
    kind='gray': fn(gray0, gray1) with float32 gray images;  kind='bgr': fn(frame0, frame1)."""

    def __init__(self, name, fn, kind="gray", blur=0.8):
        self.name, self.fn, self.kind, self.blur = name, fn, kind, blur
        self.times_ms = []

    def __call__(self, f0, f1, record=True):
        if self.kind == "gray":
            f0, f1 = to_gray(f0, self.blur), to_gray(f1, self.blur)
        sync()
        t = time.perf_counter()
        flow = self.fn(f0, f1)
        sync()
        if record:
            self.times_ms.append((time.perf_counter() - t) * 1000)
        return np.asarray(flow, np.float32)

    def warmup(self, f0, f1, n=1):
        for _ in range(n):
            self(f0, f1, record=False)

    @property
    def ms(self):
        return float(np.median(self.times_ms)) if self.times_ms else float("nan")


# ------------------------------------------------------------------ image corruptions
def add_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def change_gain(img, gain):
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def add_outliers(img, fraction, size=16, seed=0):
    """Random noisy squares (occlusion-like outliers) covering about `fraction` of the image."""
    rng = np.random.default_rng(seed)
    out, (h, w) = img.copy(), img.shape[:2]
    for _ in range(int(fraction * h * w / size ** 2)):
        y, x = rng.integers(0, h - size), rng.integers(0, w - size)
        out[y:y + size, x:x + size] = rng.integers(0, 256, 3)
    return out


# ------------------------------------------------------------------ visualisation / video
def flow_to_bgr(flow, max_mag=16.0):
    """Hue = direction, brightness = magnitude (as in the paper's Fig. 1)."""
    flow = np.nan_to_num(flow).astype(np.float32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = np.minimum(ang * 90 / np.pi, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 255 / max_mag, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def color_wheel(size=240, max_mag=16.0):
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j].astype(np.float32)
    wheel = flow_to_bgr(np.dstack((xx, yy)) * max_mag, max_mag)
    wheel[np.hypot(xx, yy) > 1] = 0
    return wheel


def put_label(img, text, org=(10, 26), scale=0.65, thick=2):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def make_grid(panels, cols=3, panel_size=(620, 188)):
    """panels: list of (title, image_bgr, subtitle or None) -> one BGR image."""
    w, h = panel_size
    tiles = []
    for title, image, subtitle in panels:
        tile = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        put_label(tile, title)
        if subtitle:
            put_label(tile, subtitle, (10, h - 10), scale=0.5, thick=1)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


class VideoWriter:
    """H.264 (yuv420p) writer through ffmpeg: plays in PowerPoint, browsers and YouTube."""

    def __init__(self, path, size, fps=8.0):
        w, h = size
        if w % 2 or h % 2:
            raise ValueError("Video width and height must be even")
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-crf", "18",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame_bgr):
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def scene_panels(scene, flows, metrics, ms, blink_frame, panel_size=(620, 188)):
    """Panels of one KITTI scene. flows/metrics: dict method -> flow / metrics dict."""
    gt, valid = scene["gt"], scene["valid"]
    mag = float(max(2.0, np.percentile(np.hypot(gt[..., 0], gt[..., 1])[valid], 99)))
    frame = scene["img0"] if blink_frame == 0 else scene["img1"]
    wheel = np.zeros_like(scene["img0"])
    d = min(wheel.shape[:2])
    y0, x0 = (wheel.shape[0] - d) // 2, (wheel.shape[1] - d) // 2
    wheel[y0:y0 + d, x0:x0 + d] = cv2.resize(color_wheel(240, mag), (d, d))
    gt_img = flow_to_bgr(np.where(valid[..., None], gt, 0), mag)
    panels = [("Frame t" if blink_frame == 0 else "Frame t+1", frame, None),
              ("Ground truth", gt_img, None),
              (f"Colour code (max {mag:.0f} px)", wheel, None)]
    for name, f in flows.items():
        m = metrics[name]
        panels.append((name, flow_to_bgr(f, mag), f"EPE {m['EPE (px)']:.2f} px | Fl {m['Fl-all (%)']:.1f}% | {ms[name]:.0f} ms"))
    return panels
