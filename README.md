# Real-time TV-L1 optical flow

A PyTorch replication of the method by Zach, Pock and Bischof (2007), with a comparison against Horn-Schunck and two webcam demos.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To use the GPU, first install the right PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/). When you run the scripts, they should print `Device: cuda` or `Device: mps`.

## Usage

Webcam demos:

```bash
python tvl1_demo.py mirror
python tvl1_demo.py pong
```

Keys: `q` or `Esc` to quit, `r` to restart Pong. If it runs slowly, try `--width 240 --iters 20`.

Three-panel comparison video (input, Horn-Schunck and TV-L1):

```bash
python compare_flow.py video.mp4 --output comparison.mp4
```

Tests on synthetic data and comparison with OpenCV:

```bash
python check_tvl1.py
```

## Files

| File | Contents |
|---|---|
| `tvl1_demo.py` | TV-L1 and the mirror and Pong demos |
| `compare_flow.py` | Horn-Schunck and comparison video generation |
| `check_tvl1.py` | Tests with known synthetic flow |
| `exp1_solvers.py` | Exp. 1: the same TV-L1 energy with two algorithms (the paper's, and smoothed \|x\| + gradient descent) |
| `exp2_yosemite.py` | Exp. 2: Horn-Schunck, TV-L1 and RAFT on Yosemite 8-9, with the paper's metric (AAE) and robustness tests |
| `comparison.ipynb` | Comparison on KITTI 2015 and Middlebury RubberWhale |

## Experiments for the presentation

Both scripts download Yosemite (cloudless, M. Black) to `data/` and save figures and tables to `results/`. They run on CPU in a few minutes.

```bash
python exp1_solvers.py
python exp2_yosemite.py
```

Results on CPU (Apple M, PyTorch 2.7):

- **Exp. 1.** With a single scale and the same energy, the paper's algorithm gets within 5% of the minimum energy in 0.04 s. Gradient descent with smoothed |x| gets there in 10.4 s with ε = 0.1. With ε = 1 it stalls 17% above, and with ε = 0.01 it is too slow.
- **Exp. 2.** On Yosemite, with 50 iterations per level, our TV-L1 achieves an AAE of 2.82° without sky (paper: 2.85°), Horn-Schunck 5.46° and RAFT 3.09°. With sky, our error is higher than the paper's (10.5° vs 5.1°). With 20% of pixels corrupted, TV-L1's EPE is 1.01 px and Horn-Schunck's is 2.13 px.

## Validation

Average endpoint error in pixels (EPE) on synthetic data, 1 warp and 4 levels. The replication runs 50 iterations per level.

| Motion | Replication | OpenCV |
|---|---|---|
| Translation | 0.058 | 0.012 |
| Smooth flow | 0.065 | 0.036 |
| Object with edges | 0.057 | 0.074 |

These values are not a reproduction of the paper's results. The comparison with the paper on Yosemite is in `exp2_yosemite.py`.

## References

- C. Zach, T. Pock and H. Bischof. *A Duality Based Approach for Realtime TV-L1 Optical Flow*. DAGM 2007.
- B. K. P. Horn and B. G. Schunck. *Determining Optical Flow*. Artificial Intelligence, 1981.

## License

MIT. See the `LICENSE` file.
