"""Bitwise CPU/CUDA gate for the vector two-player PvP API."""
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pvp import VecPvp


def main():
    sm = os.environ.get("MC_SM", "sm_120")
    subprocess.run(["make", "-C", HERE, "cpu", "cuda", f"SM={sm}"],
                   check=True)
    n = int(os.environ.get("PVP_VERIFY_N", "64"))
    seeds = np.arange(1000, 1000 + n, dtype=np.uint64)
    cpu = VecPvp(n, so_path=os.path.join(HERE, "pvp_cpu.so"))
    gpu = VecPvp(n, device=0, so_path=os.path.join(HERE, "pvp_cuda.so"))
    co = cpu.reset(seeds).copy()
    go = gpu.reset(seeds).cpu().numpy()
    if not np.array_equal(co.view(np.uint32), go.view(np.uint32)):
        raise AssertionError("reset observations differ")

    rng = np.random.default_rng(12345)
    for decision in range(256):
        a = np.zeros((n, 2, 7), dtype=np.float64)
        a[:, :, 0] = rng.integers(-1, 2, size=(n, 2))
        a[:, :, 1] = rng.integers(-1, 2, size=(n, 2))
        a[:, :, 2] = rng.choice((-15.0, 0.0, 15.0), size=(n, 2))
        a[:, :, 3] = rng.choice((-10.0, 0.0, 10.0), size=(n, 2))
        a[:, :, 4] = rng.integers(0, 2, size=(n, 2))
        a[:, :, 5] = rng.integers(0, 2, size=(n, 2))
        a[:, :, 6] = rng.integers(0, 2, size=(n, 2))
        cr = tuple(x.copy() for x in cpu.step(a, repeat=4))
        gr = tuple(x.cpu().numpy() for x in gpu.step(
            torch.as_tensor(a, device="cuda"), repeat=4))
        for channel, (x, y) in enumerate(zip(cr, gr)):
            if not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
                where = np.argwhere(x != y)
                first = tuple(where[0]) if len(where) else None
                raise AssertionError(
                    f"decision {decision} channel {channel} differs at {first}: "
                    f"cpu={x[first] if first else None} gpu={y[first] if first else None}")
    cpu.close()
    gpu.close()
    print(f"PASS pvp CPU==CUDA: {n} matches x 256 decisions x repeat4")


if __name__ == "__main__":
    main()
