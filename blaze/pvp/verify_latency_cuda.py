#!/usr/bin/env python3
"""Verify network wrapper CPU/CUDA trajectory and ping parity."""
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pvp import CPU_SO, CUDA_SO, NetworkVecPvp


def main():
    n = 64
    seeds = np.arange(1234, 1234 + n, dtype=np.uint64)
    cpu = NetworkVecPvp(n, so_path=CPU_SO)
    gpu = NetworkVecPvp(n, so_path=CUDA_SO)
    oc = cpu.reset(seeds).copy()
    og = gpu.reset(seeds).cpu().numpy()
    np.testing.assert_allclose(oc, og, atol=2e-6, rtol=2e-6)
    rng = np.random.default_rng(77)
    for _ in range(100):
        actions = np.zeros((n, 2, 9), dtype=np.float64)
        actions[..., 0:2] = rng.integers(-1, 2, size=(n, 2, 2))
        actions[..., 2] = rng.uniform(-20, 20, size=(n, 2))
        actions[..., 3] = rng.uniform(-10, 10, size=(n, 2))
        actions[..., 4:9] = rng.integers(0, 2, size=(n, 2, 5))
        cc = cpu.step(actions)
        cg = gpu.step(torch.as_tensor(actions, device="cuda"))
        np.testing.assert_allclose(cc[0], cg[0].cpu().numpy(), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(cpu.current_ping_ms(),
                                   gpu.current_ping_ms().cpu().numpy(),
                                   atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(cc[2], cg[2].cpu().numpy())
    cpu.close(); gpu.close()
    print("PASS latency-domain CPU/CUDA parity")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    main()
