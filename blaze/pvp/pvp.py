"""Zero-copy vector wrapper for the fixed two-player PvP environment."""
import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_SO = os.path.join(HERE, "pvp_cpu.so")
CUDA_SO = os.path.join(HERE, "pvp_cuda.so")
N_PLAYERS = 2
N_ACT = 9
N_OBS = 35


class VecPvp:
    def __init__(self, n, device=0, so_path=None):
        if so_path is None:
            so_path = CPU_SO
            if os.path.exists(CUDA_SO):
                try:
                    import torch
                    if torch.cuda.is_available():
                        so_path = CUDA_SO
                except ImportError:
                    pass
        self.n = int(n)
        self.device = int(device)
        self.so_path = so_path
        self.backend = "cuda" if "cuda" in os.path.basename(so_path) else "cpu"
        self.lib = ctypes.CDLL(so_path)
        self.lib.pvp_create.restype = ctypes.c_void_p
        self.lib.pvp_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.pvp_destroy.argtypes = [ctypes.c_void_p]
        self.lib.pvp_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
        self.lib.pvp_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_int] + [ctypes.c_void_p] * 5
        self.h = self.lib.pvp_create(self.device, self.n)
        if not self.h:
            raise RuntimeError(f"pvp_create failed: {so_path}")
        if self.backend == "cuda":
            import torch
            self.xp = torch
            dev = torch.device(f"cuda:{device}")
            self.obs = torch.zeros((n, N_PLAYERS, N_OBS), dtype=torch.float32,
                                   device=dev)
            self.reward = torch.zeros((n, N_PLAYERS), dtype=torch.float32,
                                      device=dev)
            self.done = torch.zeros(n, dtype=torch.uint8, device=dev)
            self.hits = torch.zeros((n, N_PLAYERS), dtype=torch.int32,
                                    device=dev)
            self.damage = torch.zeros((n, N_PLAYERS), dtype=torch.float32,
                                      device=dev)
        else:
            import numpy as np
            self.xp = np
            self.obs = np.zeros((n, N_PLAYERS, N_OBS), dtype=np.float32)
            self.reward = np.zeros((n, N_PLAYERS), dtype=np.float32)
            self.done = np.zeros(n, dtype=np.uint8)
            self.hits = np.zeros((n, N_PLAYERS), dtype=np.int32)
            self.damage = np.zeros((n, N_PLAYERS), dtype=np.float32)

    def _ptr(self, x):
        if self.backend == "cuda":
            return ctypes.c_void_p(x.data_ptr())
        return ctypes.c_void_p(x.ctypes.data)

    def reset(self, seeds, mask=None):
        import numpy as np
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        if seeds.shape != (self.n,):
            raise ValueError(f"seeds must have shape {(self.n,)}")
        mask_ptr = None
        self._mask_keepalive = None
        if mask is not None:
            self._mask_keepalive = np.ascontiguousarray(mask, dtype=np.uint8)
            if self._mask_keepalive.shape != (self.n,):
                raise ValueError(f"mask must have shape {(self.n,)}")
            mask_ptr = ctypes.c_void_p(self._mask_keepalive.ctypes.data)
        self._seeds_keepalive = seeds
        rc = self.lib.pvp_reset(self.h, mask_ptr,
                                ctypes.c_void_p(seeds.ctypes.data),
                                self._ptr(self.obs))
        if rc:
            raise RuntimeError("pvp_reset failed")
        return self.obs

    def step(self, actions, repeat=1):
        if self.backend == "cuda":
            import torch
            actions = torch.as_tensor(actions, device=self.obs.device,
                                      dtype=torch.float64).contiguous()
        else:
            import numpy as np
            actions = np.ascontiguousarray(actions, dtype=np.float64)
        if tuple(actions.shape) != (self.n, N_PLAYERS, N_ACT):
            raise ValueError(
                f"actions must have shape {(self.n, N_PLAYERS, N_ACT)}")
        self._actions_keepalive = actions
        rc = self.lib.pvp_step(
            self.h, self._ptr(actions), int(repeat), self._ptr(self.obs),
            self._ptr(self.reward), self._ptr(self.done), self._ptr(self.hits),
            self._ptr(self.damage))
        if rc:
            raise RuntimeError("pvp_step failed")
        return self.obs, self.reward, self.done, self.hits, self.damage

    def close(self):
        if getattr(self, "h", None):
            self.lib.pvp_destroy(self.h)
            self.h = None

    def __del__(self):
        self.close()
