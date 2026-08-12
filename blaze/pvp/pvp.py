"""Zero-copy vector wrapper for the fixed two-player PvP environment."""
import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_SO = os.path.join(HERE, "pvp_cpu.so")
CUDA_SO = os.path.join(HERE, "pvp_cuda.so")
N_PLAYERS = 2
N_ACT = 9
N_OBS = 35
NETWORK_FRAMES = 4
N_NETWORK_OBS = N_OBS * NETWORK_FRAMES + 1


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


class NetworkVecPvp:
    """Tick-quantized, independently delayed two-player network domain.

    RTT is sampled per lane/player at reset.  Uplink and downlink each consume
    half of that RTT, with independent tick phase.  The policy receives four
    delivered observation frames followed by *its own* normalized current RTT;
    the rival's RTT is never copied into its row.
    """
    HISTORY = 8

    def __init__(self, n, device=0, so_path=None, min_ping_ms=20.0,
                 max_ping_ms=200.0):
        self.raw = VecPvp(n, device=device, so_path=so_path)
        self.n, self.device, self.backend = self.raw.n, self.raw.device, self.raw.backend
        self.xp = self.raw.xp
        self.min_ping_ms, self.max_ping_ms = float(min_ping_ms), float(max_ping_ms)
        self.obs = None
        self._initialized = False
        self._cursor = 0

    def _zeros(self, shape, dtype):
        if self.backend == "cuda":
            import torch
            return torch.zeros(shape, dtype=dtype, device=self.raw.obs.device)
        import numpy as np
        return np.zeros(shape, dtype=dtype)

    def _as_backend(self, value, dtype):
        if self.backend == "cuda":
            import torch
            return torch.as_tensor(value, dtype=dtype, device=self.raw.obs.device)
        import numpy as np
        return np.asarray(value, dtype=dtype)

    def _initialize(self):
        if self.backend == "cuda":
            import torch
            f32, f64, i64 = torch.float32, torch.float64, torch.int64
        else:
            import numpy as np
            f32, f64, i64 = np.float32, np.float64, np.int64
        self._action_history = self._zeros(
            (self.HISTORY, self.n, N_PLAYERS, N_ACT), f64)
        self._due_history = self._zeros(
            (self.HISTORY, self.n, N_PLAYERS), i64)
        self._obs_history = self._zeros(
            (self.HISTORY, self.n, N_PLAYERS, N_OBS), f32)
        self._base_ping = self._zeros((self.n, N_PLAYERS), f32)
        self._phase = self._zeros((self.n, N_PLAYERS), f32)
        self._rate = self._zeros((self.n, N_PLAYERS), f32)
        self._steps = self._zeros((self.n, N_PLAYERS), i64)
        self._last_due = self._zeros((self.n, N_PLAYERS), i64)
        self._applied = self._zeros((self.n, N_PLAYERS, N_ACT), f64)
        self.obs = self._zeros((self.n, N_PLAYERS, N_NETWORK_OBS), f32)
        self._initialized = True

    @staticmethod
    def _seed_parameters(seeds):
        """Stable per-role parameters without a host/device RNG dependency."""
        import numpy as np
        s = np.asarray(seeds, dtype=np.uint64)[:, None]
        role = np.asarray([[0, 1]], dtype=np.uint64)
        x = s ^ (role * np.uint64(0x9E3779B97F4A7C15))
        x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        u0 = ((x & np.uint64(0xFFFFFF)).astype(np.float64) / float(1 << 24))
        u1 = (((x >> np.uint64(24)) & np.uint64(0xFFFFFF)).astype(np.float64)
              / float(1 << 24))
        u2 = (((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.float64)
              / float(1 << 16))
        return u0, u1, u2

    def current_ping_ms(self):
        xp = self.xp
        return self._base_ping * (1.0 + 0.05 * xp.sin(
            self._phase + self._steps * self._rate))

    def _delays(self, ping, phase_offset):
        """One-way milliseconds plus a changing packet/tick boundary phase."""
        xp = self.xp
        frac = 0.5 + 0.5 * xp.sin(
            self._phase * 1.731 + self._steps * (self._rate * 2.173)
            + phase_offset)
        return xp.floor(ping / 100.0 + frac).astype(self._steps.dtype) \
            if self.backend == "cpu" else xp.floor(ping / 100.0 + frac).long()

    def _gather_history(self, history, delay, frame_offset=0):
        xp = self.xp
        index = (self._cursor - delay - frame_offset) % self.HISTORY
        if self.backend == "cuda":
            import torch
            lane = torch.arange(self.n, device=self.raw.obs.device)[:, None]
            role = torch.arange(N_PLAYERS, device=self.raw.obs.device)[None, :]
        else:
            import numpy as np
            lane = np.arange(self.n)[:, None]
            role = np.arange(N_PLAYERS)[None, :]
        return history[index, lane, role]

    def _publish_observation(self, down_delay):
        frames = [self._gather_history(self._obs_history, down_delay, age)
                  for age in range(NETWORK_FRAMES)]
        ping = self.current_ping_ms()[..., None] / self.max_ping_ms
        self.obs[...] = self.xp.concatenate(frames + [ping], axis=-1)

    def reset(self, seeds, mask=None):
        import numpy as np
        raw_obs = self.raw.reset(seeds, mask)
        if not self._initialized:
            self._initialize()
        selected = np.ones(self.n, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
        u0, u1, u2 = self._seed_parameters(np.asarray(seeds, dtype=np.uint64))
        base = self.min_ping_ms + (self.max_ping_ms - self.min_ping_ms) * u0
        phase = 2.0 * np.pi * u1
        # Periods of roughly 2.5--10 seconds at 20 Hz: unstable but correlated ping.
        rate = 2.0 * np.pi / (50.0 + 150.0 * u2)
        if self.backend == "cuda":
            import torch
            sel = torch.as_tensor(selected, device=self.raw.obs.device)
        else:
            sel = selected
        self._base_ping[sel] = self._as_backend(base[selected], self._base_ping.dtype)
        self._phase[sel] = self._as_backend(phase[selected], self._phase.dtype)
        self._rate[sel] = self._as_backend(rate[selected], self._rate.dtype)
        self._steps[sel] = 0
        self._action_history[:, sel] = 0
        self._due_history[:, sel] = 0
        self._last_due[sel] = 0
        self._applied[sel] = 0
        for h in range(self.HISTORY):
            self._obs_history[h, sel] = raw_obs[sel]
        self._publish_observation(self._delays(self.current_ping_ms(), 2.7))
        return self.obs

    def step(self, actions, repeat=1):
        if repeat != 1:
            raise ValueError("network-domain simulation requires one 20 Hz tick per decision")
        if self.backend == "cuda":
            import torch
            submitted = torch.as_tensor(actions, dtype=torch.float64,
                                        device=self.raw.obs.device).contiguous()
        else:
            import numpy as np
            submitted = np.ascontiguousarray(actions, dtype=np.float64)
        self._action_history[self._cursor] = submitted
        ping = self.current_ping_ms()
        uplink = self._delays(ping, 0.3)
        due = self._steps + uplink
        # Minecraft rides TCP: jitter may coalesce packets, but it cannot make a
        # later hotbar/attack packet overtake an earlier packet from that client.
        due = self.xp.maximum(due, self._last_due)
        self._last_due[...] = due
        self._due_history[self._cursor] = due
        # Scan oldest to newest so all packets due this tick are consumed FIFO and
        # the final (newest) state becomes the command applied by the server tick.
        for age in reversed(range(self.HISTORY)):
            index = (self._cursor - age) % self.HISTORY
            ready = self._due_history[index] <= self._steps
            self._applied[...] = self.xp.where(
                ready[..., None], self._action_history[index], self._applied)
        raw_obs, reward, done, hits, damage = self.raw.step(self._applied, repeat=1)
        self._obs_history[self._cursor] = raw_obs
        downlink = self._delays(ping, 2.7)
        self._publish_observation(downlink)
        self._steps += 1
        self._cursor = (self._cursor + 1) % self.HISTORY
        return self.obs, reward, done, hits, damage

    def close(self):
        self.raw.close()

    def __del__(self):
        if hasattr(self, "raw"):
            self.close()
