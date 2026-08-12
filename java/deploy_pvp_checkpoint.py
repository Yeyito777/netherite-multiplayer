#!/usr/bin/env python3
"""Run the frozen two-policy checkpoint through two live 1.11.2 clients.

The host/guest clients must already be connected on bridge ports 25575/25576.
Movement and look use each real client input bridge. Explicit target attacks use
the host's authoritative NetHandlerPlayServer injection and are executed in
role order, matching the Java oracle contract.
"""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blaze" / "pvp"))
from train_selfplay import (CONTINUOUS_ACTION_SCHEMA, CONTINUOUS_LOOK_ACTION_SCHEMA,
                            FINE_ACTION_SCHEMA, LEGACY_ACTION_SCHEMA, FWD, PITCH,
                            PITCH_LIMIT, STRAFE, YAW, YAW_FINE, YAW_LIMIT, Policy,
                            V2_ACTION_SCHEMA, V21_ACTION_SCHEMA,
                            checkpoint_action_schema, is_continuous_schema,
                            policy_input)


LEGACY_HEAD_VALUES = (FWD, STRAFE, YAW, PITCH,
                      (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def bridge(port, message, timeout=30):
    with socket.create_connection(("127.0.0.1", port), 5) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
        line = sock.makefile().readline()
    if not line:
        raise RuntimeError(f"bridge {port} closed without a response")
    out = json.loads(line)
    if not out.get("ok"):
        raise RuntimeError(f"bridge {port}: {out}")
    return out


def timed_bridge(port, message, timeout=30):
    """Bridge call plus host-monotonic transport timestamps for parity traces."""
    sent_ns = time.monotonic_ns()
    out = bridge(port, message, timeout=timeout)
    received_ns = time.monotonic_ns()
    return out, {"sent_ns": sent_ns, "received_ns": received_ns,
                 "rtt_ms": (received_ns - sent_ns) / 1e6}


class PersistentBridge:
    """One long-lived qrl connection; avoids a TCP accept/close per game tick."""
    def __init__(self, port, timeout=30):
        self.port = port
        self.sock = socket.create_connection(("127.0.0.1", port), 5)
        self.sock.settimeout(timeout)
        self.stream = self.sock.makefile("rwb")
        self.lock = threading.Lock()

    def call(self, message):
        with self.lock:
            sent_ns = time.monotonic_ns()
            self.stream.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
            self.stream.flush()
            line = self.stream.readline()
            received_ns = time.monotonic_ns()
        if not line:
            raise RuntimeError(f"persistent bridge {self.port} closed without a response")
        out = json.loads(line)
        if not out.get("ok"):
            raise RuntimeError(f"persistent bridge {self.port}: {out}")
        return out, {"sent_ns": sent_ns, "received_ns": received_ns,
                     "rtt_ms": (received_ns - sent_ns) / 1e6}

    def close(self):
        try:
            self.stream.close()
        finally:
            self.sock.close()


def f64(bits):
    return struct.unpack(">d", (int(bits) & ((1 << 64) - 1)).to_bytes(8, "big"))[0]


def f32(bits):
    return struct.unpack(">f", (int(bits) & 0xFFFFFFFF).to_bytes(4, "big"))[0]


def decode_player(row):
    return {
        "x": f64(row["position_bits"][0]), "y": f64(row["position_bits"][1]),
        "z": f64(row["position_bits"][2]), "mx": f64(row["motion_bits"][0]),
        "my": f64(row["motion_bits"][1]), "mz": f64(row["motion_bits"][2]),
        "yaw": f32(row["rotation_bits"][0]),
        "pitch": f32(row["rotation_bits"][1]), "health": f32(row["health_bits"]),
        "cooldown": f32(row["attack_cooldown_bits"]),
        "hurt": row["hurt_resistant_time"], "on_ground": row["on_ground"],
        "sprinting": row["sprinting"], "dead": row["dead"],
    }


def observation(players, role, legacy=False):
    p, q = players[role], players[1 - role]
    rad = p["yaw"] * math.pi / 180.0
    sy, cy = math.sin(rad), math.cos(rad)
    dx, dy, dz = q["x"] - p["x"], q["y"] - p["y"], q["z"] - p["z"]
    dmx, dmz = q["mx"] - p["mx"], q["mz"] - p["mz"]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    rel = (q["yaw"] - p["yaw"]) * math.pi / 180.0
    if legacy:
        velocity = (p["mx"] * cy - p["mz"] * sy,
                    p["mx"] * sy + p["mz"] * cy)
        relative = ((dx * cy - dz * sy) / 32.0,
                    (dx * sy + dz * cy) / 32.0,
                    dmx * cy - dmz * sy, dmx * sy + dmz * cy)
    else:
        velocity = (p["mx"] * cy + p["mz"] * sy,
                    -p["mx"] * sy + p["mz"] * cy)
        relative = ((dx * cy + dz * sy) / 32.0,
                    (-dx * sy + dz * cy) / 32.0,
                    dmx * cy + dmz * sy, -dmx * sy + dmz * cy)
    return np.asarray([
        p["health"] / 20.0, q["health"] / 20.0,
        velocity[0], velocity[1], p["my"],
        relative[0], relative[1], dy / 8.0,
        relative[2], relative[3], d / 32.0,
        math.sin(rel), math.cos(rel), sy, cy, p["cooldown"],
        p["hurt"] / 20.0, q["hurt"] / 20.0,
        float(p["on_ground"]), float(p["sprinting"]),
        (p["x"] + 16.0) / 32.0, (16.0 - p["x"]) / 32.0,
        (p["z"] + 16.0) / 32.0, (16.0 - p["z"]) / 32.0,
        p["pitch"] / 90.0,
        float(p.get("weapon", 0)), float(q.get("weapon", 0)),
        float(p.get("blocking", False)), float(q.get("blocking", False)),
        float(p.get("shield_disabled", False)),
        float(q.get("shield_disabled", False)),
        max(0.0, 1.0 - p.get("shield_damage", 0) / 336.0),
        max(0.0, 1.0 - q.get("shield_damage", 0) / 336.0),
        min(1.0, p.get("shield_use_ticks", 0) / 5.0),
        min(1.0, q.get("shield_use_ticks", 0) / 5.0),
    ], dtype=np.float32)


class PolicyObservationHistory:
    """Four delivered frames plus the acting client's own current RTT.

    With simulated RTT, raw client frames enter a downlink queue. Without it,
    Minecraft/network delivery has already delayed the client-visible state.
    """
    def __init__(self, players, rows, simulated_ping_ms=None):
        self.frames = []
        self.raw_frames = []
        for role in range(2):
            first = observation(players, role)
            self.frames.append([first.copy() for _ in range(4)])
            self.raw_frames.append([first.copy() for _ in range(8)])
        self.base_ping_ms = simulated_ping_ms
        self.ping_ms = ([float(x) for x in simulated_ping_ms]
                        if simulated_ping_ms is not None else
                        [float(row.get("ping_ms", 0.0)) for row in rows])
        self.step = 0

    def current_ping(self, role):
        if self.base_ping_ms is None:
            return self.ping_ms[role]
        # Independent smooth instability, bounded to exactly +/-5%.
        phase = (0.7 + role * 2.1) + self.step * (0.071 + role * 0.019)
        return self.base_ping_ms[role] * (1.0 + 0.05 * math.sin(phase))

    def one_way_delay_ticks(self, role, phase_offset):
        ping = self.current_ping(role)
        phase = 0.5 + 0.5 * math.sin(
            (0.7 + role * 2.1) * 1.731
            + self.step * (0.071 + role * 0.019) * 2.173 + phase_offset)
        return int(math.floor(ping / 100.0 + phase))

    def encode(self, role):
        return np.concatenate(self.frames[role] + [
            np.asarray([self.ping_ms[role] / 200.0], dtype=np.float32)])

    def update(self, players, rows):
        for role in range(2):
            current = observation(players, role)
            self.raw_frames[role] = [current] + self.raw_frames[role][:7]
            delay = (self.one_way_delay_ticks(role, 2.7)
                     if self.base_ping_ms is not None else 0)
            delivered = self.raw_frames[role][min(delay, 7)]
            self.frames[role] = [delivered] + self.frames[role][:3]
            self.ping_ms[role] = (self.current_ping(role)
                                  if self.base_ping_ms is not None else
                                  float(rows[role].get("ping_ms", self.ping_ms[role])))
        self.step += 1


class SimulatedActionUplink:
    """FIFO per-player action queue with independently varying one-way delay."""
    def __init__(self, history):
        self.history = history
        self.queue = [[], []]
        self.applied = [[0.0] * 9, [0.0] * 9]
        self.last_due = [0, 0]

    def submit(self, actions):
        now = self.history.step
        for role in range(2):
            due = now + self.history.one_way_delay_ticks(role, 0.3)
            due = max(due, self.last_due[role])  # TCP/FIFO: no overtaking.
            self.last_due[role] = due
            self.queue[role].append((due, actions[role]))
            while self.queue[role] and self.queue[role][0][0] <= now:
                _, self.applied[role] = self.queue[role].pop(0)
        return [list(row) for row in self.applied]


@torch.no_grad()
def policy_action(policy, obs, stochastic=False,
                  action_schema=LEGACY_ACTION_SCHEMA,
                  sample_continuous_yaw=False, sample_continuous_pitch=False):
    policy_obs = policy_input(policy, torch.from_numpy(obs).unsqueeze(0))
    actor, _ = policy(policy_obs)
    if is_continuous_schema(action_schema):
        logits = actor["categorical"]
        if stochastic:
            indices = [int(torch.distributions.Categorical(logits=h).sample())
                       for h in logits]
        else:
            indices = [int(h.argmax(-1)) for h in logits]
        yaw_latent = actor["yaw_mean"]
        if sample_continuous_yaw:
            std = actor["yaw_log_std"].clamp(-4.0, 0.0).exp()
            yaw_latent = torch.distributions.Normal(yaw_latent, std).sample()
        yaw = float(YAW_LIMIT * torch.tanh(yaw_latent)[0])
        pitch = 0.0
        if action_schema in (CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                              V21_ACTION_SCHEMA):
            pitch_latent = actor["pitch_mean"]
            if sample_continuous_pitch:
                std = actor["pitch_log_std"].clamp(-4.0, 0.0).exp()
                pitch_latent = torch.distributions.Normal(pitch_latent, std).sample()
            pitch = float(PITCH_LIMIT * torch.tanh(pitch_latent)[0])
        if action_schema in (V2_ACTION_SCHEMA, V21_ACTION_SCHEMA):
            combat = indices[5]
            return [FWD[indices[0]], STRAFE[indices[1]], yaw, pitch,
                    float(indices[2]), float(indices[3]), float(combat == 1),
                    float(indices[4]), float(combat == 2)]
        return [FWD[indices[0]], STRAFE[indices[1]], yaw, pitch,
                float(indices[2]), float(indices[3]), float(indices[4]), 0.0, 0.0]
    logits = actor
    if stochastic:
        indices = [int(torch.distributions.Categorical(logits=h).sample()) for h in logits]
    else:
        indices = [int(h.argmax(-1)) for h in logits]
    if action_schema == FINE_ACTION_SCHEMA:
        return [FWD[indices[0]], STRAFE[indices[1]],
                YAW[indices[2]] + YAW_FINE[indices[3]], 0.0,
                float(indices[4]), float(indices[5]), float(indices[6])]
    if action_schema == LEGACY_ACTION_SCHEMA:
        return [LEGACY_HEAD_VALUES[i][index] for i, index in enumerate(indices)]
    raise ValueError(f"unsupported action schema {action_schema}")


def decode_client_player(row):
    """Translate the ordinary client observation into the PvP policy schema.

    hurtTime counts down from 10 client ticks while the simulator/oracle input is
    the server's 20-tick hurt-resistant timer.  Doubling it is the closest
    client-visible equivalent; all other fields are direct vanilla state.
    """
    return {
        "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
        "mx": float(row["vx"]), "my": float(row["vy"]), "mz": float(row["vz"]),
        "yaw": float(row["yaw"]), "pitch": float(row["pitch"]),
        "health": float(row["health"]), "cooldown": float(row["attack_cooldown"]),
        "hurt": min(20, 2 * int(row.get("hurt_time", 0))),
        "on_ground": bool(row["on_ground"]), "sprinting": bool(row["sprinting"]),
        "dead": bool(row["dead"]), "deaths": int(row.get("deaths", 0)),
        "policy_action_seq": int(row.get("policy_action_seq", 0)),
        "weapon": int(row.get("held_slot", 0) == 1),
        "blocking": bool(row.get("blocking", False)),
        "using_shield": bool(row.get("using_shield", False)),
        "shield_use_ticks": int(row.get("shield_use_ticks", 0)),
        "shield_disabled": bool(row.get("shield_disabled", False)),
        "shield_damage": int(row.get("shield_damage", 0)),
        "client_tick": int(row.get("client_tick", -1)),
        "world_tick": int(row.get("world_tick", -1)),
        "action_apply_client_tick": int(row.get("action_apply_client_tick", -1)),
        "action_apply_world_tick": int(row.get("action_apply_world_tick", -1)),
        "action_apply_nano_time": int(row.get("action_apply_nano_time", -1)),
    }


def timing_summary(rows):
    """Summarize observed action cadence and two-client phase parity."""
    if not rows:
        return {"decisions": 0}

    def percentile(values, q):
        return float(np.percentile(np.asarray(values, dtype=np.float64), q)) \
            if values else None

    starts = [row["decision_started_ns"] for row in rows]
    intervals = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
    rtts = [[row["clients"][role]["rtt_ms"] for row in rows]
            for role in range(2)]
    client_ticks = [[row["clients"][role]["client_tick"] for row in rows]
                    for role in range(2)]
    world_ticks = [[row["clients"][role]["action_apply_world_tick"] for row in rows]
                   for role in range(2)]
    client_deltas = [[b - a for a, b in zip(ticks, ticks[1:])]
                     for ticks in client_ticks]
    world_skew = [abs(row["clients"][0]["action_apply_world_tick"] -
                      row["clients"][1]["action_apply_world_tick"])
                  for row in rows
                  if min(row["clients"][0]["action_apply_world_tick"],
                         row["clients"][1]["action_apply_world_tick"]) >= 0]
    elapsed_s = ((max(row["clients"][r]["received_ns"] for row in rows for r in range(2)) -
                  min(row["clients"][r]["sent_ns"] for row in rows for r in range(2))) / 1e9)
    return {
        "decisions": len(rows),
        "effective_hz": len(rows) / max(elapsed_s, 1e-9),
        "decision_interval_ms": {
            "mean": float(np.mean(intervals)) if intervals else None,
            "p50": percentile(intervals, 50), "p95": percentile(intervals, 95),
            "p99": percentile(intervals, 99), "max": max(intervals) if intervals else None,
        },
        "client_rtt_ms": [{"mean": float(np.mean(x)), "p95": percentile(x, 95),
                           "p99": percentile(x, 99), "max": max(x)} for x in rtts],
        "client_tick_delta": [{
            "min": min(x) if x else None, "max": max(x) if x else None,
            "exactly_one_fraction": (sum(v == 1 for v in x) / len(x)) if x else None,
            "skipped": sum(v > 1 for v in x), "duplicated_or_reordered": sum(v <= 0 for v in x),
        } for x in client_deltas],
        "action_world_tick_skew": {
            "same_tick_fraction": (sum(v == 0 for v in world_skew) / len(world_skew)
                                   if world_skew else None),
            "within_one_tick_fraction": (sum(v <= 1 for v in world_skew) / len(world_skew)
                                         if world_skew else None),
            "p95_ticks": percentile(world_skew, 95),
            "max_ticks": max(world_skew) if world_skew else None,
        },
    }


def client_action(raw, client_attack=False):
    forward, strafe, dyaw, dpitch, jump, sprint, attack = raw[:7]
    weapon = int(raw[7]) if len(raw) > 7 else 0
    block = int(raw[8]) if len(raw) > 8 else 0
    return {"forward": int(forward > 0), "back": int(forward < 0),
            "left": int(strafe < 0), "right": int(strafe > 0),
            "dyaw": dyaw, "dpitch": dpitch, "jump": int(jump),
            "sprint": int(sprint), "attack": int(client_attack and attack > 0.5),
            "attack_once": int(client_attack and attack > 0.5),
            "use": block, "hotbar": weapon}


def remote_player_visibility(row):
    """Return visible remote players from one real client's render-world obs."""
    return [entity for entity in row.get("entities", [])
            if "Player" in entity.get("type", "")
            and not entity.get("dead", False)
            and not entity.get("invisible", False)
            and not entity.get("invisible_to_viewer", False)]


def wait_for_mutual_visibility(pool, timeout=10.0):
    """Gate capture/deployment until both clients can actually render the rival.

    Server presence is insufficient: a stale client entity flag or a delayed
    spawn packet can otherwise produce a valid fight and a one-sided empty video.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        futures = [pool.submit(bridge, 25575 + role, {"cmd": "obs"})
                   for role in range(2)]
        last = [future.result() for future in futures]
        if all(remote_player_visibility(row) for row in last):
            return last
        time.sleep(0.05)
    diagnostics = [{
        "entity_count": row.get("entity_count"),
        "players": [entity for entity in row.get("entities", [])
                    if "Player" in entity.get("type", "")],
    } for row in (last or [{}, {}])]
    raise RuntimeError("real-client mutual-visibility preflight failed: "
                       + json.dumps(diagnostics, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/pilots/pilot4/selfplay.pt")
    parser.add_argument("--decisions", type=int, default=300)
    parser.add_argument("--repeat-seconds", type=float,
                        help="defaults to the checkpoint's control frequency")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--sample-continuous-yaw", action="store_true",
                        help="sample yaw too; default uses its smooth mean")
    parser.add_argument("--sample-continuous-pitch", action="store_true",
                        help="sample pitch too; default uses its smooth mean")
    parser.add_argument("--stop-on-death", action="store_true",
                        help="record the terminal state and end after the first fight")
    parser.add_argument("--realtime-client-path", action="store_true",
                        help="one concurrent vanilla client step per decision (nominal 20 Hz)")
    parser.add_argument("--no-setup", action="store_true",
                        help="continue the currently loaded arena fight")
    parser.add_argument("--setup-seed", type=int, default=130013,
                        help="deterministic asymmetric real-match reset")
    parser.add_argument("--prepare-video", action="store_true",
                        help="apply the real-client recording profile before arena setup")
    parser.add_argument("--render-warmup-seconds", type=float, default=0.0,
                        help="after setup, hold the fighters still while both clients "
                             "finish chunk rebuild/JIT work before timed deployment")
    parser.add_argument("--ready-file", type=Path,
                        help="touch this after render warm-up, immediately before deployment")
    parser.add_argument("--start-file", type=Path,
                        help="when paired with --ready-file, wait for this file before fighting")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/java_pvp_deployment.jsonl")
    parser.add_argument("--timing-summary", type=Path,
                        help="defaults beside --out with suffix .timing.json")
    parser.add_argument("--simulated-ping-ms",
                        help="two comma-separated per-player baseline RTTs; applies "
                             "independent action/observation delay and +/-5%% variation")
    args = parser.parse_args()
    simulated_ping = None
    if args.simulated_ping_ms:
        simulated_ping = [float(x) for x in args.simulated_ping_ms.split(",")]
        if len(simulated_ping) != 2 or any(x < 0 or x > 200 for x in simulated_ping):
            parser.error("--simulated-ping-ms requires two values in [0,200]")
        if checkpoint_action_schema(torch.load(
                args.checkpoint, map_location="cpu", weights_only=False).get("config", {})) \
                != V21_ACTION_SCHEMA:
            parser.error("simulated ping requires a latency-aware V2.1 checkpoint")
    if (args.ready_file is None) != (args.start_file is None):
        parser.error("--ready-file and --start-file must be used together")

    # These policies are tiny MLPs.  A multithreaded BLAS dispatch costs more
    # than the matrix multiplies and makes the dual-client tick barrier jittery.
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    legacy_observation = config.get("obs_basis") != "movement_v2"
    action_schema = checkpoint_action_schema(config)
    # A bridge step already blocks until the following client tick. Fine-control
    # checkpoints therefore need no additional sleep to run at nominal 20 Hz.
    repeat_seconds = (args.repeat_seconds if args.repeat_seconds is not None else
                      (0.0 if action_schema in
                       (FINE_ACTION_SCHEMA, CONTINUOUS_ACTION_SCHEMA,
                        CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                        V21_ACTION_SCHEMA) else 0.2))
    policies = [Policy(action_schema=action_schema).eval(),
                Policy(action_schema=action_schema).eval()]
    for role in range(2):
        policies[role].load_state_dict(checkpoint["models"][role])
    bridge(25575, {"cmd": "overclock", "action": {"ms": 50}})
    if args.prepare_video:
        # Apply this before setup so the teleport rebuilds only the nearby arena.
        with ThreadPoolExecutor(max_workers=2) as prepare_pool:
            prepared = [prepare_pool.submit(bridge, 25575 + r,
                                             {"cmd": "video_prepare"})
                        for r in range(2)]
            for result in prepared:
                result.result()
    if not args.no_setup:
        rng = np.random.default_rng(args.setup_seed)
        bridge(25575, {"cmd": "pvp_setup", "action": {
            "lateral0": float(rng.uniform(-0.75, 0.75)),
            "lateral1": float(rng.uniform(-0.75, 0.75)),
            "yaw_delta0": float(rng.uniform(-7.5, 7.5)),
            "yaw_delta1": float(rng.uniform(-7.5, 7.5)),
        }})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"hits0": 0, "hits1": 0, "damage0": 0.0, "damage1": 0.0,
              "deaths0": 0, "deaths1": 0}
    completed_decisions = 0
    timing_rows = []
    with args.out.open("w") as receipt, ThreadPoolExecutor(max_workers=2) as pool:
        # pvp_setup teleports both players and invalidates a large part of each
        # render view. Recording immediately used to catch chunk rebuild and JVM
        # warm-up as several seconds of duplicate/low-FPS frames. Polling both
        # clients at tick rate holds gameplay still while allowing their render
        # loops to settle. Deployment timing deliberately starts afterwards.
        warmup_started = time.monotonic()
        while time.monotonic() - warmup_started < args.render_warmup_seconds:
            warm = [pool.submit(bridge, 25575 + r, {"cmd": "obs"})
                    for r in range(2)]
            for future in warm:
                future.result()
        # This is a deployment correctness gate, not just a video convenience:
        # both policies may keep fighting through server state even if one remote
        # player has failed to materialize in a client's render world.
        visible_rows = wait_for_mutual_visibility(pool)
        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.touch()
            while not args.start_file.exists():
                time.sleep(0.005)
        started = time.time()
        client_players = None
        policy_history = None
        simulated_uplink = None
        persistent = None
        if args.realtime_client_path:
            client_players = [decode_client_player(row) for row in visible_rows]
            if action_schema == V21_ACTION_SCHEMA:
                policy_history = PolicyObservationHistory(
                    client_players, visible_rows, simulated_ping_ms=simulated_ping)
                if simulated_ping is not None:
                    simulated_uplink = SimulatedActionUplink(policy_history)
            persistent = [PersistentBridge(25575 + role) for role in range(2)]
        try:
          for decision in range(args.decisions):
            if args.realtime_client_path:
                decision_started_ns = time.monotonic_ns()
                players = client_players
                raw = [policy_action(
                           policies[r], (policy_history.encode(r)
                                         if policy_history is not None else
                                         observation(players, r, legacy=legacy_observation)),
                           args.stochastic, action_schema,
                           args.sample_continuous_yaw,
                           args.sample_continuous_pitch)
                       for r in range(2)]
                executed_raw = (simulated_uplink.submit(raw)
                                if simulated_uplink is not None else raw)
                futures = [pool.submit(persistent[r].call,
                                       {"cmd": "step", "action":
                                        client_action(executed_raw[r], client_attack=True)})
                           for r in range(2)]
                results = [f.result() for f in futures]
                next_players = [decode_client_player(result[0]) for result in results]
                if policy_history is not None:
                    policy_history.update(next_players, [result[0] for result in results])
                client_timing = []
                for role, (raw_obs, transport) in enumerate(results):
                    client_timing.append({**transport,
                        "client_tick": int(raw_obs.get("client_tick", -1)),
                        "world_tick": int(raw_obs.get("world_tick", -1)),
                        "action_apply_client_tick": int(raw_obs.get("action_apply_client_tick", -1)),
                        "action_apply_world_tick": int(raw_obs.get("action_apply_world_tick", -1)),
                        "policy_action_seq": int(raw_obs.get("policy_action_seq", -1))})
                timing_row = {"decision": decision,
                              "decision_started_ns": decision_started_ns,
                              "clients": client_timing}
                timing_rows.append(timing_row)
                killed_role = next((r for r in range(2)
                                    if next_players[r]["deaths"] > players[r]["deaths"]), None)
                for attacker in range(2):
                    victim = 1 - attacker
                    damage = max(0.0, players[victim]["health"] -
                                 next_players[victim]["health"])
                    if damage > 0.0:
                        totals[f"hits{attacker}"] += 1
                        totals[f"damage{attacker}"] += damage
                if killed_role is not None:
                    totals[f"deaths{killed_role}"] += 1
                row = {
                    "decision": decision,
                    "health": [p["health"] for p in next_players],
                    "actions": raw, "terminal": killed_role is not None,
                    "executed_actions": executed_raw,
                    "policy_ping_ms": ([policy_history.ping_ms[r] for r in range(2)]
                                       if policy_history is not None else None),
                    "killed_role": killed_role,
                    "policy_action_seq": [p["policy_action_seq"] for p in next_players],
                    "timing": timing_row,
                    "path": "realtime_client", **totals,
                }
                receipt.write(json.dumps(row, sort_keys=True) + "\n")
                receipt.flush()
                completed_decisions += 1
                client_players = next_players
                if killed_role is not None and args.stop_on_death:
                    break
                if repeat_seconds > 0.0:
                    time.sleep(repeat_seconds)
                continue
            state = bridge(25575, {"cmd": "pvp_state"})
            players = [decode_player(row) for row in state["players"]]
            if players[0]["dead"] or players[1]["dead"]:
                totals["deaths0"] += int(players[0]["dead"])
                totals["deaths1"] += int(players[1]["dead"])
                receipt.write(json.dumps({
                    "decision": decision, "terminal": True,
                    "server_tick": state["server_tick"],
                    "health": [p["health"] for p in players], **totals,
                }, sort_keys=True) + "\n")
                receipt.flush()
                if args.stop_on_death:
                    break
                bridge(25575, {"cmd": "pvp_setup"})
                continue
            raw = [policy_action(
                       policies[r], observation(players, r, legacy=legacy_observation),
                       args.stochastic, action_schema,
                       args.sample_continuous_yaw,
                       args.sample_continuous_pitch)
                   for r in range(2)]
            futures = [pool.submit(bridge, 25575 + r,
                                   {"cmd": "step", "action": client_action(raw[r])})
                       for r in range(2)]
            for future in futures:
                future.result()
            attacks = []
            killed_role = None
            for role in range(2):
                if raw[role][6] > 0.5:
                    result = bridge(25575, {"cmd": "pvp_attack", "action": {"role": role}})
                    attacks.append(result)
                    if result["accepted"]:
                        damage = f32(result["health_before_bits"]) - f32(result["health_after_bits"])
                        totals[f"hits{role}"] += 1
                        totals[f"damage{role}"] += damage
                        if f32(result["health_after_bits"]) <= 0.0:
                            killed_role = 1 - role
                            totals[f"deaths{killed_role}"] += 1
            row = {"decision": decision, "server_tick": state["server_tick"],
                   "health": [p["health"] for p in players], "actions": raw,
                   "attacks": attacks, "terminal": killed_role is not None,
                   "killed_role": killed_role, **totals}
            receipt.write(json.dumps(row, sort_keys=True) + "\n")
            receipt.flush()
            completed_decisions += 1
            # Headless clients auto-respawn before the next state read, so the
            # authoritative attack receipt is the lossless terminal edge.
            if killed_role is not None and args.stop_on_death:
                break
            if repeat_seconds > 0.0:
                time.sleep(repeat_seconds)
        finally:
            if persistent is not None:
                for client in persistent:
                    client.close()
    elapsed = time.time() - started
    parity = timing_summary(timing_rows) if args.realtime_client_path else None
    timing_path = args.timing_summary or args.out.with_suffix(".timing.json")
    if parity is not None:
        timing_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "decisions": completed_decisions, **totals,
                      "action_schema": action_schema,
                      "deployment_path": ("realtime_client" if args.realtime_client_path
                                          else "canonical_oracle"),
                      "control_hz": config.get("control_hz", 5),
                      "wall_seconds": elapsed,
                      "measured_decisions_per_second": completed_decisions / max(elapsed, 1e-9),
                      "timing_summary": parity,
                      "timing_summary_path": str(timing_path) if parity is not None else None,
                      "receipt": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
