"""Pure timing-parity reducer tests; no Minecraft process required."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_deploy_module():
    sys.path.insert(0, str(ROOT / "blaze" / "pvp"))
    spec = importlib.util.spec_from_file_location(
        "deploy_pvp_checkpoint", ROOT / "java" / "deploy_pvp_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perfect_20hz_trace_passes_tick_and_phase_metrics():
    timing_summary = load_deploy_module().timing_summary
    rows = []
    for i in range(10):
        start = i * 50_000_000
        clients = []
        for role in range(2):
            clients.append({"sent_ns": start, "received_ns": start + 45_000_000,
                            "rtt_ms": 45.0, "client_tick": 100 + i,
                            "action_apply_world_tick": 200 + i})
        rows.append({"decision_started_ns": start, "clients": clients})
    out = timing_summary(rows)
    assert np.isclose(out["decision_interval_ms"]["mean"], 50.0)
    assert out["client_tick_delta"][0]["exactly_one_fraction"] == 1.0
    assert out["client_tick_delta"][1]["skipped"] == 0
    assert out["action_world_tick_skew"]["same_tick_fraction"] == 1.0


def test_reducer_detects_skipped_ticks_and_cross_client_skew():
    timing_summary = load_deploy_module().timing_summary
    rows = []
    ticks0 = [10, 11, 13]
    ticks1 = [20, 21, 22]
    for i in range(3):
        start = i * 60_000_000
        rows.append({"decision_started_ns": start, "clients": [
            {"sent_ns": start, "received_ns": start + 55_000_000,
             "rtt_ms": 55.0, "client_tick": ticks0[i],
             "action_apply_world_tick": 30 + i},
            {"sent_ns": start, "received_ns": start + 55_000_000,
             "rtt_ms": 55.0, "client_tick": ticks1[i],
             "action_apply_world_tick": 32 + i},
        ]})
    out = timing_summary(rows)
    assert out["client_tick_delta"][0]["skipped"] == 1
    assert out["action_world_tick_skew"]["same_tick_fraction"] == 0.0
    assert out["action_world_tick_skew"]["max_ticks"] == 2
