import copy
import itertools

import pytest

import pvp_scheduler as ps


def msg(tick, role, attack=0):
    action = ps.noop_action()
    action["attack"] = attack
    return {"episode": 7, "tick": tick, "role": role, "action": action}


def test_socket_arrival_order_cannot_change_frozen_batch():
    messages = [msg(0, 0, 1), msg(0, 1), msg(1, 0), msg(1, 1, 1)]
    receipts = set()
    for order in itertools.permutations(messages):
        scheduler = ps.PvpScheduler(7)
        for item in order:
            assert scheduler.submit(item)
        receipt = [scheduler.freeze(), scheduler.freeze()]
        receipts.add(ps.canonical_json(receipt))
    assert len(receipts) == 1


def test_missing_duplicate_late_and_wrong_episode_are_explicit():
    scheduler = ps.PvpScheduler(7)
    assert scheduler.submit(msg(0, 0))
    assert not scheduler.submit(msg(0, 0))
    assert scheduler.freeze() is None
    batch = scheduler.freeze(allow_timeout=True)
    assert batch["timed_out_roles"] == [1]
    assert batch["actions"][1]["action"] == ps.noop_action()
    assert not scheduler.submit(msg(0, 1))
    wrong = msg(1, 1); wrong["episode"] = 8
    assert not scheduler.submit(wrong)
    assert [r.reason for r in scheduler.rejections] == [
        "duplicate", "late", "wrong_episode"]


def player(role):
    row = {key: 0 for key in ps.PLAYER_KEYS}
    row.update({"role": role, "uuid": f"uuid-{role}", "name": f"p{role}",
                "position_bits": [0, 0, 0], "motion_bits": [0, 0, 0],
                "rotation_bits": [0, 0], "inventory": [], "armor": [],
                "effects": []})
    return row


def valid_trace():
    return [
        {"type": "pvp_header", "schema": 1, "episode": 7,
         "world_snapshot_sha256": "a" * 64, "config_sha256": "b" * 64,
         "roles": [{"role": 0, "uuid": "uuid-0", "name": "p0"},
                   {"role": 1, "uuid": "uuid-1", "name": "p1"}]},
        {"type": "pvp_tick", "episode": 7, "tick": 0,
         "actions": [{"role": 0, "accepted": True, "action": ps.noop_action()},
                     {"role": 1, "accepted": True, "action": ps.noop_action()}],
         "players": [player(0), player(1)], "combat_events": []},
    ]


def test_trace_contract_requires_two_stable_complete_player_records():
    rows = valid_trace()
    assert ps.validate_trace_rows(rows)
    assert ps.row_hash(rows[1]) == ps.row_hash(copy.deepcopy(rows[1]))
    broken = copy.deepcopy(rows)
    del broken[1]["players"][1]["health_bits"]
    with pytest.raises(ValueError, match="health_bits"):
        ps.validate_trace_rows(broken)
    broken = copy.deepcopy(rows)
    broken[1]["players"][1]["uuid"] = "reconnected-as-someone-else"
    with pytest.raises(ValueError, match="identity"):
        ps.validate_trace_rows(broken)
