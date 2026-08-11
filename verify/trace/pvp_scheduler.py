"""Deterministic two-role action scheduler and PvP trace contract.

This pure-Python reference is the executable contract for the future 1.11.2
Forge server bridge. Socket arrival order is deliberately erased before a
server-tick batch is emitted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


ROLES = (0, 1)
ACTION_KEYS = ("forward", "strafe", "dyaw", "dpitch", "jump", "sprint", "attack")
PLAYER_KEYS = (
    "role", "uuid", "name", "entity_id", "position_bits", "motion_bits",
    "rotation_bits", "on_ground", "health_bits", "absorption_bits", "food",
    "saturation_bits", "inventory", "held_slot", "armor", "effects",
    "sprinting", "sneaking", "hurt_resistant_time", "dead",
    "attack_cooldown_bits",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def row_hash(value):
    """Stable receipt hash used by both fixture tests and the Java bridge."""
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def noop_action():
    return {key: 0 for key in ACTION_KEYS}


@dataclass(frozen=True)
class Rejection:
    episode: int
    tick: int
    role: int
    reason: str


class PvpScheduler:
    """Collect unordered socket messages and freeze canonical server batches."""

    def __init__(self, episode, first_tick=0):
        self.episode = int(episode)
        self.next_tick = int(first_tick)
        self.pending = {}
        self.rejections = []

    def submit(self, message):
        episode = int(message["episode"])
        tick = int(message["tick"])
        role = int(message["role"])
        if episode != self.episode:
            return self._reject(episode, tick, role, "wrong_episode")
        if role not in ROLES:
            return self._reject(episode, tick, role, "unknown_role")
        if tick < self.next_tick:
            return self._reject(episode, tick, role, "late")
        key = (tick, role)
        if key in self.pending:
            return self._reject(episode, tick, role, "duplicate")
        action = message["action"]
        if set(action) != set(ACTION_KEYS):
            return self._reject(episode, tick, role, "invalid_action")
        self.pending[key] = {key: action[key] for key in ACTION_KEYS}
        return True

    def _reject(self, episode, tick, role, reason):
        self.rejections.append(Rejection(episode, tick, role, reason))
        return False

    def freeze(self, allow_timeout=False):
        """Emit role 0 then role 1 for exactly ``next_tick``.

        Missing actions stall by default. With ``allow_timeout``, they become
        explicit no-ops and are represented in ``timed_out_roles``.
        """
        tick = self.next_tick
        missing = [r for r in ROLES if (tick, r) not in self.pending]
        if missing and not allow_timeout:
            return None
        actions = []
        for role in ROLES:
            action = self.pending.pop((tick, role), None)
            actions.append({"role": role,
                            "action": noop_action() if action is None else action})
        batch = {"episode": self.episode, "tick": tick, "actions": actions,
                 "timed_out_roles": missing}
        self.next_tick += 1
        return batch


def validate_trace_rows(rows):
    """Validate a complete PvP JSONL receipt; raise ValueError on drift."""
    if not rows or rows[0].get("type") != "pvp_header":
        raise ValueError("first row must be pvp_header")
    header = rows[0]
    for key in ("schema", "episode", "world_snapshot_sha256", "config_sha256",
                "roles"):
        if key not in header:
            raise ValueError(f"header missing {key}")
    if [r.get("role") for r in header["roles"]] != [0, 1]:
        raise ValueError("header roles must be canonical [0, 1]")
    identities = [(r.get("uuid"), r.get("name")) for r in header["roles"]]
    expected_tick = 0
    for row in rows[1:]:
        if row.get("type") != "pvp_tick" or row.get("tick") != expected_tick:
            raise ValueError("ticks must be contiguous pvp_tick rows")
        if row.get("episode") != header["episode"]:
            raise ValueError("episode changed")
        players = row.get("players", [])
        if len(players) != 2:
            raise ValueError("tick must contain two players")
        for role, player in enumerate(players):
            missing = set(PLAYER_KEYS) - set(player)
            if missing:
                raise ValueError(f"player {role} missing {sorted(missing)}")
            if player["role"] != role or (player["uuid"], player["name"]) != identities[role]:
                raise ValueError("role identity changed")
        actions = row.get("actions", [])
        if [a.get("role") for a in actions] != [0, 1]:
            raise ValueError("actions must be resolved in role order")
        if not all("accepted" in a and "action" in a for a in actions):
            raise ValueError("action outcome missing")
        if "combat_events" not in row:
            raise ValueError("combat events missing")
        expected_tick += 1
    return True
