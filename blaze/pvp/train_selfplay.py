"""PPO curriculum and adversarial self-play for the two-player PvP arena.

Role 0 first learns engagement against stationary and scripted opponents. Its
weights are then cloned into role 1 and the two policies optimize independently,
avoiding symmetric zero-sum gradient cancellation. Historical league opponents
remain a later step, after a policy demonstrates competence against fixed tests.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pvp import N_ACT, N_NETWORK_OBS, NetworkVecPvp, VecPvp

HEADS = (3, 3, 3, 3, 2, 2, 2)
CONTINUOUS_HEADS = (3, 3, 2, 2, 2)
CONTINUOUS_DISCRETE_COLS = (0, 1, 4, 5, 6)
V2_HEADS = (3, 3, 2, 2, 2, 3)
V2_DISCRETE_COLS = (0, 1, 4, 5, 7, 6)
FWD = (-1.0, 0.0, 1.0)
STRAFE = (-1.0, 0.0, 1.0)
YAW = (-15.0, 0.0, 15.0)
PITCH = (-10.0, 0.0, 10.0)
YAW_FINE = (-5.0, 0.0, 5.0)
LEGACY_ACTION_SCHEMA = "legacy_5hz_v1"
FINE_ACTION_SCHEMA = "fine_yaw_20hz_v2"
CONTINUOUS_ACTION_SCHEMA = "continuous_yaw_20hz_v3"
CONTINUOUS_LOOK_ACTION_SCHEMA = "continuous_look_20hz_v4"
V2_ACTION_SCHEMA = "iron_gear_20hz_v5"
V21_ACTION_SCHEMA = "iron_gear_latency_20hz_v6"
YAW_LIMIT = 20.0
PITCH_LIMIT = 10.0


def is_continuous_schema(action_schema):
    return action_schema in (CONTINUOUS_ACTION_SCHEMA, CONTINUOUS_LOOK_ACTION_SCHEMA,
                             V2_ACTION_SCHEMA, V21_ACTION_SCHEMA)


def has_continuous_pitch(action_schema):
    return action_schema in (CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                             V21_ACTION_SCHEMA)


def is_iron_gear_schema(action_schema):
    return action_schema in (V2_ACTION_SCHEMA, V21_ACTION_SCHEMA)


def categorical_contract(action_schema):
    if is_iron_gear_schema(action_schema):
        return V2_HEADS, V2_DISCRETE_COLS
    return CONTINUOUS_HEADS, CONTINUOUS_DISCRETE_COLS


def action_observation_dim(action_schema):
    if action_schema == V21_ACTION_SCHEMA:
        return N_NETWORK_OBS
    if action_schema == V2_ACTION_SCHEMA:
        return 35
    return 25 if action_schema == CONTINUOUS_LOOK_ACTION_SCHEMA else 24


def checkpoint_action_schema(config):
    """Missing schema means the frozen pilot-10-era 5 Hz contract."""
    schema = config.get("action_schema", LEGACY_ACTION_SCHEMA)
    if schema not in (LEGACY_ACTION_SCHEMA, FINE_ACTION_SCHEMA,
                      CONTINUOUS_ACTION_SCHEMA, CONTINUOUS_LOOK_ACTION_SCHEMA,
                      V2_ACTION_SCHEMA, V21_ACTION_SCHEMA):
        raise ValueError(f"unsupported action schema {schema}")
    return schema


class Policy(nn.Module):
    def __init__(self, hidden=128, action_schema=LEGACY_ACTION_SCHEMA, obs_dim=None):
        super().__init__()
        self.action_schema = action_schema
        self.obs_dim = obs_dim or action_observation_dim(action_schema)
        self.body = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        heads = (categorical_contract(action_schema)[0]
                 if is_continuous_schema(action_schema) else HEADS)
        self.actor = nn.Linear(hidden, sum(heads))
        if is_continuous_schema(action_schema):
            self.yaw_mean = nn.Linear(hidden, 1)
            # About 2.7 degrees near zero at initialization. PPO may adapt it,
            # but deployment can use the mean while sampling tactical heads.
            self.yaw_log_std = nn.Parameter(torch.tensor([-2.0]))
        if has_continuous_pitch(action_schema):
            self.pitch_mean = nn.Linear(hidden, 1)
            self.pitch_log_std = nn.Parameter(torch.tensor([-2.0]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        raw = self.actor(h)
        if is_continuous_schema(self.action_schema):
            actor = {"categorical": torch.split(
                         raw, categorical_contract(self.action_schema)[0], dim=-1),
                     "yaw_mean": self.yaw_mean(h).squeeze(-1),
                     "yaw_log_std": self.yaw_log_std.expand(h.shape[0])}
            if has_continuous_pitch(self.action_schema):
                actor.update({"pitch_mean": self.pitch_mean(h).squeeze(-1),
                              "pitch_log_std": self.pitch_log_std.expand(h.shape[0])})
        else:
            actor = torch.split(raw, HEADS, dim=-1)
        return actor, self.critic(h).squeeze(-1)


def transfer_v1_look_policy(target, source_state):
    """Copy V1.2 locomotion/look into the expanded V2 policy.

    New observation columns start with zero influence and weapon/combat logits
    remain freshly initialized. This preserves aiming without pretending the old
    boxing attack head understands shields.
    """
    if target.action_schema != V2_ACTION_SCHEMA:
        raise ValueError("V1 transfer target must use the V2 action schema")
    own = target.state_dict()
    for name in ("body.0.bias", "body.2.weight", "body.2.bias",
                 "yaw_mean.weight", "yaw_mean.bias", "yaw_log_std",
                 "pitch_mean.weight", "pitch_mean.bias", "pitch_log_std",
                 "critic.weight", "critic.bias"):
        own[name].copy_(source_state[name])
    own["body.0.weight"].zero_()
    old_in = source_state["body.0.weight"].shape[1]
    own["body.0.weight"][:, :old_in].copy_(source_state["body.0.weight"])
    # Forward, strafe, jump and sprint occupy the first ten logits in both actors.
    own["actor.weight"][:10].copy_(source_state["actor.weight"][:10])
    own["actor.bias"][:10].copy_(source_state["actor.bias"][:10])
    target.load_state_dict(own)


def transfer_v2_latency_policy(target, source_state):
    """Expand a 35-input V2 fighter to four delayed frames plus own RTT.

    The newest delivered frame starts with the complete V2 behavior. Historical
    frames and ping begin at zero influence and are learned during V2.1 training.
    """
    if target.action_schema != V21_ACTION_SCHEMA:
        raise ValueError("latency transfer target must use the V2.1 schema")
    own = target.state_dict()
    for name, value in source_state.items():
        if name != "body.0.weight":
            own[name].copy_(value)
    own["body.0.weight"].zero_()
    own["body.0.weight"][:, :35].copy_(source_state["body.0.weight"])
    target.load_state_dict(own)


def policy_input(policy, obs):
    return obs[..., :policy.obs_dim]


def _look_distribution(actor, name):
    return Normal(actor[f"{name}_mean"],
                  actor[f"{name}_log_std"].clamp(-4.0, 0.0).exp())


def _look_log_prob(actor, name, action, limit):
    scaled = (action / limit).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    latent = torch.atanh(scaled)
    return (_look_distribution(actor, name).log_prob(latent)
            - torch.log(limit * (1.0 - scaled.square()) + 1e-6))


def sample_actions(actor, action_schema=LEGACY_ACTION_SCHEMA,
                   deterministic_yaw=False, deterministic_pitch=False):
    if is_continuous_schema(action_schema):
        actions = torch.zeros((actor["yaw_mean"].shape[0], N_ACT),
                              dtype=torch.float32, device=actor["yaw_mean"].device)
        logps, entropy = [], []
        _, categorical_cols = categorical_contract(action_schema)
        for head, col in zip(actor["categorical"], categorical_cols):
            dist = Categorical(logits=head)
            a = dist.sample()
            actions[:, col] = a
            logps.append(dist.log_prob(a))
            entropy.append(dist.entropy())
        yaw_dist = _look_distribution(actor, "yaw")
        latent = actor["yaw_mean"] if deterministic_yaw else yaw_dist.rsample()
        actions[:, 2] = YAW_LIMIT * torch.tanh(latent)
        logps.append(_look_log_prob(actor, "yaw", actions[:, 2], YAW_LIMIT))
        # The latent entropy is a stable exploration proxy. The exact squashed
        # entropy has no simple closed form and is not needed by PPO ratios.
        entropy.append(yaw_dist.entropy())
        if has_continuous_pitch(action_schema):
            pitch_dist = _look_distribution(actor, "pitch")
            latent = (actor["pitch_mean"] if deterministic_pitch
                      else pitch_dist.rsample())
            actions[:, 3] = PITCH_LIMIT * torch.tanh(latent)
            logps.append(_look_log_prob(
                actor, "pitch", actions[:, 3], PITCH_LIMIT))
            entropy.append(pitch_dist.entropy())
        return (actions, torch.stack(logps, dim=-1).sum(-1),
                torch.stack(entropy, dim=-1).sum(-1))
    logits = actor
    actions, logps, entropy = [], [], []
    for head in logits:
        dist = Categorical(logits=head)
        a = dist.sample()
        actions.append(a)
        logps.append(dist.log_prob(a))
        entropy.append(dist.entropy())
    return (torch.stack(actions, dim=-1), torch.stack(logps, dim=-1).sum(-1),
            torch.stack(entropy, dim=-1).sum(-1))


def greedy_actions(actor, action_schema=LEGACY_ACTION_SCHEMA):
    if is_continuous_schema(action_schema):
        actions = torch.zeros((actor["yaw_mean"].shape[0], N_ACT),
                              dtype=torch.float32, device=actor["yaw_mean"].device)
        _, categorical_cols = categorical_contract(action_schema)
        for head, col in zip(actor["categorical"], categorical_cols):
            actions[:, col] = head.argmax(-1)
        actions[:, 2] = YAW_LIMIT * torch.tanh(actor["yaw_mean"])
        if has_continuous_pitch(action_schema):
            actions[:, 3] = PITCH_LIMIT * torch.tanh(actor["pitch_mean"])
        return actions
    return torch.stack([head.argmax(-1) for head in actor], dim=-1)


def evaluate_actions(actor, actions, action_schema=LEGACY_ACTION_SCHEMA):
    if is_continuous_schema(action_schema):
        logps, entropy = [], []
        _, categorical_cols = categorical_contract(action_schema)
        for head, col in zip(actor["categorical"], categorical_cols):
            dist = Categorical(logits=head)
            logps.append(dist.log_prob(actions[:, col].long()))
            entropy.append(dist.entropy())
        yaw_dist = _look_distribution(actor, "yaw")
        logps.append(_look_log_prob(actor, "yaw", actions[:, 2], YAW_LIMIT))
        entropy.append(yaw_dist.entropy())
        if has_continuous_pitch(action_schema):
            pitch_dist = _look_distribution(actor, "pitch")
            logps.append(_look_log_prob(
                actor, "pitch", actions[:, 3], PITCH_LIMIT))
            entropy.append(pitch_dist.entropy())
        return (torch.stack(logps, dim=-1).sum(-1),
                torch.stack(entropy, dim=-1).sum(-1))
    logits = actor
    logps, entropy = [], []
    for i, head in enumerate(logits):
        dist = Categorical(logits=head)
        logps.append(dist.log_prob(actions[:, i]))
        entropy.append(dist.entropy())
    return torch.stack(logps, dim=-1).sum(-1), torch.stack(entropy, dim=-1).sum(-1)


def decode_actions(a, device, action_schema=LEGACY_ACTION_SCHEMA):
    rows = torch.zeros((*a.shape[:-1], N_ACT), dtype=torch.float64, device=device)
    rows[..., 0] = torch.tensor(FWD, dtype=torch.float64, device=device)[a[..., 0].long()]
    rows[..., 1] = torch.tensor(STRAFE, dtype=torch.float64, device=device)[a[..., 1].long()]
    if action_schema in (CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                          V21_ACTION_SCHEMA):
        rows[..., 2] = a[..., 2].to(torch.float64).clamp(-YAW_LIMIT, YAW_LIMIT)
        rows[..., 3] = a[..., 3].to(torch.float64).clamp(-PITCH_LIMIT, PITCH_LIMIT)
    elif action_schema == CONTINUOUS_ACTION_SCHEMA:
        rows[..., 2] = a[..., 2].to(torch.float64).clamp(-YAW_LIMIT, YAW_LIMIT)
        rows[..., 3] = 0.0
    elif action_schema == FINE_ACTION_SCHEMA:
        coarse = torch.tensor(YAW, dtype=torch.float64, device=device)[a[..., 2].long()]
        # Reuse the old pitch head as a fine yaw residual. The Cartesian sum is
        # exactly {-20,-15,-10,-5,0,5,10,15,20}; pitch is fixed for flat boxing.
        fine = torch.tensor(YAW_FINE, dtype=torch.float64, device=device)[a[..., 3].long()]
        rows[..., 2] = coarse + fine
        rows[..., 3] = 0.0
    elif action_schema == LEGACY_ACTION_SCHEMA:
        coarse = torch.tensor(YAW, dtype=torch.float64, device=device)[a[..., 2].long()]
        rows[..., 2] = coarse
        rows[..., 3] = torch.tensor(PITCH, dtype=torch.float64, device=device)[a[..., 3].long()]
    else:
        raise ValueError(f"unsupported action schema {action_schema}")
    # Frozen pre-V2 callers may still provide the original seven policy columns.
    tail = a.shape[-1] - 4
    rows[..., 4:4 + tail] = a[..., 4:].to(torch.float64)
    if is_iron_gear_schema(action_schema):
        # Policy col 6 is one categorical combat intent: idle/attack/block.
        rows[..., 6] = (a[..., 6] == 1).to(torch.float64)
        rows[..., 8] = (a[..., 6] == 2).to(torch.float64)
    return rows


def _teacher_pitch_delta(obs):
    """Aim from own eye to the opponent AABB center, then close pitch error."""
    horizontal = torch.sqrt((obs[:, 5] * 32.0).square()
                            + (obs[:, 6] * 32.0).square()).clamp_min(1e-4)
    target_center_from_eye = obs[:, 7] * 8.0 + 0.9 - 1.62
    desired_pitch = torch.atan2(-target_center_from_eye, horizontal) * (180.0 / np.pi)
    current_pitch = obs[:, 24] * 90.0
    return (desired_pitch - current_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT)


def scripted_action_indices(obs, attack=True, action_schema=LEGACY_ACTION_SCHEMA):
    """Categorical policy targets for the deterministic boxing teacher."""
    lateral = obs[:, 5]
    longitudinal = obs[:, 6]  # positive is in front
    dist = obs[:, 10] * 32.0
    bearing = torch.atan2(lateral.abs(), longitudinal)
    aligned_move = bearing < (30.0 * np.pi / 180.0)
    aligned_attack = bearing < (20.0 * np.pi / 180.0)
    if is_continuous_schema(action_schema):
        out = torch.zeros((obs.shape[0], N_ACT), dtype=torch.float32,
                          device=obs.device)
        out[:, 0] = 1
        out[:, 1] = 1
    else:
        out = torch.zeros((obs.shape[0], N_ACT), dtype=torch.long,
                          device=obs.device)
        out[:, :len(HEADS)] = 1
    # Brake before turning. Unconditional forward+sprint while the target was
    # behind taught the exact wide-orbit failure observed in real deployment.
    out[:, 0] = torch.where((dist > 1.7) & aligned_move, 2, 1)
    out[:, 1] = 1
    # Pure lateral error is zero when the target is exactly behind. Always turn
    # clockwise in that tie so crossing an opponent cannot create a blind state.
    if is_continuous_schema(action_schema):
        desired = torch.atan2(-lateral, longitudinal) * (180.0 / np.pi)
        desired = torch.where((longitudinal < 0.0) & (lateral.abs() < 0.01),
                              torch.full_like(desired, 20.0), desired)
        out[:, 2] = desired.clamp(-YAW_LIMIT, YAW_LIMIT)
        if has_continuous_pitch(action_schema):
            out[:, 3] = _teacher_pitch_delta(obs)
    elif action_schema == FINE_ACTION_SCHEMA:
        desired = torch.atan2(-lateral, longitudinal) * (180.0 / np.pi)
        desired = torch.where((longitudinal < 0.0) & (lateral.abs() < 0.01),
                              torch.full_like(desired, 20.0), desired)
        bins = ((desired.clamp(-20.0, 20.0) + 20.0) / 5.0).round().long()
        out[:, 2] = bins // 3
        out[:, 3] = bins % 3
    else:
        out[:, 2] = torch.where(longitudinal < 0.0, 2,
                                torch.where(lateral > 0.01, 0,
                                            torch.where(lateral < -0.01, 2, 1)))
        out[:, 3] = 1
    out[:, 4] = 0
    out[:, 5] = ((dist > 1.7) & aligned_move).long()
    ready_attack = ((dist < 3.0) & aligned_attack & (obs[:, 15] > 0.9))
    if is_iron_gear_schema(action_schema):
        opponent_blocking = obs[:, 28] > 0.5
        shield_available = obs[:, 29] <= 0.0
        # Keep use held through post-hit knockback; a shield needs five
        # consecutive use ticks before vanilla considers it blocking.
        threatened = (dist < 5.0) & aligned_attack
        out[:, 7] = opponent_blocking.long()  # axe into shield, sword otherwise
        out[:, 6] = torch.where(
            ready_attack & attack, torch.ones_like(out[:, 6]),
            torch.where(threatened & shield_available,
                        torch.full_like(out[:, 6], 2), torch.zeros_like(out[:, 6])))
    else:
        out[:, 6] = ready_attack.long() if attack else 0
    return out


def scripted_baseline(obs, device, attack=True,
                      action_schema=LEGACY_ACTION_SCHEMA):
    """Deterministic turn, approach, sprint, and optionally punch controller."""
    n = obs.shape[0]
    rows = torch.zeros((n, N_ACT), dtype=torch.float64, device=device)
    lateral = obs[:, 5]
    longitudinal = obs[:, 6]
    dist = obs[:, 10] * 32.0
    bearing = torch.atan2(lateral.abs(), longitudinal)
    aligned_move = bearing < (30.0 * np.pi / 180.0)
    aligned_attack = bearing < (20.0 * np.pi / 180.0)
    rows[:, 0] = ((dist > 1.7) & aligned_move).to(torch.float64)
    if is_continuous_schema(action_schema):
        desired = torch.atan2(-lateral, longitudinal) * (180.0 / np.pi)
        desired = torch.where((longitudinal < 0.0) & (lateral.abs() < 0.01),
                              torch.full_like(desired, 20.0), desired)
        rows[:, 2] = desired.clamp(-YAW_LIMIT, YAW_LIMIT).to(torch.float64)
        if has_continuous_pitch(action_schema):
            rows[:, 3] = _teacher_pitch_delta(obs).to(torch.float64)
    elif action_schema == FINE_ACTION_SCHEMA:
        desired = torch.atan2(-lateral, longitudinal) * (180.0 / np.pi)
        desired = torch.where((longitudinal < 0.0) & (lateral.abs() < 0.01),
                              torch.full_like(desired, 20.0), desired)
        rows[:, 2] = ((desired.clamp(-20.0, 20.0) / 5.0).round() * 5.0).to(torch.float64)
    else:
        rows[:, 2] = torch.where(
            longitudinal < 0.0, torch.tensor(15.0, device=device),
            torch.where(lateral > 0.01, torch.tensor(-15.0, device=device),
                        torch.where(lateral < -0.01, torch.tensor(15.0, device=device),
                                    torch.tensor(0.0, device=device)))).to(torch.float64)
    rows[:, 5] = ((dist > 1.7) & aligned_move).to(torch.float64)
    if is_iron_gear_schema(action_schema):
        opponent_blocking = obs[:, 28] > 0.5
        shield_available = obs[:, 29] <= 0.0
        ready = (dist < 3.0) & aligned_attack & (obs[:, 15] > 0.9)
        # Keep use held through post-hit knockback; a shield needs five
        # consecutive use ticks before vanilla considers it blocking.
        threatened = (dist < 5.0) & aligned_attack
        rows[:, 7] = opponent_blocking.to(torch.float64)
        rows[:, 6] = (ready & attack).to(torch.float64)
        rows[:, 8] = ((~ready | (not attack)) & threatened &
                      shield_available).to(torch.float64)
    elif attack:
        # Only swing when charged. This is stronger and more Minecraft-like
        # than holding attack every tick through hurt resistance.
        rows[:, 6] = ((dist < 3.0) & aligned_attack &
                      (obs[:, 15] > 0.9)).to(torch.float64)
    return rows


def env_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, device=device)


def behavior_clone(policy, env, obs, seeds, device, steps, epochs, minibatch,
                   action_schema, repeat, perturb_rate=0.0):
    """Warm-start engagement from a frozen teacher, then return current obs."""
    if steps <= 0 or epochs <= 0:
        return obs, {"bc_loss": None, "bc_accuracy": None}
    examples, targets = [], []
    for step in range(steps):
        labels = [scripted_action_indices(obs[:, role], action_schema=action_schema)
                  for role in range(2)]
        rows = torch.stack([scripted_baseline(obs[:, role], device,
                                              action_schema=action_schema)
                            for role in range(2)], dim=1)
        # DAgger-style state coverage: the old teacher only visited already
        # aligned pursuit states, so 99% label accuracy still yielded a policy
        # that had never learned recovery when its own sampled action put the
        # opponent behind. Execute controlled random disturbances while retaining
        # the teacher label for every resulting next-state example.
        if perturb_rate > 0.0:
            perturb = torch.rand((env.n, 2), device=device) < perturb_rate
            random_rows = torch.zeros_like(rows)
            move_values = torch.tensor(FWD, dtype=torch.float64, device=device)
            random_rows[:, :, 0] = move_values[
                torch.randint(0, 3, (env.n, 2), device=device)]
            if is_continuous_schema(action_schema):
                random_rows[:, :, 2] = (
                    torch.rand((env.n, 2), dtype=torch.float64, device=device) * 40.0 - 20.0)
                if has_continuous_pitch(action_schema):
                    random_rows[:, :, 3] = (
                        torch.rand((env.n, 2), dtype=torch.float64, device=device)
                        * 20.0 - 10.0)
            elif action_schema == FINE_ACTION_SCHEMA:
                yaw_values = torch.arange(-20.0, 20.1, 5.0,
                                          dtype=torch.float64, device=device)
                random_rows[:, :, 2] = yaw_values[
                    torch.randint(0, len(yaw_values), (env.n, 2), device=device)]
            else:
                yaw_values = torch.tensor(YAW, dtype=torch.float64, device=device)
                random_rows[:, :, 2] = yaw_values[
                    torch.randint(0, len(yaw_values), (env.n, 2), device=device)]
            random_rows[:, :, 5] = (random_rows[:, :, 0] > 0).to(torch.float64)
            rows = torch.where(perturb[:, :, None], random_rows, rows)
        if is_iron_gear_schema(action_schema):
            # Symmetric teachers lower both shields to swing on the same tick,
            # producing no axe-vs-shield examples. Alternate a persistent blocker
            # role in 16-tick windows so both egocentric roles observe raised
            # shields, disable them with axes, then learn the sword follow-up.
            forced_role = (step // 16) % 3 - 1
            if forced_role >= 0:
                rows[:, forced_role, 6] = 0.0
                rows[:, forced_role, 7] = 0.0
                rows[:, forced_role, 8] = 1.0
        examples.append(obs[..., :policy.obs_dim].reshape(-1, policy.obs_dim).clone())
        targets.append(torch.stack(labels, dim=1).reshape(-1, N_ACT))
        next_obs, _, done, _, _ = env.step(rows, repeat=repeat)
        next_obs = env_tensor(next_obs, device)
        done_t = env_tensor(done, device).bool()
        if done_t.any():
            mask = done_t.cpu().numpy().astype(np.uint8)
            seeds[mask.astype(bool)] += np.uint64(env.n * 100 + step + 1)
            env.reset(seeds, mask)
            next_obs = env_tensor(env.obs, device)
        obs = next_obs
    x = torch.cat(examples)
    y = torch.cat(targets)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    # Attack and large turn corrections are rare in a teacher trajectory. Plain
    # aggregate CE reached high accuracy by predicting no-attack/straight, which
    # reproduced the real deployment failure. Balance every categorical head.
    if is_continuous_schema(action_schema):
        categorical_sizes, categorical_cols = categorical_contract(action_schema)
    else:
        categorical_sizes, categorical_cols = HEADS, tuple(range(len(HEADS)))
    class_weights = []
    for col, size in zip(categorical_cols, categorical_sizes):
        counts = torch.bincount(y[:, col].long(), minlength=size).float().clamp_min(1.0)
        weights = (counts.sum() / (float(size) * counts)).clamp_max(10.0)
        class_weights.append(weights)
    loss_total = 0.0
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], minibatch):
            ix = order[start:start + minibatch]
            actor, _ = policy(x[ix])
            if is_continuous_schema(action_schema):
                losses = [F.cross_entropy(actor["categorical"][h],
                                          y[ix, col].long(), weight=class_weights[h])
                          for h, col in enumerate(categorical_cols)]
                yaw_target = torch.atanh(
                    (y[ix, 2] / YAW_LIMIT).clamp(-0.995, 0.995))
                losses.append(F.smooth_l1_loss(actor["yaw_mean"], yaw_target))
                if has_continuous_pitch(action_schema):
                    pitch_target = torch.atanh(
                        (y[ix, 3] / PITCH_LIMIT).clamp(-0.995, 0.995))
                    losses.append(F.smooth_l1_loss(
                        actor["pitch_mean"], pitch_target))
            else:
                losses = [F.cross_entropy(actor[h], y[ix, h].long(),
                                          weight=class_weights[h])
                          for h in range(len(HEADS))]
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.detach())
            updates += 1
    # Report final-model accuracy, not the misleading average of stale predictions
    # made throughout optimization. Per-head values expose collapse on rare but
    # essential turn and attack decisions that aggregate accuracy can conceal.
    correct = torch.zeros(len(categorical_cols), dtype=torch.float64, device=device)
    yaw_abs_error = torch.tensor(0.0, dtype=torch.float64, device=device)
    pitch_abs_error = torch.tensor(0.0, dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, x.shape[0], minibatch):
            actor, _ = policy(x[start:start + minibatch])
            yy = y[start:start + minibatch]
            logits = (actor["categorical"] if is_continuous_schema(action_schema)
                      else actor)
            for h, col in enumerate(categorical_cols):
                correct[h] += (logits[h].argmax(-1) == yy[:, col].long()).sum()
            if is_continuous_schema(action_schema):
                yaw_abs_error += (YAW_LIMIT * torch.tanh(actor["yaw_mean"])
                                  - yy[:, 2]).abs().sum().double()
                if has_continuous_pitch(action_schema):
                    pitch_abs_error += (PITCH_LIMIT * torch.tanh(actor["pitch_mean"])
                                        - yy[:, 3]).abs().sum().double()
    head_accuracy = (correct / x.shape[0]).cpu().tolist()
    metrics = {"bc_loss": loss_total / max(1, updates),
               "bc_accuracy": float(np.mean(head_accuracy))}
    if is_continuous_schema(action_schema):
        names = (("forward", "strafe", "jump", "sprint", "weapon", "combat")
                 if is_iron_gear_schema(action_schema) else
                 ("forward", "strafe", "jump", "sprint", "attack"))
        metrics["bc_yaw_mae_deg"] = float(yaw_abs_error / x.shape[0])
        if has_continuous_pitch(action_schema):
            metrics["bc_pitch_mae_deg"] = float(pitch_abs_error / x.shape[0])
    elif action_schema == FINE_ACTION_SCHEMA:
        names = ("forward", "strafe", "yaw_coarse", "yaw_fine", "jump", "sprint", "attack")
    else:
        names = ("forward", "strafe", "yaw", "pitch", "jump", "sprint", "attack")
    metrics.update({f"bc_accuracy_{name}": head_accuracy[h]
                    for h, name in enumerate(names)})
    return obs, metrics


def save_checkpoint(path, policies, optimizers, chunk, config):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"models": [p.state_dict() for p in policies],
                "optimizers": [o.state_dict() for o in optimizers],
                "chunk": chunk, "config": config}, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(policy, device, episodes=256, horizon=300, repeat=4,
             opponent="chaser", stochastic=False,
             action_schema=LEGACY_ACTION_SCHEMA):
    """Evaluate exactly one episode per lane against a frozen opponent."""
    if opponent not in ("stationary", "random", "chaser"):
        raise ValueError(f"unknown opponent {opponent}")
    n = episodes
    env = (NetworkVecPvp(n, device=device.index or 0)
           if action_schema == V21_ACTION_SCHEMA else
           VecPvp(n, device=device.index or 0))
    seeds = np.arange(900000, 900000 + n, dtype=np.uint64)
    obs = env_tensor(env.reset(seeds), device)
    wins = losses = draws = 0
    hits = np.zeros(2, dtype=np.int64)
    damage = np.zeros(2, dtype=np.float64)
    active = torch.ones(n, dtype=torch.bool, device=device)
    pursuit_steps = bearing_sum = bearing_aligned = behind_steps = 0
    forward_behind = reach_steps = wall_steps = yaw_saturated = 0
    distance_sum = 0.0
    first_hit = torch.full((n,), -1, dtype=torch.int64, device=device)
    for step in range(horizon):
        actor, _ = policy(policy_input(policy, obs[:, 0]))
        if stochastic:
            a0, _, _ = sample_actions(actor, action_schema)
        else:
            a0 = greedy_actions(actor, action_schema)
        lateral, longitudinal = obs[:, 0, 5], obs[:, 0, 6]
        bearing = torch.atan2(lateral.abs(), longitudinal) * (180.0 / np.pi)
        distance = obs[:, 0, 10] * 32.0
        behind = longitudinal < 0.0
        forward = a0[:, 0] == 2
        if is_continuous_schema(action_schema):
            yaw_delta = a0[:, 2]
            saturated = yaw_delta.abs() >= (YAW_LIMIT - 0.1)
        elif action_schema == FINE_ACTION_SCHEMA:
            yaw_delta = (a0[:, 2] - 1) * 15 + (a0[:, 3] - 1) * 5
            saturated = yaw_delta.abs() == 20
        else:
            saturated = (a0[:, 2] == 0) | (a0[:, 2] == 2)
        pursuit_steps += int(active.sum())
        bearing_sum += float(bearing[active].sum())
        bearing_aligned += int(((bearing < 15.0) & active).sum())
        behind_steps += int((behind & active).sum())
        forward_behind += int((forward & behind & active).sum())
        reach_steps += int(((distance < 3.0) & active).sum())
        wall_steps += int(((obs[:, 0, 20:24].amin(-1) < 0.05) & active).sum())
        yaw_saturated += int((saturated & active).sum())
        distance_sum += float(distance[active].sum())
        rows = torch.zeros((n, 2, N_ACT), dtype=torch.float64, device=device)
        rows[:, 0] = decode_actions(a0, device, action_schema)
        if opponent == "chaser":
            rows[:, 1] = scripted_baseline(obs[:, 1], device, attack=True,
                                           action_schema=action_schema)
        elif opponent == "random":
            if is_continuous_schema(action_schema):
                random_actions = torch.zeros((n, N_ACT), device=device)
                sizes, cols = categorical_contract(action_schema)
                for size, col in zip(sizes, cols):
                    random_actions[:, col] = torch.randint(size, (n,), device=device)
                random_actions[:, 2] = torch.rand(n, device=device) * 40.0 - 20.0
                if has_continuous_pitch(action_schema):
                    random_actions[:, 3] = torch.rand(n, device=device) * 20.0 - 10.0
            else:
                random_actions = torch.stack(
                    [torch.randint(size, (n,), device=device) for size in HEADS], dim=-1)
            rows[:, 1] = decode_actions(random_actions, device, action_schema)
        obs, reward, done, hs, dmg = env.step(rows, repeat=repeat)
        obs = env_tensor(obs, device)
        hits += env_tensor(hs, device)[active].sum(0).cpu().numpy()
        damage += env_tensor(dmg, device)[active].sum(0).cpu().numpy()
        landed = (env_tensor(hs, device)[:, 0] > 0) & active & (first_hit < 0)
        first_hit[landed] = step + 1
        newly_done = env_tensor(done, device).bool() & active
        if newly_done.any():
            rr = env_tensor(reward, device)[newly_done]
            wins += int((rr[:, 0] > rr[:, 1]).sum())
            losses += int((rr[:, 0] < rr[:, 1]).sum())
            draws += int((rr[:, 0] == rr[:, 1]).sum())
            active[newly_done] = False
        if not active.any():
            break
    # A lane still active after the evaluation horizon is an explicit draw.
    draws += int(active.sum())
    env.close()
    prefix = f"eval_{opponent}_{'sampled' if stochastic else 'greedy'}"
    pursuit_steps = max(1, pursuit_steps)
    first_hit_seen = first_hit[first_hit >= 0]
    return {f"{prefix}_wins": wins, f"{prefix}_losses": losses,
            f"{prefix}_draws": draws,
            f"{prefix}_win_rate": wins / max(1, wins + losses + draws),
            f"{prefix}_hits_role0": int(hits[0]),
            f"{prefix}_hits_opponent": int(hits[1]),
            f"{prefix}_damage_role0": float(damage[0]),
            f"{prefix}_damage_opponent": float(damage[1]),
            f"{prefix}_mean_abs_bearing_error_deg": bearing_sum / pursuit_steps,
            f"{prefix}_bearing_under_15_fraction": bearing_aligned / pursuit_steps,
            f"{prefix}_behind_fraction": behind_steps / pursuit_steps,
            f"{prefix}_forward_while_behind_fraction":
                forward_behind / max(1, behind_steps),
            f"{prefix}_forward_while_behind_player_fraction":
                forward_behind / pursuit_steps,
            f"{prefix}_inside_reach_fraction": reach_steps / pursuit_steps,
            f"{prefix}_near_wall_fraction": wall_steps / pursuit_steps,
            f"{prefix}_yaw_saturated_fraction": yaw_saturated / pursuit_steps,
            f"{prefix}_mean_distance_blocks": distance_sum / pursuit_steps,
            f"{prefix}_mean_decisions_to_first_hit":
                (float(first_hit_seen.float().mean()) if len(first_hit_seen) else None),
            f"{prefix}_mean_minecraft_ticks_to_first_hit":
                (repeat * float(first_hit_seen.float().mean())
                 if len(first_hit_seen) else None)}


def main():
    action_schema = os.environ.get("PVP_ACTION_SCHEMA", FINE_ACTION_SCHEMA)
    if action_schema not in (LEGACY_ACTION_SCHEMA, FINE_ACTION_SCHEMA,
                             CONTINUOUS_ACTION_SCHEMA, CONTINUOUS_LOOK_ACTION_SCHEMA,
                             V2_ACTION_SCHEMA, V21_ACTION_SCHEMA):
        raise ValueError(f"unsupported PVP_ACTION_SCHEMA {action_schema}")
    default_repeat = ("1" if action_schema in
                      (FINE_ACTION_SCHEMA, CONTINUOUS_ACTION_SCHEMA,
                       CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                       V21_ACTION_SCHEMA) else "4")
    cfg = {
        "n": int(os.environ.get("PVP_N", "4096")),
        "rollout": int(os.environ.get("PVP_ROLLOUT", "64")),
        "chunks": int(os.environ.get("PVP_CHUNKS", "100")),
        "repeat": int(os.environ.get("PVP_REPEAT", default_repeat)),
        "epochs": int(os.environ.get("PVP_EPOCHS", "2")),
        "minibatch": int(os.environ.get("PVP_MINIBATCH", "65536")),
        "lr": float(os.environ.get("PVP_LR", "3e-4")),
        "gamma": float(os.environ.get("PVP_GAMMA", "0.995")),
        "gae": float(os.environ.get("PVP_GAE", "0.95")),
        "clip": float(os.environ.get("PVP_CLIP", "0.2")),
        "target_kl": float(os.environ.get("PVP_TARGET_KL", "0.01")),
        "entropy": float(os.environ.get("PVP_ENTROPY", "0.01")),
        "bootstrap_static": int(os.environ.get("PVP_BOOTSTRAP_STATIC", "10")),
        "bootstrap_chaser": int(os.environ.get("PVP_BOOTSTRAP_CHASER", "20")),
        "approach_reward": float(os.environ.get("PVP_APPROACH_REWARD", "0.1")),
        # Opt-in deployment-domain randomization, calibrated from the measured
        # real-client timing report rather than broad arbitrary perturbations.
        "action_hold_prob": float(os.environ.get("PVP_ACTION_HOLD_PROB", "0")),
        "extra_repeat_prob": float(os.environ.get("PVP_EXTRA_REPEAT_PROB", "0")),
        "bc_steps": int(os.environ.get("PVP_BC_STEPS", "64")),
        "bc_epochs": int(os.environ.get("PVP_BC_EPOCHS", "4")),
        "bc_perturb": float(os.environ.get(
            "PVP_BC_PERTURB", "0.25" if action_schema in
            (FINE_ACTION_SCHEMA, CONTINUOUS_ACTION_SCHEMA,
             CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
             V21_ACTION_SCHEMA) else "0")),
        "network_domain": action_schema == V21_ACTION_SCHEMA,
        "ping_min_ms": float(os.environ.get("PVP_PING_MIN_MS", "20")),
        "ping_max_ms": float(os.environ.get("PVP_PING_MAX_MS", "200")),
        "ping_variation_fraction": 0.05 if action_schema == V21_ACTION_SCHEMA else 0.0,
        "network_history_frames": 4 if action_schema == V21_ACTION_SCHEMA else 1,
        "checkpoint_contract": ("netherite_pvp_latency_actor_critic_v6"
                                if action_schema == V21_ACTION_SCHEMA else
                                "netherite_pvp_iron_gear_actor_critic_v5"
                                if action_schema == V2_ACTION_SCHEMA else
                                "netherite_pvp_mixed_look_actor_critic_v4"
                                if action_schema == CONTINUOUS_LOOK_ACTION_SCHEMA else
                                "netherite_pvp_mixed_actor_critic_v3"
                                if action_schema == CONTINUOUS_ACTION_SCHEMA else
                                "netherite_pvp_actor_critic_v2"),
        "observation_schema": ("four_frame_iron_gear_plus_own_ping_141_v5"
                               if action_schema == V21_ACTION_SCHEMA else
                               "egocentric_iron_gear_state_35_v4"
                               if action_schema == V2_ACTION_SCHEMA else
                               "egocentric_state_25_v3"
                               if action_schema == CONTINUOUS_LOOK_ACTION_SCHEMA
                               else "egocentric_state_24_v2"),
        "obs_basis": "movement_v2",
        "action_schema": action_schema,
        "control_hz": 20 // int(os.environ.get("PVP_REPEAT", default_repeat)),
        "eval_horizon": int(os.environ.get(
            "PVP_EVAL_HORIZON",
            str(1200 // int(os.environ.get("PVP_REPEAT", default_repeat))))),
        "seed": int(os.environ.get("PVP_SEED", "12345")),
        "init_checkpoint": os.environ.get(
            "PVP_INIT_CHECKPOINT",
            str(HERE.parent.parent / "artifacts/pilots/pilot21/selfplay.pt")
            if action_schema == V21_ACTION_SCHEMA else
            str(HERE.parent.parent / "artifacts/pilots/pilot17/selfplay.pt")
            if action_schema == V2_ACTION_SCHEMA else ""),
    }
    out = pathlib.Path(os.environ.get("PVP_OUT", str(HERE / "out")))
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    checkpoint = out / "selfplay.pt"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and
                          (HERE / "pvp_cuda.so").exists() else "cpu")
    # Independent policies avoid the exact gradient cancellation of one shared
    # policy receiving both sides of a symmetric zero-sum encounter.
    policies = [Policy(action_schema=action_schema).to(device),
                Policy(action_schema=action_schema).to(device)]
    if is_iron_gear_schema(action_schema) and cfg["init_checkpoint"]:
        init = torch.load(cfg["init_checkpoint"], map_location=device,
                          weights_only=False)
        for role in range(2):
            if action_schema == V21_ACTION_SCHEMA:
                transfer_v2_latency_policy(policies[role], init["models"][role])
            else:
                transfer_v1_look_policy(policies[role], init["models"][role])
    optimizers = [torch.optim.Adam(p.parameters(), lr=cfg["lr"], eps=1e-5)
                  for p in policies]
    env = (NetworkVecPvp(cfg["n"], device=device.index or 0,
                         min_ping_ms=cfg["ping_min_ms"],
                         max_ping_ms=cfg["ping_max_ms"])
           if cfg["network_domain"] else
           VecPvp(cfg["n"], device=device.index or 0))
    seeds = np.arange(cfg["seed"], cfg["seed"] + cfg["n"], dtype=np.uint64)
    obs = env_tensor(env.reset(seeds), device)
    obs, bc_metrics = behavior_clone(
        policies[0], env, obs, seeds, device, cfg["bc_steps"], cfg["bc_epochs"],
        cfg["minibatch"], cfg["action_schema"], cfg["repeat"], cfg["bc_perturb"])
    policies[1].load_state_dict(policies[0].state_dict())
    optimizers = [torch.optim.Adam(p.parameters(), lr=cfg["lr"], eps=1e-5)
                  for p in policies]
    # Preserve and score the behavioral-cloning initialization independently.
    # A later PPO regression must not erase the only known engaging controller.
    torch.save({"models": [p.state_dict() for p in policies], "chunk": -1,
                "config": cfg, "behavior_clone": bc_metrics},
               out / "behavior_clone.pt")
    eval_n = int(os.environ.get("PVP_EVAL_N", "128"))
    bc_row = {"chunk": -1, "phase": "behavior_clone", **bc_metrics}
    bc_row.update(evaluate(policies[0], device, episodes=eval_n,
                           opponent="stationary", stochastic=False,
                           repeat=cfg["repeat"], action_schema=cfg["action_schema"],
                           horizon=cfg["eval_horizon"]))
    bc_row.update(evaluate(policies[0], device, episodes=eval_n,
                           opponent="chaser", stochastic=False,
                           repeat=cfg["repeat"], action_schema=cfg["action_schema"],
                           horizon=cfg["eval_horizon"]))
    with metrics_path.open("a") as f:
        f.write(json.dumps(bc_row, sort_keys=True) + "\n")
    global_decisions = 0
    previous_rows = torch.zeros((cfg["n"], 2, N_ACT),
                                dtype=torch.float64, device=device)

    print(json.dumps({"device": str(device), **cfg, **bc_metrics}, sort_keys=True),
          flush=True)
    for chunk in range(cfg["chunks"]):
        if chunk == cfg["bootstrap_chaser"]:
            # Begin adversarial training from two copies of the policy that
            # learned to engage, then let them diverge independently.
            policies[1].load_state_dict(policies[0].state_dict())
            optimizers[1] = torch.optim.Adam(
                policies[1].parameters(), lr=cfg["lr"], eps=1e-5)
        t0 = time.time()
        ob, ac, lp, va, rw, dn = [], [], [], [], [], []
        chunk_hits = torch.zeros(2, dtype=torch.int64, device=device)
        chunk_damage = torch.zeros(2, device=device)
        kills = torch.zeros(2, dtype=torch.int64, device=device)
        draws = 0
        gear_counts = {name: torch.zeros(2, dtype=torch.int64, device=device)
                       for name in ("axe", "attack", "block_intent",
                                    "blocking", "switch", "shield_disable")}
        mutual_block_ticks = 0
        chunk_held_actions = 0
        chunk_extra_repeats = 0
        for _ in range(cfg["rollout"]):
            with torch.no_grad():
                role_out = []
                for role in range(2):
                    actor, value = policies[role](
                        policy_input(policies[role], obs[:, role]))
                    action, logp, _ = sample_actions(actor, cfg["action_schema"])
                    role_out.append((action, logp, value))
                action = torch.stack([x[0] for x in role_out], dim=1)
                logp = torch.stack([x[1] for x in role_out], dim=1)
                value = torch.stack([x[2] for x in role_out], dim=1)
            rows = decode_actions(action, device, cfg["action_schema"])
            if chunk < cfg["bootstrap_static"]:
                rows[:, 1].zero_()
            elif chunk < cfg["bootstrap_chaser"]:
                rows[:, 1] = scripted_baseline(
                    obs[:, 1], device, attack=True,
                    action_schema=cfg["action_schema"])
            # VecPvp reuses its observation output buffer. Preserve the state
            # that produced this action *before* env.step overwrites that buffer;
            # pairing actions/log-probs with the next state creates a fake PPO KL.
            decision_obs = obs.clone()
            executed_rows = rows
            if cfg["action_hold_prob"] > 0.0:
                hold = (torch.rand((cfg["n"], 2), device=device) <
                        cfg["action_hold_prob"])
                executed_rows = torch.where(hold[:, :, None], previous_rows, rows)
                chunk_held_actions += int(hold.sum())
            # The newly selected action is what persists if the following real
            # client command misses its tick window.
            previous_rows = rows.clone()
            step_repeat = cfg["repeat"]
            if (cfg["extra_repeat_prob"] > 0.0 and
                    float(torch.rand((), device=device)) < cfg["extra_repeat_prob"]):
                step_repeat += 1
                chunk_extra_repeats += 1
            next_obs, reward, done, hits, damage = env.step(
                executed_rows, repeat=step_repeat)
            next_obs = env_tensor(next_obs, device)
            if is_iron_gear_schema(cfg["action_schema"]):
                gear_counts["axe"] += (rows[:, :, 7] > 0.5).sum(0)
                gear_counts["attack"] += (rows[:, :, 6] > 0.5).sum(0)
                gear_counts["block_intent"] += (rows[:, :, 8] > 0.5).sum(0)
                gear_counts["blocking"] += (decision_obs[:, :, 27] > 0.5).sum(0)
                gear_counts["switch"] += (
                    rows[:, :, 7].long() != decision_obs[:, :, 25].long()).sum(0)
                gear_counts["shield_disable"] += (
                    (next_obs[:, :, 29] > 0.5) &
                    (decision_obs[:, :, 29] <= 0.5)).sum(0)
                mutual_block_ticks += int(((decision_obs[:, 0, 27] > 0.5) &
                                           (decision_obs[:, 1, 27] > 0.5)).sum())
            reward = env_tensor(reward, device)
            train_reward = reward.clone()
            if chunk < cfg["bootstrap_chaser"]:
                # Potential-based approach shaping solves the exploration problem
                # without paying the agent merely for standing near its target.
                # Observation distance is normalized by the 32-block arena.
                progress_blocks = (obs[:, 0, 10] - next_obs[:, 0, 10]) * 32.0
                train_reward[:, 0] += cfg["approach_reward"] * progress_blocks
            done_t = env_tensor(done, device).bool()
            hits_t = env_tensor(hits, device)
            damage_t = env_tensor(damage, device)
            # Reward/done buffers must likewise be copied before the next step.
            ob.append(decision_obs); ac.append(action); lp.append(logp); va.append(value)
            rw.append(train_reward)
            dn.append(done_t[:, None].expand(-1, 2).clone())
            chunk_hits += hits_t.sum(0)
            chunk_damage += damage_t.sum(0)
            if done_t.any():
                terminal = reward[done_t]
                kills[0] += (terminal[:, 0] > terminal[:, 1]).sum()
                kills[1] += (terminal[:, 1] > terminal[:, 0]).sum()
                draws += int((terminal[:, 0] == terminal[:, 1]).sum())
                mask = done_t.cpu().numpy().astype(np.uint8)
                seeds[mask.astype(bool)] += np.uint64(cfg["n"] * 1000 + chunk + 1)
                env.reset(seeds, mask)
                previous_rows[done_t] = 0.0
                next_obs = env_tensor(env.obs, device)
            obs = next_obs
            global_decisions += cfg["n"] * 2

        with torch.no_grad():
            next_value = torch.stack(
                [policies[r](policy_input(policies[r], obs[:, r]))[1]
                 for r in range(2)], dim=1)
        rewards = torch.stack(rw)
        dones = torch.stack(dn)
        values = torch.stack(va)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        nv = next_value
        for t in reversed(range(cfg["rollout"])):
            live = (~dones[t]).float()
            delta = rewards[t] + cfg["gamma"] * nv * live - values[t]
            gae = delta + cfg["gamma"] * cfg["gae"] * live * gae
            advantages[t] = gae
            nv = values[t]
        returns = advantages + values
        bobs = torch.cat(ob)       # [T*N, 2, obs]
        bact = torch.cat(ac)       # [T*N, 2, heads]
        blogp = torch.cat(lp)      # [T*N, 2]
        badv = advantages.reshape(-1, 2)
        bret = returns.reshape(-1, 2)

        losses = {}
        total = bobs.shape[0]
        train_roles = (0,) if chunk < cfg["bootstrap_chaser"] else (0, 1)
        with torch.no_grad():
            for role in train_roles:
                replay_actor, _ = policies[role](
                    policy_input(policies[role], bobs[:, role]))
                replay_logp, _ = evaluate_actions(
                    replay_actor, bact[:, role], cfg["action_schema"])
                losses[f"rollout_logp_replay_max_role{role}"] = float(
                    (replay_logp - blogp[:, role]).abs().max())
        for role in train_roles:
            adv = badv[:, role]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            accum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                     "approx_kl": 0.0, "clip_frac": 0.0}
            updates = 0
            early_stop = False
            for _ in range(cfg["epochs"]):
                order = torch.randperm(total, device=device)
                for start in range(0, total, cfg["minibatch"]):
                    ix = order[start:start + cfg["minibatch"]]
                    actor, value_now = policies[role](
                        policy_input(policies[role], bobs[ix, role]))
                    newlogp, entropy = evaluate_actions(
                        actor, bact[ix, role], cfg["action_schema"])
                    ratio = (newlogp - blogp[ix, role]).exp()
                    pg1 = ratio * adv[ix]
                    pg2 = ratio.clamp(1.0 - cfg["clip"],
                                      1.0 + cfg["clip"]) * adv[ix]
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = 0.5 * (value_now - bret[ix, role]).square().mean()
                    ent = entropy.mean()
                    loss = policy_loss + 0.5 * value_loss - cfg["entropy"] * ent
                    optimizers[role].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policies[role].parameters(), 0.5)
                    optimizers[role].step()
                    with torch.no_grad():
                        # Recompute after the optimizer step. Besides being the
                        # trust-region quantity we actually care about, this avoids
                        # retaining an autograd output across backward/step and then
                        # mistaking that stale tensor for the updated policy.
                        post_actor, _ = policies[role](
                            policy_input(policies[role], bobs[ix, role]))
                        post_logp, _ = evaluate_actions(
                            post_actor, bact[ix, role], cfg["action_schema"])
                        logratio = post_logp - blogp[ix, role]
                        kl = ((logratio.exp() - 1.0) - logratio).mean()
                        post_ratio = logratio.exp()
                        cf = ((post_ratio - 1.0).abs() > cfg["clip"]).float().mean()
                    for k, v in (("policy_loss", policy_loss),
                                 ("value_loss", value_loss), ("entropy", ent),
                                 ("approx_kl", kl), ("clip_frac", cf)):
                        accum[k] += float(v.detach())
                    updates += 1
                    if cfg["target_kl"] > 0.0 and float(kl) > cfg["target_kl"]:
                        early_stop = True
                        break
                if early_stop:
                    break
            for k in accum:
                losses[f"{k}_role{role}"] = accum[k] / max(1, updates)
            losses[f"ppo_updates_role{role}"] = updates
            losses[f"kl_early_stop_role{role}"] = early_stop
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            if len(train_roles) == 1:
                losses[f"{k}_role1"] = None
                losses[k] = losses[f"{k}_role0"]
            else:
                losses[k] = 0.5 * (losses[f"{k}_role0"] +
                                   losses[f"{k}_role1"])

        elapsed = time.time() - t0
        phase = ("static_bootstrap" if chunk < cfg["bootstrap_static"] else
                 "chaser_bootstrap" if chunk < cfg["bootstrap_chaser"] else
                 "adversarial")
        row = {"chunk": chunk, "phase": phase, **bc_metrics,
               "global_agent_decisions": global_decisions,
               "wall_seconds": elapsed,
               "agent_decisions_per_second": cfg["n"] * 2 * cfg["rollout"] / elapsed,
               "env_ticks_per_second": cfg["n"] * cfg["rollout"] * cfg["repeat"] / elapsed,
               "reward_mean": float(rewards.mean()),
               "reward_role0": float(rewards[:, :, 0].mean()),
               "reward_role1": float(rewards[:, :, 1].mean()),
               "hits_role0": int(chunk_hits[0]), "hits_role1": int(chunk_hits[1]),
               "hits_per_second": float(chunk_hits.sum()) / elapsed,
               "damage_role0": float(chunk_damage[0]),
               "damage_role1": float(chunk_damage[1]),
               "damage_per_second": float(chunk_damage.sum()) / elapsed,
               "kills_role0": int(kills[0]), "kills_role1": int(kills[1]),
               "draws": draws, **losses}
        row["action_hold_fraction"] = (
            chunk_held_actions / max(1, cfg["n"] * 2 * cfg["rollout"]))
        row["extra_repeat_fraction"] = chunk_extra_repeats / max(1, cfg["rollout"])
        if is_iron_gear_schema(cfg["action_schema"]):
            denom = max(1, cfg["n"] * cfg["rollout"])
            for name, counts in gear_counts.items():
                row[f"{name}_role0"] = int(counts[0])
                row[f"{name}_role1"] = int(counts[1])
                row[f"{name}_fraction"] = float(counts.sum()) / (2 * denom)
            row["mutual_block_fraction"] = mutual_block_ticks / denom
        if chunk % 5 == 0 or chunk == cfg["chunks"] - 1:
            row.update(evaluate(policies[0], device, episodes=eval_n,
                                opponent="stationary", stochastic=False,
                                repeat=cfg["repeat"],
                                action_schema=cfg["action_schema"],
                                horizon=cfg["eval_horizon"]))
            row.update(evaluate(policies[0], device, episodes=eval_n,
                                opponent="chaser", stochastic=False,
                                repeat=cfg["repeat"],
                                action_schema=cfg["action_schema"],
                                horizon=cfg["eval_horizon"]))
            row.update(evaluate(policies[0], device, episodes=eval_n,
                                opponent="chaser", stochastic=True,
                                repeat=cfg["repeat"],
                                action_schema=cfg["action_schema"],
                                horizon=cfg["eval_horizon"]))
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        save_checkpoint(checkpoint, policies, optimizers, chunk, cfg)
        print(json.dumps(row, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    main()
