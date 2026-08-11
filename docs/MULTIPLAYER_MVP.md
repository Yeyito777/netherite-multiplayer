# Multiplayer MVP contract

This fork extends Blaze from `N worlds x 1 player` to batched two-player,
shared-world matches. The first product is deliberately narrow: Minecraft
1.11.2 fist combat on a dry, flat 32x32 stone arena.

## Scope

- Two survival players, no armor, weapons, items, mobs, blocks edits, fluids,
  hunger, regeneration, or fall damage.
- Twenty health per player. Death, arena exit, or timeout terminates a match.
- Actions: forward, strafe, yaw delta, pitch delta, jump, sprint, and attack.
- Both players submit an intent before either player's combat result mutates
  shared state.
- The training interface is batched and GPU-resident. A match is one
  environment; the two agents share its state.
- The initial observation is privileged structured state. A matching Java mod
  observation is required before a policy-transfer claim.

## Non-goals for the MVP

- General Minecraft networking or protocol compatibility.
- More than two players per match.
- RGB-only policies.
- Inventory, armor, weapon, shield, projectile, block-place, or block-break
  combat.
- Public-server deployment.

## Deterministic tick transaction

1. Decode both actions and update look state.
2. Snapshot both attack intents.
3. Advance both players' movement against the same arena state.
4. Resolve player collision symmetrically.
5. Evaluate both attacks from the post-movement, pre-damage snapshot.
6. Apply accepted damage and knockback.
7. Advance cooldown and hurt timers.
8. Compute observations, zero-sum rewards, death, winner, and timeout.

No result may depend accidentally on array iteration order. If the Java 1.11.2
server has an authoritative ordering rule, it must be represented explicitly
and covered by a golden trace.

For the Java oracle, socket threads only enqueue `{episode,tick,role,action}`.
At `ServerTickEvent.START`, the server thread freezes a complete batch and
executes role 0 then role 1 through each player's `NetHandlerPlayServer`; at
`ServerTickEvent.END`, it records both authoritative player states and ordered
combat events. Players are bound to roles by configured UUID, never list order
or nearest-player selection. Missing, duplicate, late, and wrong-episode actions
are explicit outcomes. `verify/trace/pvp_scheduler.py` is the executable,
arrival-order-independent reference for this protocol and trace schema.

## Verification gates

1. Existing single-player Blaze gates do not regress beyond the pinned
   upstream baseline.
2. Scalar CPU and CUDA match state byte-for-byte on every tick.
3. Swapping players and mirroring/rotating the arena produces swapped,
   mirrored/rotated results.
4. Batch-lane permutation cannot change a match trajectory.
5. Canonical two-player traces match an instrumented Java 1.11.2 server for
   movement, accepted-hit ticks, health, hurt timers, and knockback.
6. Training success is measured against frozen scripted and historical
   opponents, not only the simultaneously learning opponent.

## Training acceptance

- Hits, deaths, and non-draw matches occur without scripted action injection.
- A trained checkpoint materially exceeds random and stationary opponents on
  held-out seeds.
- Performance does not collapse against a league of historical checkpoints.
- Metrics include reward, policy/value loss, entropy, KL, hit rate, accepted
  hits per second, damage, kills, deaths, draws, match duration, combo length,
  evaluation win rates, and simulator/trainer throughput.

## Deployment acceptance

Two modded clients connect to a private Minecraft 1.11.2 server. The same
frozen policy and preprocessing used in simulation run closed-loop through the
mod bridge. Server-authoritative traces preserve tick/action IDs and permit a
paired simulator-vs-Java evaluation.

Current gate status: CPU/CUDA parity, adversarial self-play, fine-control skill
gates, and real-client deployment are implemented. Two isolated Forge clients
join one private integrated server as stable `Player0`/`Player1` roles. The mod
builds the arena and executes movement, look, and vanilla raycast attacks from
the frozen Pilot 13 policies. The realtime path advances exactly one action
sequence per decision and measured 19.71 decisions/second while simultaneously
recording both clients. The canonical serialized oracle path remains available
for auditable server-authoritative traces.
