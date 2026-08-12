# Training ↔ deployment parity report

Date: 2026-08-12  
Checkpoint used for timing: Pilot 20 (`iron_gear_20hz_v5`)  
Minecraft oracle/deployment: Forge 1.11.2-13.20.1.2588

## Contract and gates

A deployment decision is one policy observation followed by exactly one action
application and the next client-tick observation. The receipt now records, for
both clients, host-monotonic send/receive times, client tick, world tick, action
application tick, and policy action sequence.

Targets:

- mean cadence near 20 Hz and p95 interval <= 60 ms;
- no duplicate/reordered client ticks;
- >= 99% one-client-tick transitions;
- both roles apply within one world tick >= 99% of the time;
- deterministic policy-observation tapes remain close until a documented combat
  approximation is exercised.

## Measured baseline

400 real-Minecraft decisions, no video capture:

| Metric | Original 40 ms barrier / fresh TCP | Persistent bridge + 48 ms barrier |
|---|---:|---:|
| Effective rate | 19.40 Hz | 19.59 Hz |
| Mean interval | 51.62 ms | 51.05 ms |
| p95 interval | 61.49 ms | 59.15 ms |
| p99 interval | 102.05 ms | 86.23 ms |
| Exact one-tick transitions, role 0 | 97.24% | 98.50% |
| Exact one-tick transitions, role 1 | 98.25% | 99.00% |
| Same world-tick application | 56.25% | 63.75% |
| Within one world tick | 97.25% | 99.50% |
| Maximum phase skew | 17 ticks (startup/outlier) | 2 ticks |

Artifacts are under `artifacts/parity/` locally. Timing receipts are intentionally
not source-controlled.

Conclusion: deployment is normally one transition per 50 ms tick, but it is not
an exact synchronous two-actor clock. Roughly 1-1.5% of transitions span an
extra client tick and 36% of action pairs land on adjacent, rather than identical,
world ticks. Policy inference is not the limiting operation; the two independently
phased Minecraft client loops are.

## Deterministic action tapes

`java/pvp_parity_tape.py` resets Blaze and Minecraft to an exact face-off and
compares the complete 35-value policy observation after each action. A 20-tick
settle window is required because player teleport/equipment packets are
asynchronous in real Minecraft.

After settling and fixing the cooldown mismatch:

| Tape | Mean absolute observation error | p95 absolute error |
|---|---:|---:|
| Idle | ~0 | ~0 |
| Straight approach | 0.00013 | 0.00015 |
| Turning approach | 0.0010 | 0.0050 |
| Gear/shield, outside combat | 0.00023 | 0.00017 |
| Gear/shield with contact combat | 0.0242 | 0.1200 |

Movement/look transfer is therefore extremely close. Contact combat has expected
residual divergence in knockback, vertical/on-ground state, hurt timing, and
post-armor health.

## Avoidable mismatch found and fixed

The simulator reset attack cooldown for every attack *intent*, including attacks
that raycast only air. Vanilla resets attack strength only when
`attackTargetEntityWithCurrentItem` runs on an entity. Real deployment therefore
showed full cooldown during pursuit while training showed near-zero cooldown.

`pvp_arena.h` now resets the timer only for an in-reach entity target. After the
fix, the gear tape cooldown MAE fell from **0.548** to **0.0055** outside contact.
Pilot 20 still completes all 256 evaluation fights under the corrected rule, so
this fix does not destroy the learned behavior, but the next canonical checkpoint
must be trained with the corrected semantics.

## Residual robustness recipe

Training/evaluation now has opt-in, measured deployment-domain perturbations:

- `PVP_ACTION_HOLD_PROB`: independently retain a role's prior action for one
  transition, approximating adjacent-tick client phase skew;
- `PVP_EXTRA_REPEAT_PROB`: occasionally span one extra environment tick,
  approximating measured skipped client windows.

The initial calibrated recipe is `PVP_ACTION_HOLD_PROB=0.10` and
`PVP_EXTRA_REPEAT_PROB=0.02`. This is deliberately narrower than the worst-case
trace and avoids hiding implementation defects behind broad randomization.

## Corrected robust candidate: Pilot 21

Pilot 21 was retrained from the same V1.2 look-policy initialization using the
corrected cooldown rule and the calibrated 0.10 action-hold / 0.02 extra-tick
recipe. Training plus periodic evaluation took about 6m25s on one L4 and produced
approximately 31.5 million agent decisions.

All 256 simulator fights complete in clean and perturbed native/swapped modes.
A stochastic real-Minecraft gate completed in 344 decisions with 11/17 hits and
a role-1 kill. Its measured cadence was **19.83 Hz**, p95 interval **52.41 ms**,
exact one-tick transitions **99.71% / 99.42%**, same-world-tick application
**97.67%**, and within-one-tick application **99.71%**. This run meets the cadence,
p95, and within-one-tick gates; one role remains just below the strict 99% exact
one-tick target across the full fight.

## Residual limitations

- AI-vs-AI rendering clients are still independently phased; exact simultaneous
  server-epoch injection would require a larger server-authoritative movement
  scheduler.
- Contact combat is not bit-identical vanilla, especially knockback/on-ground
  ordering and sequential packet resolution.
- Human-vs-AI is intentionally asymmetric: native human input is continuous while
  the AI acts on policy ticks.
