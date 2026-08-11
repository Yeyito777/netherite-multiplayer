# Pilot 13 real Minecraft 1.11.2 deployment

Two independently optimized pilot 13 policies were deployed into two real Forge
clients on deterministic asymmetric resets. Movement/look traveled through both
client bridges; melee used role-ordered authoritative `NetHandlerPlayServer`
injection. Death is taken from the lethal attack receipt because the headless
client auto-respawns before the following state read.

## Repeated match results

| Seed | Decisions | Hits role 0 / role 1 | Damage role 0 / role 1 | Winner |
|---:|---:|---:|---:|---:|
| 130013 | 106 | 19 / 18 | 20.0 / 18.5 | role 0, knockout |
| 130015 | 106 | 21 / 19 | 20.0 / 19.0 | role 0, knockout |

The recorded second fight visibly shows close facing, rapid reacquisition, and
sustained exchanges rather than pilot 10's wide perimeter orbit. Its complete
receipt is `match-recorded.jsonl`; `match1-first-fight.jsonl` normalizes the lethal
edge from an older runner that continued after immediate auto-respawn.

The superseded 5 Hz recording was intentionally removed. Its receipts are kept
as the auditable baseline for the later realtime deployment.

The simulator and checkpoint action contract are 20 Hz (`repeat=1`). The rigorous
Java oracle path serializes authoritative state, client steps, and attack receipts,
so the recorded match measured 5.47 policy decisions per wall-clock second and
3.72 server ticks per decision. This preserves auditable vanilla ordering but is a
known deployment-throughput limitation; it does not change the policy's fine-yaw
semantics. A future server-tick batched injector is required for true wall-clock
20 Hz while retaining the same authoritative trace guarantees.
