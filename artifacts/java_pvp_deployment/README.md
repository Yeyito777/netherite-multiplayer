# Pilot 4 checkpoint deployed into real Minecraft 1.11.2

The two policies from `artifacts/pilots/pilot4/selfplay.pt` ran closed-loop for
100 decisions through two real Forge clients connected to the private integrated
server. Every decision decoded the authoritative two-player server state into
the same 24 structured observations used by CUDA, evaluated the frozen PyTorch
policies, applied movement/look through each client bridge, and submitted attack
intents in canonical role order through `NetHandlerPlayServer`.

Result:

- 100 completed policy decisions;
- role 0: 3 accepted vanilla hits, 3.5 damage;
- role 1: 3 accepted vanilla hits, 3.0 damage;
- no deaths;
- health changed from `[20,20]` to `[17,16.5]`.

This proves actual checkpoint inference and accepted player-vs-player damage in
Minecraft—not merely simulator replay. It also exposed a real policy defect:
both greedy policies selected `forward+sprint+attack` for every observed state.
They exchanged hits during the initial charge, passed one another, and then ran
into the arena boundary without turning back. `host_final.png` and
`guest_final.png` show this terminal wall-facing behavior.

`metrics.jsonl` is the decision/action/attack receipt. Root-cause audit found
that the original egocentric observation used the wrong X signs when projecting
onto Minecraft's `right=(cos,sin)` and `forward=(-sin,cos)` movement basis. That
made front/behind ambiguous across arena axes and helped produce this degenerate
policy. The next recipe uses corrected `movement_v2` observations, independent
lateral/yaw reset perturbations, and a teacher that explicitly turns when the
opponent is behind. Repeated deployment resets would conceal rather than repair
this failure.
