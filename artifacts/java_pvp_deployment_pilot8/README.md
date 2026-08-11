# Pilot 8 checkpoint in real Minecraft 1.11.2

The `artifacts/pilots/pilot8/selfplay.pt` policies ran closed-loop through two real
Forge clients after a fresh private LAN reset and `stone32` arena setup. Both
players were controlled from authoritative integrated-server state. Movement and
look went through each client; melee went through canonical role-ordered
`NetHandlerPlayServer` attack injection.

## Result

- 100/100 policy decisions completed;
- role 0: 5 accepted vanilla hits and 5.0 damage;
- role 1: 4 accepted vanilla hits and 3.5 damage;
- final observed health: role 0 = 16.5, role 1 = 15.0;
- no deaths in this 100-decision match;
- five distinct action vectors from each policy.

The corrected model approached, stopped to punch, and selected both left and right
yaw corrections. It is no longer pilot 4's single constant forward/attack policy,
although both agents still spend too much time circling into the perimeter after
the first exchange.

## Deployment liveness fix

Earlier attempts appeared to hang after 5-18 decisions. The root cause was not
Minecraft or the model: step finalization used a zero-timeout
`SynchronousQueue.offer`. If the socket thread had not entered `poll` at that exact
instant, Java silently dropped the response and the bridge waited for its 120-second
timeout. The PvP bridge now uses the existing timed `reply` handoff for lockstep and
server-thread PvP responses. A source-contract regression test protects this fix.

`metrics.jsonl` is the complete decision/action/attack receipt. The screenshots are
the final views from both real clients.
