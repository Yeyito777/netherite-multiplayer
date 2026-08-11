# Pilot 8: competent behavior clone with conservative self-play

This run used 50 class-balanced behavior-cloning epochs followed by 40 curriculum
and adversarial chunks with a deliberately tiny `1e-6` PPO learning rate. Final
BC accuracy was 96.0% forward, 87.9% yaw, and 96.6% attack.

The preserved BC policy won 97/128 evaluations against the scripted boxer, lost
10, drew 21, and landed 456 hits against stationary targets. After self-play, the
greedy policy won 100/128 against the boxer, lost 9, drew 19, and retained 469
stationary hits. Training accumulated 518,818 hits and 7,706 kills.

This is the first corrected-observation deployment candidate with clearly dynamic
engagement behavior. Both `behavior_clone.pt` and `selfplay.pt` are retained; the
BC checkpoint is the cleaner control and the self-play checkpoint is the primary
candidate. The run also confirmed that the old KL diagnostic was invalid, leading
to a fresh post-update recomputation in the next code revision.
