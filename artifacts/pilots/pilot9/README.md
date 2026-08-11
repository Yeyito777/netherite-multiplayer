# Pilot 9: corrected post-update KL measurement

This run repeated the 50-epoch behavior clone with a `3e-5` PPO learning rate and
recomputed KL from the policy after each optimizer step. The BC initialization
again won 97/128 scripted-boxer evaluations.

A single Adam update moved the policy far outside the intended trust region:
post-update KL was about 1.40 in chunk 0 despite gradient clipping. The gate then
stopped every chunk after one update, but cannot undo that first oversized step.
By the final evaluation the policy won 105/128 against the scripted boxer, yet
stationary performance fell from six wins/456 hits to one win/308 hits. Self-play
recorded 650,599 hits and 10,043 kills.

Verdict: not retained as a deployment checkpoint. It is useful evidence that the
next PPO recipe needs a much smaller step, an adaptive backtracking update, or BC
regularization—not merely an epoch-level early-stop gate.
