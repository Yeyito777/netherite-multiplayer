# Pilot 5: corrected movement basis and randomized engagement

Configuration: corrected `movement_v2` egocentric observation basis, independent
lateral/yaw reset perturbations, 128 teacher steps, six behavior-cloning epochs,
and 80 adversarial PPO chunks on 2,048 L4 matches.

The run sustained heavy self-play engagement and deaths, but the frozen greedy
policy scored 0/128 against the stationary target at the final evaluation.
Behavior cloning reported 87.4% aggregate categorical accuracy, yet this was
misleading: common straight/no-attack labels dominated accuracy while rare turn
and attack classes were missed.

Verdict: failed deployment candidate. The next recipe uses per-head inverse
frequency weights during behavior cloning so straight/no-attack predictions
cannot hide failure to learn the action classes boxing actually requires.
