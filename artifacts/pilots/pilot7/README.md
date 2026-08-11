# Pilot 7: first KL-gated curriculum

This run added a preserved pre-PPO checkpoint, per-head behavior-cloning metrics,
a static/chaser curriculum, and a nominal `0.01` PPO KL gate. Ten BC epochs were
not enough: forward, yaw, and attack accuracy were 71.0%, 54.7%, and 75.2%, and
the frozen policy landed no hits against a stationary target.

Self-play produced 245,740 hits and 9,606 kills, but the final greedy policy still
landed no stationary hits and lost 85/128 chaser evaluations. Diagnostics also
revealed that KL was being measured from a retained pre-step autograd output rather
than from a fresh post-update policy evaluation.

Verdict: failed candidate. It motivated longer BC fitting and corrected post-update
trust-region measurement.
