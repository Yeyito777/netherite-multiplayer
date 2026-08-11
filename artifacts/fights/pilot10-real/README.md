# Pilot 10 real-Minecraft AI-vs-AI fight

Two independently optimized pilot 10 policies were deployed through two real
Minecraft 1.11.2 Forge clients connected to the private integrated server. The
fight used stochastic policy sampling and the authoritative vanilla attack path.

Result after the 700-decision time limit:

- 39 accepted vanilla hits (20 by role 0, 19 by role 1);
- 19.5 damage by role 0 and 18.0 by role 1;
- final health: role 0 = 2.0, role 1 = 0.5;
- role 0 wins by health at the time limit.

The superseded Pilot 10 recording was intentionally removed. This directory
retains the compact metrics and contact sheet as the coarse-control baseline.
