# V1.2 smooth-start real-Minecraft fight

This reruns Pilot 17 after fixing the low-frame-rate opening seen in the first
V1.2 recordings. The cause was not Minecraft's lost-focus throttle: both LWJGL
displays reported active and the qrl guard had already disabled
`pauseOnLostFocus`. Capture began immediately after the arena teleport, while
both clients were rebuilding the flat-world view, and CPU video encoding
competed with the two render loops.

The fixed path:

1. applies a recording profile that retains the complete 32x32 arena but uses a
   three-chunk render distance and disables unnecessary graphics work;
2. waits ten seconds after `pvp_setup` for chunk rebuild and JVM/render warm-up;
3. exposes ready/start files so capture begins only after that warm-up;
4. encodes the two full-resolution captures with AMD VAAPI rather than stealing
   CPU from Minecraft; and
5. trims the half-second capture-arm lead-in.

The real-client policy loop sustained 19.30 decisions/s while recording both
1918x1058 POVs at nominal 20 fps. Role 0 won after 299 decisions; accepted hits
were 35/35 and damage was 20.0/19.416.

- [Player 0 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-smooth-player0-pov.mp4)
- [Player 1 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-smooth-player1-pov.mp4)
