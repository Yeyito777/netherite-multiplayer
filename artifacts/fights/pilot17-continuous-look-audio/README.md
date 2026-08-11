# V1.2 per-POV audio capture

This deployment validates synchronized sound in the two real Minecraft 1.11.2
POV recordings. Each JVM is launched into a separate PipeWire/PulseAudio null
sink (`netherite_pvp0` or `netherite_pvp1`), and FFmpeg records that sink's
monitor alongside the matching X11 display. Consequently each video contains
its own player's spatial game mix instead of a shared desktop mix or microphone.

`video_prepare` standardizes master/player/block volume, disables random music,
and retains the smooth-start render profile. Both outputs contain 48 kHz stereo
AAC audio and 20 fps H.264 video. The trace records 323 decisions, 28/26 accepted
hits, and a role-1 knockout. The policy loop measured 17.59 decisions/s during
this host-contention run; audio/video timestamps remain synchronized.

- [Player 0 POV with audio](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-smooth-audio-player0-pov.mp4)
- [Player 1 POV with audio](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-smooth-audio-player1-pov.mp4)
