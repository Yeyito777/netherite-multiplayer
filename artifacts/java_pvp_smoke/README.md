# Minecraft 1.11.2 two-client PvP smoke

This is the first real-game deployment gate, captured from two isolated nested
X11 clients on 2026-08-10.

- Host `Player0` launched an integrated Minecraft 1.11.2 Forge server.
- `open_lan` disabled online authentication for the private local development
  server and returned its ephemeral LAN port.
- Guest `Player1` connected through the mod's non-interactive `connect` command.
- `pvp_setup` resolved both roles by configured player name, removed non-player
  entities, built one shared 32x32 stone floor, cleared the six blocks above it,
  disabled spawning/daylight/natural regeneration, emptied both inventories,
  restored health, and placed the players at `(-4,65,0)` and `(4,65,0)`.
- `pvp_state` ran on the authoritative server thread and recorded both stable
  UUID roles, raw-bit pose/motion/rotation, health, hunger, hurt resistance,
  death state, and 1.11.2 attack cooldown.
- `pvp_attack` injected an actual `CPacketUseEntity(ATTACK)` through role 0's
  `NetHandlerPlayServer`. Vanilla accepted the in-reach fist attack: role 1
  changed from 20 to 19 health, entered 20 ticks of hurt resistance, and role
  0's attack-strength cooldown reset.

`host.png` and `guest.png` are the two client views of the same arena.
`state.json` is the authoritative two-player setup receipt and
`attack_receipt.json` contains state immediately around the accepted vanilla
attack. This proves private two-client transport, arena setup, UUID-stable state
capture, and the authoritative melee injection primitive. Full simulator combat
parity and closed-loop checkpoint control remain subsequent gates.
