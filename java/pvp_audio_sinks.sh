#!/usr/bin/env bash
# Create isolated PipeWire/PulseAudio outputs for two Minecraft POV recordings.
# Launch each client with PULSE_SINK=netherite_pvp{0,1}, then record from the
# corresponding netherite_pvp{0,1}.monitor source. No fight audio reaches the
# user's physical speakers and the two client mixes cannot contaminate each other.
set -euo pipefail

sinks=(netherite_pvp0 netherite_pvp1)

create() {
    local sink
    for sink in "${sinks[@]}"; do
        if ! pactl list short sinks | awk '{print $2}' | grep -Fxq "$sink"; then
            pactl load-module module-null-sink \
                "sink_name=$sink" \
                "sink_properties=device.description=Netherite-${sink##*pvp}-POV" \
                rate=48000 channels=2 >/dev/null
        fi
    done
    pactl list short sinks | grep -E $'\tnetherite_pvp[01]\t'
}

destroy() {
    local id name args
    while IFS=$'\t' read -r id name args _; do
        if [[ $name == module-null-sink && $args =~ sink_name=(netherite_pvp0|netherite_pvp1) ]]; then
            pactl unload-module "$id"
        fi
    done < <(pactl list short modules)
}

case "${1:-}" in
    create) create ;;
    destroy) destroy ;;
    *) echo "usage: $0 {create|destroy}" >&2; exit 2 ;;
esac
