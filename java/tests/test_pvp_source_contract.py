"""Asset-free structural gates for the selected 1.11.2 PvP mod path."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1] / "Minecraft" / "src" / "main"
JAVA = ROOT / "java"


def java_sources():
    return list(JAVA.rglob("*.java"))


def test_exactly_one_qrl_forge_mod_and_it_is_the_selected_recorder():
    declarations = []
    for path in java_sources():
        text = path.read_text()
        if "@Mod(" in text and re.search(r'MODID\s*=\s*"qrl"', text):
            declarations.append(path.relative_to(JAVA).as_posix())
    assert declarations == ["netheritemod/Recorder.java"]


def test_oracle_contract_has_no_implicit_player_selection():
    paths = list((JAVA / "netheritemod").glob("*Pvp*.java"))
    for path in paths:
        text = path.read_text()
        assert "getPlayers().get(0)" not in text
        assert "@p" not in text


def test_private_two_client_and_authoritative_attack_commands_are_wired():
    text = (JAVA / "netheritemod" / "Recorder.java").read_text()
    for command in ("open_lan", "connect", "pvp_setup", "pvp_state", "pvp_attack"):
        assert f'case "{command}"' in text
    assert "server.setOnlineMode(false)" in text
    assert "getPlayerByUsername" in text
    assert "server.addScheduledTask" in text
    assert "attacker.connection.processUseEntity" in text
    assert "new net.minecraft.network.play.client.CPacketUseEntity(target)" in text
    assert 'r.action.has("lateral0")' in text
    assert 'r.action.has("yaw_delta1")' in text


def test_lockstep_and_pvp_responses_cannot_be_lost_to_synchronous_queue_race():
    text = (JAVA / "netheritemod" / "Recorder.java").read_text()
    assert "reply(inFlight, obs);" in text
    for name in ("setupReq", "attackReq", "stateReq"):
        assert f"{name}.resp.offer(" not in text


def test_v2_setup_equips_fixed_iron_gear_and_observes_shield_state():
    text = (JAVA / "netheritemod" / "Recorder.java").read_text()
    for item in ("IRON_SWORD", "IRON_AXE", "SHIELD", "IRON_HELMET",
                 "IRON_CHESTPLATE", "IRON_LEGGINGS", "IRON_BOOTS"):
        assert f"net.minecraft.init.Items.{item}" in text
    assert "EntityEquipmentSlot.OFFHAND" in text
    assert '"blocking"' in text
    assert '"shield_disabled"' in text
    assert '"shield_use_ticks"' in text


def test_pvp_reset_clears_stale_visibility_and_deployment_gates_both_views():
    recorder = (JAVA / "netheritemod" / "Recorder.java").read_text()
    for reset in ("clearActivePotions()", "setInvisible(false)",
                  "setGameType(net.minecraft.world.GameType.SURVIVAL)"):
        assert reset in recorder
    assert '"invisible_to_viewer"' in recorder
    deploy = (ROOT.parents[2] / "deploy_pvp_checkpoint.py").read_text()
    assert "wait_for_mutual_visibility(pool)" in deploy
    assert "real-client mutual-visibility preflight failed" in deploy


def test_deployment_observation_exposes_parity_clocks():
    recorder = (JAVA / "netheritemod" / "Recorder.java").read_text()
    for field in ("client_tick", "world_tick", "action_apply_client_tick",
                  "action_apply_world_tick", "action_apply_nano_time"):
        assert f'"{field}"' in recorder
    deploy = (ROOT.parents[2] / "deploy_pvp_checkpoint.py").read_text()
    assert "def timing_summary(rows):" in deploy
    assert '"same_tick_fraction"' in deploy
    assert '"exactly_one_fraction"' in deploy
