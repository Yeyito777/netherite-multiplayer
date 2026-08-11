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
