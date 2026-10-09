"""Tests for ArmClient against a stand-in server process (no arm, no LeRobot)."""

from __future__ import annotations

import sys
import textwrap

from reachy_emotion import arm_client

_FAKE_SERVER = textwrap.dedent("""
    import json, sys, time
    mode = sys.argv[1]
    if mode == "fail":
        print(json.dumps({"ready": False, "error": "no arm on that port"}), flush=True)
        sys.exit(1)
    print(json.dumps({"ready": True, "actions": ["rest", "wave", "stop"]}), flush=True)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["action"] == "crash":
            sys.exit(3)
        if msg["action"] == "wave":
            time.sleep(0.2)
        print(json.dumps({"id": msg["id"], "ok": True, "did": msg["action"]}), flush=True)
        if msg["action"] == "shutdown":
            break
""")


def _client(tmp_path, mode="ok"):
    script = tmp_path / "fake_server.py"
    script.write_text(_FAKE_SERVER)
    return arm_client.ArmClient([sys.executable, str(script), mode], ready_timeout=10)


def test_start_call_and_close(tmp_path):
    arm = _client(tmp_path)
    arm.start()
    assert arm.actions == ["rest", "wave", "stop"]
    assert arm.call("wave") == {"ok": True, "did": "wave"}
    arm.close()
    assert arm._proc.poll() == 0
    assert arm.call("rest") == {"ok": False, "error": "arm server is not running"}


def test_start_reports_the_servers_error(tmp_path):
    arm = _client(tmp_path, "fail")
    try:
        arm.start()
    except RuntimeError as exc:
        assert "no arm on that port" in str(exc)
    else:
        raise AssertionError("start() should fail")


def test_a_server_crash_fails_the_waiting_call(tmp_path):
    arm = _client(tmp_path)
    arm.start()
    assert arm.call("crash", timeout=5) == {"ok": False, "error": "arm server exited"}


def test_no_arm_configured(monkeypatch):
    monkeypatch.setattr("reachy_emotion.conversation_app._load_env", lambda: None)
    monkeypatch.delenv("ARM_PORT", raising=False)
    assert arm_client.arm_from_env() is None
