"""Tests for ArmClient against a stand-in server process (no arm, no LeRobot)."""

from __future__ import annotations

import sys
import textwrap
import threading
from pathlib import Path

from reachy_emotion import arm_client

_FAKE_SERVER = textwrap.dedent("""
    import json, os, sys, time
    mode = sys.argv[1]
    if mode == "fail" or os.environ.get("ARM_ID") == "broken":
        print(json.dumps({"ready": False, "error": "no arm on that port"}), flush=True)
        sys.exit(1)
    arm = os.environ.get("ARM_ID", "follower")
    print(json.dumps({"ready": True, "arm": arm, "actions": ["rest", "wave", "stop"]}), flush=True)
    if mode == "event":
        print(json.dumps({"event": "fault", "arm": arm, "faults": {"gripper": ["overload"]}}), flush=True)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["action"] == "crash":
            sys.exit(3)
        if msg["action"] == "wave":
            time.sleep(0.2)
        reply = {"id": msg["id"], "ok": True, "did": msg["action"]}
        if "args" in msg:
            reply["args"] = msg["args"]
        print(json.dumps(reply), flush=True)
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
    assert arm.name == "follower" and arm.actions == ["rest", "wave", "stop"]
    assert arm.call("wave") == {"ok": True, "did": "wave"}
    arm.close()
    assert arm._proc.poll() == 0
    assert arm.call("rest") == {"ok": False, "error": "arm server is not running"}


def test_args_go_in_the_request_only_when_given(tmp_path):
    arm = _client(tmp_path)
    arm.start()
    target = {"x": 0.3, "y": 0.18, "yaw_deg": 95.0, "label": "pink eraser"}
    assert arm.call("grasp", target, timeout=5) == {"ok": True, "did": "grasp", "args": target}
    assert arm.submit("hover", {"x": 0.3, "y": 0.18}).result(5)["args"] == {"x": 0.3, "y": 0.18}
    assert arm.call("rest", timeout=5) == {"ok": True, "did": "rest"}  # no "args" key at all
    arm.close()


def test_close_waits_long_enough_to_put_something_back():
    import inspect

    # A put-back plus the rest can take well over 20 s; killing the server mid-move
    # would leave the arm powered where it stopped.
    assert inspect.signature(arm_client.ArmClient.close).parameters["timeout"].default >= 60


def test_start_reports_the_servers_error(tmp_path):
    arm = _client(tmp_path, "fail")
    try:
        arm.start()
    except RuntimeError as exc:
        assert "no arm on that port" in str(exc)
    else:
        raise AssertionError("start() should fail")


def test_submit_sends_in_order_and_resolves_each_reply(tmp_path):
    arm = _client(tmp_path)
    arm.start()
    futures = [arm.submit(a) for a in ("wave", "rest", "wave")]  # no waiting in between
    assert [f.result(5)["did"] for f in futures] == ["wave", "rest", "wave"]
    arm.close()


def test_a_caller_giving_up_does_not_break_the_reader(tmp_path):
    # Ending the session cancels the asyncio side of an in-flight move; the server's late
    # reply must still be taken cleanly, so later replies (the shutdown) keep arriving.
    import asyncio

    arm = _client(tmp_path)
    arm.start()

    async def give_up():
        task = asyncio.ensure_future(asyncio.wrap_future(arm.submit("wave")))  # takes 0.2 s
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(give_up())
    assert arm.call("rest", timeout=5) == {"ok": True, "did": "rest"}
    arm.close()
    assert arm._proc.poll() == 0


def test_events_before_anyone_listens_are_handed_over_later(tmp_path):
    arm = _client(tmp_path, "event")
    arm.start()
    assert arm.call("rest", timeout=5)["ok"]  # the event line came before this reply
    events = []
    arm.on_event = events.append
    assert events == [{"event": "fault", "arm": "follower", "faults": {"gripper": ["overload"]}}]
    arm.close()


def test_a_server_crash_fails_the_waiting_call(tmp_path):
    arm = _client(tmp_path)
    arm.start()
    assert arm.call("crash", timeout=5) == {"ok": False, "error": "arm server exited"}


def test_no_arm_configured(monkeypatch):
    monkeypatch.setattr("reachy_emotion.conversation_app._load_env", lambda: None)
    monkeypatch.delenv("ARM_PORT", raising=False)
    assert arm_client.arms_from_env() == {}


def test_server_events_reach_on_event_not_the_callers(tmp_path):
    arm = _client(tmp_path, "event")
    events, got = [], threading.Event()
    arm.on_event = lambda e: (events.append(e), got.set())
    arm.start()
    assert got.wait(5)
    assert events == [{"event": "fault", "arm": "follower", "faults": {"gripper": ["overload"]}}]
    assert arm.call("wave") == {"ok": True, "did": "wave"}  # replies still go to their caller
    arm.close()


def test_each_arm_gets_its_own_server_and_a_broken_one_is_left_out(tmp_path, monkeypatch):
    # A second follower arm is one more config entry; one that fails to start doesn't
    # take the others down.
    script = tmp_path / "fake_server.py"
    script.write_text(_FAKE_SERVER)
    monkeypatch.setattr("reachy_emotion.conversation_app._load_env", lambda: None)
    monkeypatch.setattr(arm_client, "ARM_PYTHON", Path(sys.executable))
    monkeypatch.setattr(arm_client, "SERVER", script)  # argv[1] is "serve": a working arm
    monkeypatch.setattr(arm_client, "_arm_configs", lambda: [
        {"ARM_ID": "right"}, {"ARM_ID": "broken"}, {"ARM_ID": "left"},
    ])
    arms = arm_client.arms_from_env()
    try:
        assert sorted(arms) == ["left", "right"]
        assert arms["left"].call("wave") == {"ok": True, "did": "wave"}
    finally:
        for arm in arms.values():
            arm.close()
