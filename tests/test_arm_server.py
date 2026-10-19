"""Tests for the SO-101 arm server logic (arm_server.py) with a fake LeRobot bus — no arm."""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_emotion import arm_server as srv

REST = {"shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
        "wrist_flex": 60.0, "wrist_roll": 0.0, "gripper": 5.0}
ALL_OFF = [("off", m) for m in REST]  # rest lets go of each motor in turn


class FakePort:
    def __init__(self):
        self.is_using = False  # left True when a Ctrl-C lands mid-packet

    def clearPort(self):
        pass


class FakeBus:
    """Joints follow their goals instantly; records the order of writes. ``status`` sets a
    servo's Feetech status bits (0x20 = overload). ``block`` keeps a joint within (lo, hi)
    whatever its goal (an object between the jaws, the table under the arm). Present_Load
    is ``load`` (per motor) plus 40 per unit of lag behind the goal, like a P controller."""

    def __init__(self, pos, torque=0, stuck=(), block=None, load=None):
        self.pos = dict(pos)
        self.goal = dict(pos)
        self.torque = torque
        self.stuck = set(stuck)  # joints that can't move (blocked)
        self.block = dict(block or {})
        self.load = dict(load or {})
        self.status = {}
        self.events = []
        self.port_handler = FakePort()
        self.motors = {m: SimpleNamespace(id=i) for i, m in enumerate(self.pos, start=1)}

    def sync_read(self, name, motors=None, num_retry=0, normalize=True):
        if self.port_handler.is_using:
            raise ConnectionError("Port is in use!")
        if name == "Torque_Enable":
            return {m: self.torque for m in self.pos}
        if name == "Status":
            return {m: self.status.get(m, 0) for m in self.pos}
        if isinstance(motors, str):
            motors = [motors]
        names = [m for m in self.pos if motors is None or m in motors]
        if name == "Present_Load":
            return {m: self.load.get(m, 0) + round(40 * abs(self.goal[m] - self.pos[m])) for m in names}
        return {m: self.pos[m] for m in names}

    def sync_write(self, name, values):
        self.events.append(("goal", dict(values)))
        for m, v in values.items():
            self.goal[m] = v
            if m not in self.stuck:
                lo, hi = self.block.get(m, (-1e9, 1e9))
                self.pos[m] = min(max(v, lo), hi)

    def enable_torque(self, motors=None):
        self.events.append(("on", motors))

    def disable_torque(self, motors=None, num_retry=0):
        self.events.append(("off", motors))

    def disconnect(self, disable_torque=True):
        self.events.append(("disconnect", disable_torque))


class FakeRobot:
    def __init__(self, pos, on_action=None, **bus):
        self.bus = FakeBus(pos, **bus)
        self.on_action = on_action
        self.actions = 0
        self.configured = 0
        self.sent = []  # (goals sent, where the joints were just before)

    def configure(self):
        # Like LeRobot: torque off while writing settings, then back on.
        self.configured += 1
        self.bus.disable_torque()
        self.bus.enable_torque()

    def send_action(self, action):
        self.actions += 1
        if self.on_action:
            self.on_action(self)
        goals = {k.removesuffix(".pos"): v for k, v in action.items()}
        self.sent.append((goals, dict(self.bus.pos)))
        self.bus.sync_write("Goal_Position", goals)
        return action


def _motions(**gestures):
    poses = {"rest": dict(REST), "gripper_open": {**REST, "gripper": 80.0},
             "gripper_closed": {**REST, "gripper": 2.0}}
    return {"poses": poses, "gestures": gestures}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(srv.time, "sleep", lambda s: None)


def test_hold_sets_target_to_present_before_torque_on():
    # A joint moved by hand while relaxed must not jump back when torque comes on.
    robot = FakeRobot({**REST, "elbow_flex": 30.0})
    srv.Arm(robot, _motions()).hold()
    assert robot.bus.events == [("goal", {**REST, "elbow_flex": 30.0}), ("on", None)]


def test_move_to_is_smooth_and_ends_on_target():
    robot = FakeRobot(REST)
    arm = srv.Arm(robot, _motions())
    seen = []
    robot.on_action = lambda r: seen.append(r.bus.pos["shoulder_pan"])
    arm.move_to({"shoulder_pan": 45.0})
    assert robot.bus.pos["shoulder_pan"] == pytest.approx(45.0)
    assert seen == sorted(seen) and len(seen) > 10  # many small increasing steps, no jump


def test_rest_moves_to_rest_then_turns_motors_off():
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    srv.Arm(robot, _motions()).rest()
    assert robot.bus.pos == pytest.approx(REST)
    assert robot.bus.events[-6:] == ALL_OFF


def test_gripper_moves_only_the_gripper_with_only_its_torque_on():
    robot = FakeRobot(REST)
    srv.Arm(robot, _motions()).run("gripper_open")
    assert robot.bus.pos["gripper"] == pytest.approx(80.0)
    assert ("on", "gripper") in robot.bus.events and ("on", None) not in robot.bus.events
    assert all(set(v) == {"gripper"} for k, v in robot.bus.events if k == "goal")


def test_gesture_replays_then_returns_to_rest_motors_off():
    frames = [[0.0, {**REST, "shoulder_pan": 20.0}], [0.1, {**REST, "shoulder_pan": -20.0}],
              [0.2, {**REST, "shoulder_pan": 20.0}]]
    robot = FakeRobot(REST)
    srv.Arm(robot, _motions(wave={"fps": 30, "frames": frames})).run("wave")
    assert robot.bus.pos == pytest.approx(REST)
    assert robot.bus.events[-6:] == ALL_OFF
    assert any(v.get("shoulder_pan") == -20.0 for k, v in robot.bus.events if k == "goal")


def test_untaught_moves_give_a_clear_error():
    arm = srv.Arm(FakeRobot(REST), {"poses": {}, "gestures": {}})
    with pytest.raises(RuntimeError, match="scripts/arm.sh teach rest"):
        arm.run("rest")
    with pytest.raises(RuntimeError, match="scripts/arm.sh record wave"):
        arm.run("wave")
    assert srv.Arm(FakeRobot(REST), _motions()).actions() == ["rest", "gripper_open", "gripper_close", "stop"]


def _controller(robot, motions):
    replies, got = [], threading.Event()

    def reply(msg):
        replies.append(msg)
        got.set()

    c = srv.Controller(srv.Arm(robot, motions), reply)
    threading.Thread(target=c.work, daemon=True).start()
    return c, replies


def _wait_for(replies, n):
    for _ in range(500):
        if len(replies) >= n:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"expected {n} replies, got {replies}")


def _blocking_robot(at_action=30):
    """A robot whose send_action pauses at the *at_action*-th command until released."""
    started, release = threading.Event(), threading.Event()

    def on_action(robot):
        if robot.actions == at_action:
            started.set()
            release.wait(2)

    return FakeRobot(REST, on_action), started, release


def test_stop_interrupts_a_gesture_cancels_the_queue_and_freezes_in_place():
    frames = [[i * 0.03, {**REST, "shoulder_pan": float(i)}] for i in range(200)]
    robot, started, release = _blocking_robot()
    c, replies = _controller(robot, _motions(wave={"fps": 30, "frames": frames}))
    c.handle({"id": 1, "action": "wave"})
    assert started.wait(2)
    c.handle({"id": 2, "action": "wave"})  # queued behind the running wave
    c.handle({"id": 3, "action": "stop"})
    release.set()
    _wait_for(replies, 3)
    assert replies[0] == {"id": 2, "ok": False, "error": "cancelled: stop", "cancelled": True}
    assert {"id": 1, "ok": False, "error": "stopped", "cancelled": True} in replies
    assert replies[-1] == {"id": 3, "ok": True}
    assert robot.bus.events[-1] == ("on", None)  # frozen: motors on where it stopped
    assert robot.bus.pos["shoulder_pan"] < 199  # it never finished the gesture


def test_moves_queue_up_and_run_in_order():
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    c, replies = _controller(robot, _motions())
    for rid, action in enumerate(["gripper_open", "gripper_close", "gripper_open", "rest"], start=1):
        c.handle({"id": rid, "action": action})
    _wait_for(replies, 4)
    assert replies == [{"id": i, "ok": True} for i in range(1, 5)]  # no "busy": each waited its turn
    assert robot.bus.pos == pytest.approx(REST)  # the last one (rest) ran last
    assert robot.bus.events[-6:] == ALL_OFF


def test_a_full_queue_refuses_more_moves():
    robot, started, release = _blocking_robot(at_action=1)
    c, replies = _controller(robot, _motions())
    c.handle({"id": 0, "action": "gripper_open"})
    assert started.wait(2)
    for rid in range(1, srv._QUEUE_MAX + 2):
        c.handle({"id": rid, "action": "gripper_close"})
    assert replies == [{"id": srv._QUEUE_MAX + 1, "ok": False, "executed": False,
                        "error": f"too many arm moves queued (max {srv._QUEUE_MAX})"}]
    release.set()
    _wait_for(replies, srv._QUEUE_MAX + 2)
    assert all(r["ok"] for r in replies[1:])  # the queued ones all ran


def test_a_motor_fault_mid_move_cancels_the_queue_freezes_and_is_announced(monkeypatch):
    monkeypatch.setattr(srv, "_FAULT_POLL_S", 0.0)  # check the servos every tick
    frames = [[i * 0.03, {**REST, "wrist_roll": float(i)}] for i in range(100)]

    def overload(robot):
        if robot.actions == 40:
            robot.bus.status["wrist_roll"] = 0x20

    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot(REST, overload)
    c, replies = _controller(robot, _motions(wave={"fps": 30, "frames": frames}))
    c.handle({"id": 1, "action": "wave"})
    c.handle({"id": 2, "action": "gripper_open"})
    _wait_for(replies, 3)
    threading.Event().wait(0.1)  # idle checks still see the bit: not announced a second time
    assert len(replies) == 3
    wave, cancelled, event = replies
    assert wave["id"] == 1 and not wave["ok"] and wave["faults"] == {"wrist_roll": ["overload"]}
    assert "motor fault: wrist_roll overload; arm frozen in place" == wave["error"]
    assert cancelled == {"id": 2, "ok": False, "error": "cancelled: motor fault: wrist_roll overload",
                         "cancelled": True}
    assert event == {"event": "fault", "arm": "arm", "faults": {"wrist_roll": ["overload"]},
                     "interrupted": True, "cancelled": 1, "detail": "arm frozen in place"}
    kind, goal = robot.bus.events[-1]
    assert kind == "goal" and goal == robot.bus.pos  # frozen where it was: target = present
    assert robot.bus.pos["wrist_roll"] < 99  # the wave never finished


def test_a_fault_while_holding_a_pose_is_announced_once(monkeypatch):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot(REST)
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "gripper_open"})  # the gripper is left holding
    _wait_for(replies, 1)
    robot.bus.status["gripper"] = 0x20  # squeezing something, say
    _wait_for(replies, 2)
    threading.Event().wait(0.1)  # several more idle checks
    assert replies[1:] == [{"event": "fault", "arm": "arm", "faults": {"gripper": ["overload"]},
                            "interrupted": False, "cancelled": 0, "detail": "arm frozen in place"}]
    robot.bus.status.clear()  # cleared, then trips again: announced again
    threading.Event().wait(0.05)
    robot.bus.status["gripper"] = 0x08
    _wait_for(replies, 3)
    assert replies[2]["faults"] == {"gripper": ["overcurrent"]}


def test_a_write_reporting_overload_counts_as_a_fault():
    robot = FakeRobot(REST)

    def refuse(motors=None):
        raise RuntimeError("Failed to write 'Torque_Enable' on id_=6 with '1' after 1 tries. "
                           "[RxPacketError] Overload error!")

    robot.bus.enable_torque = refuse
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "gripper_open"})
    _wait_for(replies, 2)
    assert replies[0]["faults"] == {"gripper": ["overload"]} and not replies[0]["ok"]
    assert replies[1]["event"] == "fault" and replies[1]["interrupted"]


def test_a_fault_while_parking_on_shutdown_freezes_before_letting_go(monkeypatch):
    monkeypatch.setattr(srv, "_FAULT_POLL_S", 0.0)

    def blocked(robot):  # something under the arm on its way to rest
        if robot.actions == 5:
            robot.bus.status["shoulder_lift"] = 0x20

    robot = FakeRobot({**REST, "shoulder_lift": 0.0}, blocked, stuck={"shoulder_lift"})
    c, replies = _controller(robot, _motions())
    assert c.handle({"id": 9, "action": "shutdown"}) is False
    _wait_for(replies, 2)
    assert replies[0]["id"] == 9 and replies[0]["faults"] == {"shoulder_lift": ["overload"]}
    kind, goal = robot.bus.events[-2]
    assert kind == "goal" and goal == robot.bus.pos  # stopped pushing, then let go of the port
    assert robot.bus.events[-1] == ("disconnect", False)


def test_shutdown_without_a_rest_pose_freezes_and_lets_go():
    robot = FakeRobot(REST)
    c, replies = _controller(robot, {"poses": {}, "gestures": {}})
    c.arm.powered = set(REST)  # held since bring-up, as in serve()
    c.handle({"id": 1, "action": "shutdown"})
    _wait_for(replies, 1)
    assert not replies[0]["ok"] and replies[0]["error"].endswith("; arm frozen in place")
    assert [e[0] for e in robot.bus.events[-2:]] == ["goal", "disconnect"]


def test_losing_the_arm_while_holding_is_announced_once(monkeypatch):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot(REST)
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "gripper_open"})  # holding
    _wait_for(replies, 1)
    read = robot.bus.sync_read

    def unplugged(name, *a, **kw):
        if name == "Status":
            raise ConnectionError("[TxRxResult] There is no status packet!")
        return read(name, *a, **kw)

    robot.bus.sync_read = unplugged
    _wait_for(replies, 2)
    threading.Event().wait(0.1)
    assert replies[1:] == [{"event": "unreachable", "arm": "arm",
                            "detail": "[TxRxResult] There is no status packet!"}]
    robot.bus.sync_read = read  # back again; then lost again: announced again
    threading.Event().wait(0.05)
    robot.bus.sync_read = unplugged
    _wait_for(replies, 3)
    assert replies[2]["event"] == "unreachable"


def test_serve_watches_a_held_arm_even_without_a_rest_pose(monkeypatch):
    # Bring-up leaves the motors holding; with no rest pose to park at, a fault while it
    # holds must still be caught and announced.
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot(REST)
    robot.bus.status["shoulder_lift"] = 0x20
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
    monkeypatch.setattr(srv, "load_motions", lambda path: {"poses": {}, "gestures": {}})
    out = []

    class QuietApp:  # sends nothing for a moment, then exits (stdin closes)
        def __iter__(self):
            threading.Event().wait(0.2)
            return iter([])

    class Out:
        def write(self, s):
            out.append(json.loads(s))

        def flush(self):
            pass

    monkeypatch.setattr(srv.sys, "stdin", QuietApp())
    assert srv.serve({"port": "x", "id": "follower", "motions": None}, out=Out()) == 0
    assert out[0] == {"ready": True, "arm": "follower", "actions": ["stop"]}
    faults = [m for m in out if m.get("event") == "fault"]
    assert len(faults) == 1 and faults[0]["faults"] == {"shoulder_lift": ["overload"]}
    assert not faults[0]["interrupted"]


def _fault_reply(motor_id, kind="Overload"):
    return RuntimeError(f"Failed to write 'Torque_Enable' on id_={motor_id} with '0' after 4 tries. "
                        f"[RxPacketError] {kind} error!")


def test_release_lets_go_of_every_motor_even_if_one_reply_reports_a_fault():
    robot = FakeRobot(REST)
    off = robot.bus.disable_torque

    def disable(motors=None, num_retry=0):
        off(motors, num_retry)
        if motors == "shoulder_lift":  # the write lands; its reply carries an overheat bit
            raise _fault_reply(2, "Overheat")

    robot.bus.disable_torque = disable
    arm = srv.Arm(robot, _motions())
    arm.rest()
    assert robot.bus.events[-6:] == ALL_OFF and not arm.torque_on


def test_a_failed_release_still_tries_every_motor_and_freezes_only_the_unsure_one():
    robot = FakeRobot(REST)

    def disable(motors=None, num_retry=0):
        robot.bus.events.append(("off", motors))
        if motors == "shoulder_lift":  # lost even after retries: may still be on
            raise ConnectionError("[TxRxResult] There is no status packet!")

    robot.bus.disable_torque = disable
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "shutdown"})
    _wait_for(replies, 1)
    assert replies[0]["error"].endswith("; arm frozen in place")
    assert robot.bus.events[-8:-2] == ALL_OFF  # the motors after it were still let go
    kind, goal = robot.bus.events[-2]
    assert kind == "goal" and list(goal) == ["shoulder_lift"]  # nothing re-powered but it
    assert robot.bus.events[-1] == ("disconnect", False)


def test_a_gripper_fault_after_parking_does_not_power_the_parked_joints(monkeypatch):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "rest"})
    c.handle({"id": 2, "action": "gripper_close"})  # only the gripper is powered now
    _wait_for(replies, 2)
    robot.bus.status["gripper"] = 0x20  # closed on something
    _wait_for(replies, 3)
    assert replies[2]["faults"] == {"gripper": ["overload"]}
    kind, goal = robot.bus.events[-1]
    assert kind == "goal" and list(goal) == ["gripper"]


def test_a_fault_found_on_a_write_is_not_announced_again_by_the_idle_check(monkeypatch):
    # The write error names only the first motor; the servos' status shows both.
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    robot = FakeRobot(REST)
    robot.bus.status.update({"shoulder_lift": 0x20, "elbow_flex": 0x20})

    def refuse(motors=None):
        raise _fault_reply(2)

    robot.bus.enable_torque = refuse
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "rest"})
    _wait_for(replies, 2)
    threading.Event().wait(0.1)
    assert [r.get("event") for r in replies] == [None, "fault"]  # the move's error + one event
    both = {"shoulder_lift": ["overload"], "elbow_flex": ["overload"]}
    assert replies[0]["faults"] == both and replies[1]["faults"] == both  # every tripped servo


def test_losing_the_arm_mid_move_is_reported_once(monkeypatch):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    monkeypatch.setattr(srv, "_FAULT_POLL_S", 0.0)
    frames = [[i * 0.03, {**REST, "wrist_roll": float(i)}] for i in range(50)]
    robot = FakeRobot(REST)
    c, replies = _controller(robot, _motions(wave={"fps": 30, "frames": frames}))
    read = robot.bus.sync_read

    def unplugged(name, *a, **kw):
        if name == "Status" and robot.actions > 10:
            raise ConnectionError("[TxRxResult] There is no status packet!")
        return read(name, *a, **kw)

    robot.bus.sync_read = unplugged
    c.handle({"id": 1, "action": "wave"})
    _wait_for(replies, 1)
    threading.Event().wait(0.1)  # the idle check fails too, but the move's error already said it
    assert replies == [{"id": 1, "ok": False, "error": "[TxRxResult] There is no status packet!"}]


def test_a_blocked_park_at_startup_freezes_before_letting_go(monkeypatch):
    monkeypatch.setattr(srv, "_FAULT_POLL_S", 0.0)

    def blocked(robot):
        if robot.actions == 5:
            robot.bus.status["shoulder_lift"] = 0x20

    robot = FakeRobot({**REST, "shoulder_lift": 0.0}, blocked, stuck={"shoulder_lift"})
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
    monkeypatch.setattr(srv, "load_motions", lambda path: _motions())
    out = []

    class Out:
        def write(self, s):
            out.append(json.loads(s))

        def flush(self):
            pass

    assert srv.serve({"port": "x", "id": "follower", "motions": None}, out=Out()) == 1
    assert not out[0]["ready"] and out[0]["error"].endswith("; arm frozen in place")
    kind, goal = robot.bus.events[-2]
    assert kind == "goal" and goal == robot.bus.pos and robot.bus.events[-1] == ("disconnect", False)


def test_other_errors_fail_only_that_move():
    robot = FakeRobot(REST)
    c, replies = _controller(robot, _motions())
    c.handle({"id": 1, "action": "wave"})  # not recorded
    c.handle({"id": 2, "action": "gripper_open"})
    _wait_for(replies, 2)
    assert not replies[0]["ok"] and "isn't recorded" in replies[0]["error"]
    assert replies[1] == {"id": 2, "ok": True}  # the queue carried on


def test_shutdown_parks_at_rest_and_releases_the_port():
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    c, replies = _controller(robot, _motions())
    assert c.handle({"id": 7, "action": "shutdown"}) is False
    _wait_for(replies, 1)
    assert replies == [{"id": 7, "ok": True}]
    assert robot.bus.pos == pytest.approx(REST)
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]


def test_motions_roundtrip_and_dotenv(tmp_path):
    path = tmp_path / "arm_motions.json"
    assert srv.load_motions(path) == {"poses": {}, "gestures": {}}
    srv.save_motions(path, _motions())
    assert srv.load_motions(path) == _motions()
    env = tmp_path / ".env"
    env.write_text("# comment\nARM_PORT=/dev/tty.usbmodem1\nLEADER_ID='lead'\nGEMINI_SYSTEM_PROMPT=a=b c\n")
    assert srv._read_dotenv(env) == {
        "ARM_PORT": "/dev/tty.usbmodem1", "LEADER_ID": "lead", "GEMINI_SYSTEM_PROMPT": "a=b c",
    }


def test_rest_keeps_the_motors_on_if_the_arm_cant_get_there():
    # A blocked joint must not be let go short of rest (it would drop).
    robot = FakeRobot({**REST, "shoulder_lift": 0.0}, stuck={"shoulder_lift"})
    with pytest.raises(RuntimeError, match="didn't reach rest .*; holding it in place"):
        srv.Arm(robot, _motions()).rest()
    kind, goal = robot.bus.events[-1]
    assert kind == "goal" and goal == robot.bus.pos  # held where it is, not pushing on
    assert not [e for e in robot.bus.events if e[0] == "off"]


def test_bring_up_never_drops_an_arm_that_is_still_holding():
    held = FakeRobot({**REST, "shoulder_lift": 0.0}, torque=1)
    srv._bring_up(held)
    assert held.configured == 0  # configure() would turn torque off and let it sag
    assert held.bus.events == [("goal", held.bus.pos), ("on", None)]

    limp = FakeRobot(REST, torque=0)
    srv._bring_up(limp)
    assert limp.configured == 1
    assert limp.bus.events[0] == ("goal", REST)  # target = present before torque comes on


def test_serve_still_parks_when_the_app_is_gone(monkeypatch):
    # The app dies mid-wave: stdin closes and every reply hits a broken pipe.
    frames = [[i * 0.03, {**REST, "shoulder_pan": float(i)}] for i in range(50)]
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
    monkeypatch.setattr(srv, "load_motions", lambda path: _motions(wave={"fps": 30, "frames": frames}))

    class DeadPipe:
        def write(self, s):
            if '"ready"' not in s:
                raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    monkeypatch.setattr(srv.sys, "stdin", io.StringIO('{"id": 1, "action": "wave"}\n'))
    assert srv.serve({"port": "x", "id": "follower", "motions": None}, out=DeadPipe()) == 0
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]  # parked at rest
    assert robot.bus.pos == pytest.approx(REST)


def test_ctrl_c_while_recording_parks_the_follower_and_saves_nothing(monkeypatch, tmp_path):
    import types

    robot = FakeRobot({**REST, "shoulder_lift": 0.0}, torque=1)

    class FakeLeader:
        is_calibrated, is_connected, reads = True, False, 0

        def __init__(self, config):
            pass

        def connect(self, calibrate=True):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_action(self):
            self.reads += 1
            if self.reads == 5:  # Ctrl-C lands mid-packet on the follower's port
                robot.bus.port_handler.is_using = True
                raise KeyboardInterrupt
            return {f"{m}.pos": v + self.reads for m, v in REST.items()}

    leader_mod = types.ModuleType("lerobot.teleoperators.so_leader")
    leader_mod.SO101Leader, leader_mod.SO101LeaderConfig = FakeLeader, lambda **kw: kw
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.so_leader", leader_mod)
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    path = tmp_path / "arm_motions.json"
    srv.save_motions(path, _motions())
    cfg = {"port": "x", "id": "follower", "leader_port": "y", "leader_id": "leader", "motions": path}

    assert srv.record(cfg, "wave", seconds=60) == 1
    assert robot.bus.pos == pytest.approx(REST)
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]  # at rest, motors off
    assert srv.load_motions(path) == _motions()  # no half-recorded wave


# --------------------------------------------------------------------------- #
# Arm tasks: hover, grasp, place (a fake IK; placo isn't in this venv)          #
# --------------------------------------------------------------------------- #

ARM = srv.ARM_JOINTS
READY = {"shoulder_pan": 0.0, "shoulder_lift": -40.0, "elbow_flex": 30.0, "wrist_flex": 80.0,
         "wrist_roll": 0.0, "gripper": 5.0}
HOVER = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 70.0, "wrist_roll": 0.0}
PLACE_HOVER = {**HOVER, "shoulder_pan": -10.0}
DOWN = [-20.0 + 10.0 * k / 12 for k in range(13)]  # shoulder_lift on the way down: jaws 6 cm -> 4 mm
OPEN, CLOSED = 40.0, 2.0
SPOT = {"x": 0.30, "y": 0.20, "label": "eraser"}


class FakeReach:
    """Stands in for the IK: plans are straight joint-space lines in which shoulder_lift
    sets the jaw tips' height (6 cm at -20 down to 4 mm at -10). Records each plan asked for."""

    mount = {"x": 0.05, "y": 0.22, "z": 0.0, "yaw": 0.0}

    def __init__(self, refuse=None):
        self.refuse = refuse
        self.calls = []

    @staticmethod
    def tip_z(q):
        return 0.06 - (q["shoulder_lift"] + 20.0) * 0.0056

    def points_b(self, q):
        z = self.tip_z(q) - 0.0024
        return {"gripper": (0.20, 0.0, z + 0.0075), "jaw tips": (0.20, 0.0, z)}

    def points(self, q):
        return {name: srv.to_table(self.mount, p) for name, p in self.points_b(q).items()}

    def _line(self, start, lifts):
        path = [{**start, "shoulder_lift": v} for v in lifts]
        return path, [self.tip_z(q) for q in path]

    def _asked(self, *call):
        self.calls.append(call)
        if self.refuse:
            raise srv.Refused(self.refuse)

    def plan_grasp(self, x, y, yaw, current, ready):
        self._asked("grasp", x, y, yaw)
        down, zs = self._line(HOVER, DOWN)
        return {"pitch": 90.0, "hover": dict(HOVER), "down": down, "down_z": zs}

    def plan_place(self, x, y, height, current, ready):
        self._asked("place", x, y, height)
        down, zs = self._line(PLACE_HOVER, DOWN[:7])
        return {"pitch": 90.0, "up": [current], "up_z": [self.tip_z(current)], "hover": dict(PLACE_HOVER),
                "down": down, "down_z": zs}

    def plan_hover(self, x, y, current, ready, height=None):
        self._asked("hover", x, y, height)
        return {"pitch": 75.0, "hover": dict(HOVER)}

    def pose_warnings(self, poses):
        return []


def _task_motions(**gestures):
    motions = _motions(**gestures)
    motions["poses"].update(ready=dict(READY), gripper_open={**REST, "gripper": OPEN},
                            gripper_closed={**REST, "gripper": CLOSED})
    return motions


def _task_arm(reach=None, **bus):
    """An arm at rest with arm tasks on; by default an object 15 units wide sits where it grasps."""
    bus.setdefault("block", {"gripper": (15.0, 100.0)})
    robot = FakeRobot(REST, **bus)
    return srv.Arm(robot, _task_motions(), reach or FakeReach()), robot


def _at(goals, pose):
    return all(j in goals and goals[j] == pytest.approx(pose[j]) for j in ARM)


def _first(sent, test, after=-1):
    return next(i for i, (goals, _) in enumerate(sent) if i > after and test(goals))


def _start(arm):
    """A Controller running *arm* on its worker thread, and the list of its replies."""
    replies = []
    c = srv.Controller(arm, replies.append)
    threading.Thread(target=c.work, daemon=True).start()
    return c, replies


def test_to_base_applies_the_mount():
    mount = {"x": 0.10, "y": 0.20, "z": 0.0, "yaw": 0.0}
    assert srv.to_base(mount, (0.30, 0.20, 0.0)) == pytest.approx((0.2388, 0.0, -0.0024))
    turned = {**mount, "yaw": 90.0}  # the arm faces Reachy's left
    assert srv.to_base(turned, (0.10, 0.40, 0.0)) == pytest.approx((0.2388, 0.0, -0.0024))
    assert srv.to_base(turned, (0.30, 0.20, 0.0)) == pytest.approx((0.0388, -0.20, -0.0024))  # on its right
    odd = {"x": 0.05, "y": 0.22, "z": 0.01, "yaw": -35.0}
    for p in [(0.30, 0.10, 0.0), (-0.10, 0.40, 0.12)]:
        assert srv.to_table(odd, srv.to_base(odd, p)) == pytest.approx(p)


def test_reach_settings_from_env():
    s = srv.reach_settings({"ARM_BASE_X_CM": "5", "ARM_BASE_Y_CM": "22", "ARM_BASE_YAW_DEG": "-10",
                            "ARM_JOINT_OFFSETS_DEG": "1, 0, 0, 0, -2"})
    assert s["mount"] == pytest.approx({"x": 0.05, "y": 0.22, "z": 0.0, "yaw": -10.0})
    assert (s["keepout_m"], s["roll_limit_deg"], s["offsets"]) == (0.16, 90.0, (1.0, 0.0, 0.0, 0.0, -2.0))
    assert s["urdf"] == srv.REPO / "third_party" / "so101" / "so101_new_calib.urdf"
    assert srv.reach_settings({"ARM_BASE_X_CM": "5", "ARM_BASE_Y_CM": "22"})["mount"] is None  # no yaw: off
    with pytest.raises(ValueError, match="ARM_JOINT_OFFSETS_DEG"):
        srv.reach_settings({"ARM_JOINT_OFFSETS_DEG": "1,2"})
    with pytest.raises(ValueError, match="ARM_KEEPOUT_CM"):
        srv.reach_settings({"ARM_KEEPOUT_CM": "far"})


def test_load_reach_says_why_arm_tasks_are_off(tmp_path, capsys):
    mount = {"ARM_BASE_X_CM": "5", "ARM_BASE_Y_CM": "22", "ARM_BASE_YAW_DEG": "0"}
    no_poses = {"poses": {}}
    assert srv._load_reach({"reach": srv.reach_settings({})}, no_poses) is None
    assert "set ARM_BASE_X_CM" in capsys.readouterr().err
    missing = srv.reach_settings({**mount, "ARM_URDF": str(tmp_path / "none.urdf")})
    assert srv._load_reach({"reach": missing}, no_poses) is None
    assert "couldn't load the IK" in capsys.readouterr().err
    assert srv._load_reach({"reach": None, "reach_error": "ARM_KEEPOUT_CM should be a number"}, no_poses) is None
    assert "ARM_KEEPOUT_CM" in capsys.readouterr().err


def test_arm_tasks_are_listed_only_with_the_ik_the_mount_and_the_poses_they_use():
    plain = ["rest", "gripper_open", "gripper_close", "stop"]
    assert srv.Arm(FakeRobot(REST), _task_motions()).actions() == plain  # no IK
    unmounted = FakeReach()
    unmounted.mount = None
    assert srv.Arm(FakeRobot(REST), _task_motions(), unmounted).actions() == plain
    assert srv.Arm(FakeRobot(REST), _motions(), FakeReach()).actions() == plain  # ready not taught
    no_grip = _task_motions()
    del no_grip["poses"]["gripper_closed"]  # grasp would refuse, so it isn't offered
    assert srv.Arm(FakeRobot(REST), no_grip, FakeReach()).actions() == ["rest", "gripper_open", "stop"]
    assert srv.Arm(FakeRobot(REST), _task_motions(), FakeReach()).actions() == [
        "rest", "gripper_open", "gripper_close", "hover", "grasp", "place", "stop"]


def test_serve_loads_the_ik_before_connecting_and_lists_arm_tasks(monkeypatch):
    order, out, robot = [], [], FakeRobot(REST)
    monkeypatch.setattr(srv, "_load_reach", lambda cfg, motions: order.append("ik") or FakeReach())
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: order.append("connect") or robot)
    monkeypatch.setattr(srv, "load_motions", lambda path: _task_motions())
    monkeypatch.setattr(srv.sys, "stdin", io.StringIO(""))

    class Out:
        def write(self, s):
            out.append(json.loads(s))

        def flush(self):
            pass

    assert srv.serve({"port": "x", "id": "follower", "motions": None}, out=Out()) == 0
    assert order == ["ik", "connect"]  # no arm needed for the IK; a slow load can't stall a moving arm
    assert out[0]["actions"] == ["rest", "gripper_open", "gripper_close", "hover", "grasp", "place", "stop"]


def test_the_json_channel_keeps_fd_1_noise_off_the_protocol():
    # placo prints from C++ straight to fd 1, past sys.stdout: only protocol lines may reach the app.
    code = (
        "import importlib.util, os\n"
        f"spec = importlib.util.spec_from_file_location('arm_server', {srv.__file__!r})\n"
        "srv = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(srv)\n"
        "out = srv._json_channel()\n"
        "os.write(1, b'WARNING: base_link collides with shoulder_link\\n')\n"
        "print('a library print')\n"
        "out.write('{\"ready\": true}\\n')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.stdout == '{"ready": true}\n'
    assert "collides with" in done.stderr and "a library print" in done.stderr


def test_args_reach_the_arm_through_the_queue_and_status_says_what_it_holds():
    arm, robot = _task_arm()
    c, replies = _start(arm)
    c.handle({"id": 1, "action": "grasp", "args": {**SPOT, "yaw_deg": 95}})
    _wait_for(replies, 1)
    assert replies[0] == {"id": 1, "ok": True, "holding": True}
    assert arm.reach.calls == [("grasp", 0.30, 0.20, 95.0)]
    c.handle({"id": 2, "action": "status"})
    _wait_for(replies, 2)
    assert replies[1]["holding"] == "eraser"
    c.handle({"id": 3, "action": "place", "args": ["not", "an", "object"]})
    _wait_for(replies, 3)
    assert replies[2] == {"id": 3, "ok": False, "executed": False, "error": "args must be a JSON object"}


def test_stop_cancels_a_queued_arm_task():
    frames = [[i * 0.03, {**REST, "shoulder_pan": float(i)}] for i in range(200)]
    robot, started, release = _blocking_robot()
    arm = srv.Arm(robot, _task_motions(wave={"fps": 30, "frames": frames}), FakeReach())
    c, replies = _start(arm)
    c.handle({"id": 1, "action": "wave"})
    assert started.wait(2)
    c.handle({"id": 2, "action": "grasp", "args": SPOT})
    c.handle({"id": 3, "action": "stop"})
    release.set()
    _wait_for(replies, 3)
    assert replies[0] == {"id": 2, "ok": False, "error": "cancelled: stop", "cancelled": True}
    assert arm.reach.calls == []  # never planned, never moved


@pytest.mark.parametrize("action, args, reach, why", [
    ("grasp", SPOT, FakeReach("out of reach: 33 cm from the arm's base (max 31)"), "out of reach"),
    ("grasp", {"x": 0.3}, FakeReach(), "missing argument: y"),
    ("grasp", {"x": "far", "y": 0.2}, FakeReach(), "bad argument: x"),
    ("place", {"x": 0.3, "y": 0.2, "height": 0.02}, FakeReach(), "not holding anything"),
    ("hover", {"x": 0.3, "y": 0.2}, None, "arm tasks are off"),
    ("hover", {"x": 0.3, "y": 0.2, "height": 0.5}, FakeReach(), "release height 50.0 cm is out of range"),
])
def test_a_refused_task_moves_nothing(action, args, reach, why):
    robot = FakeRobot(REST)
    arm = srv.Arm(robot, _task_motions(), reach)
    with pytest.raises(srv.Refused, match=why):
        arm.run(action, dict(args))
    assert robot.bus.events == [] and robot.actions == 0  # zero bus writes


def test_a_refusal_replies_executed_false():
    arm = srv.Arm(FakeRobot(REST), _motions(), FakeReach())  # ready isn't taught
    arm.reach.mount = dict(FakeReach.mount)
    c, replies = _start(arm)
    c.handle({"id": 4, "action": "grasp", "args": SPOT})
    _wait_for(replies, 1)
    assert replies == [{"id": 4, "ok": False, "executed": False,
                        "error": "pose 'ready' isn't taught yet; run: scripts/arm.sh teach ready"}]
    assert arm.robot.bus.events == []


def test_grasp_opens_goes_ready_hover_down_closes_and_lifts_ending_held_with_torque_on():
    arm, robot = _task_arm()
    assert arm.run("grasp", dict(SPOT)) == {"holding": True}
    sent = robot.sent
    opened = _first(sent, lambda g: g.get("gripper") == pytest.approx(OPEN))
    ready = _first(sent, lambda g: _at(g, READY))
    hover = _first(sent, lambda g: _at(g, HOVER))
    bottom = _first(sent, lambda g: g.get("shoulder_lift") == pytest.approx(DOWN[-1]))
    close = _first(sent, lambda g: set(g) == {"gripper"}, after=bottom)
    lifted = max(i for i, (g, _) in enumerate(sent) if _at(g, HOVER))
    assert opened < ready < hover < bottom < close < lifted
    assert arm.holding == {"x": 0.30, "y": 0.20, "yaw": None, "pitch": 90.0, "label": "eraser"}
    assert not [e for e in robot.bus.events if e[0] == "off"] and arm.powered == set(REST)
    # Compliant close: the jaws stopped at 15, so it holds 1.5 past that (load 60: in 40..120).
    assert robot.bus.goal["gripper"] == pytest.approx(13.5) and robot.bus.pos["gripper"] == 15.0
    assert robot.bus.pos["shoulder_lift"] == pytest.approx(HOVER["shoulder_lift"])  # lifted back up


def test_compliant_close_eases_a_squeeze_that_loads_the_gripper_too_much():
    arm, robot = _task_arm(load={"gripper": 80})  # 80 + 40 per unit of squeeze
    arm.run("grasp", dict(SPOT))
    assert robot.bus.goal["gripper"] == pytest.approx(14.0)  # 1.5 units read 140: eased by 0.5 to 120


def test_grasp_with_nothing_between_the_jaws_opens_and_rests():
    arm, robot = _task_arm(block={})
    assert arm.run("grasp", dict(SPOT)) == {"holding": False}
    assert arm.holding is None
    closed_at = _first(robot.sent, lambda g: g.get("gripper") == pytest.approx(CLOSED))
    reopened = _first(robot.sent, lambda g: g.get("gripper") == pytest.approx(OPEN), after=closed_at)
    assert reopened > closed_at and robot.bus.events[-6:] == ALL_OFF  # opened, then parked, motors off


def test_a_contact_on_the_way_down_backs_off_and_closes_there():
    # The table (or the object) stops shoulder_lift at -15: 3.2 cm instead of 4 mm.
    arm, robot = _task_arm(block={"gripper": (15.0, 100.0), "shoulder_lift": (-200.0, -15.0)})
    assert arm.run("grasp", dict(SPOT)) == {"holding": True}
    lifts = [g["shoulder_lift"] for g, _ in robot.sent if "shoulder_lift" in g]
    assert max(lifts) < -11  # stopped well short of the bottom (-10)
    hover = _first(robot.sent, lambda g: _at(g, HOVER))
    close = _first(robot.sent, lambda g: set(g) == {"gripper"}, after=hover)
    backed_off = robot.sent[close][1]["shoulder_lift"]
    assert backed_off == pytest.approx(-20 + (0.06 - (0.032 + 0.003)) / 0.0056, abs=0.01)  # 3 mm up


def test_hover_goes_ready_over_the_spot_and_back_to_rest():
    arm, robot = _task_arm(block={})
    arm.run("hover", {"x": 0.3, "y": 0.2})
    ready = _first(robot.sent, lambda g: _at(g, READY))
    hover = _first(robot.sent, lambda g: _at(g, HOVER))
    back = _first(robot.sent, lambda g: _at(g, READY), after=hover)
    assert ready < hover < back and robot.bus.events[-6:] == ALL_OFF
    assert arm.reach.calls == [("hover", 0.3, 0.2, None)]
    arm.run("hover", {"x": 0.3, "y": 0.2, "height": 0.12})  # where a place into a cup would start down
    assert arm.reach.calls[-1] == ("hover", 0.3, 0.2, 0.12)


def test_a_hover_for_a_place_goes_as_high_as_that_place_starts_down():
    # Reach's plan without placo: stub the IK and the checks, keep the plan's own logic.
    reach = srv.Reach.__new__(srv.Reach)
    reach.mount, reach.keepout = dict(FakeReach.mount), 0.16
    solved, floors = [], []
    reach.seed = lambda p, current: dict(HOVER)
    reach.solve = lambda p, pitch, yaw, seed, centre=0.0: solved.append(p[2]) or dict(HOVER)
    reach.check = lambda path, contact=False, floor=0.0: floors.append(floor)
    reach.tip_z = lambda q: 0.20  # ready holds the jaw tips 20 cm up
    ready = {j: READY[j] for j in ARM}
    for height in (None, 0.02, 0.12):  # a test hover; a place on the table; into a 9 cm cup
        assert reach.plan_hover(0.23, 0.20, ready, ready, height)["pitch"] == 90.0
    # The gripper origin 10 cm up, unless the place starts down higher (jaw tips 3 cm over
    # its release, 7.5 mm below the origin); on the way there the jaw tips stay that high.
    assert solved == pytest.approx([0.10, 0.10, 0.12 + 0.03 + 0.0075])
    assert floors == pytest.approx([0.0, 0.045, 0.145])


def test_place_goes_over_the_spot_lets_go_and_rests():
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    robot.bus.block.pop("gripper")  # the object leaves the jaws once they open
    n = len(robot.sent)
    arm.run("place", {"x": 0.25, "y": 0.30, "height": 0.02})
    sent = robot.sent[n:]
    over = _first(sent, lambda g: _at(g, PLACE_HOVER))
    down = _first(sent, lambda g: g.get("shoulder_lift") == pytest.approx(DOWN[6]))
    let_go = _first(sent, lambda g: "gripper" in g)  # the gripper kept its squeeze till then
    ready = _first(sent, lambda g: _at(g, READY))
    assert over < down < let_go < ready
    assert robot.bus.goal["gripper"] == pytest.approx(REST["gripper"]) and arm.holding is None
    assert any(g.get("gripper") == pytest.approx(OPEN) for g, _ in sent[let_go:ready])
    assert robot.bus.events[-6:] == ALL_OFF
    assert arm.reach.calls[-1] == ("place", 0.25, 0.30, 0.02)


def test_place_lets_go_where_it_touches_down_early():
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    robot.bus.block = {"shoulder_lift": (-200.0, -19.0)}  # something under it; the object leaves the open jaws
    n = len(robot.sent)
    arm.run("place", {"x": 0.25, "y": 0.30, "height": 0.02})
    sent = robot.sent[n:]
    let_go = _first(sent, lambda g: set(g) == {"gripper"})
    assert sent[let_go][1]["shoulder_lift"] == pytest.approx(-19.0)  # opened where it touched down
    assert max(g["shoulder_lift"] for g, _ in sent if "shoulder_lift" in g) < DOWN[6]  # never pushed on
    assert arm.holding is None and robot.bus.events[-6:] == ALL_OFF


@pytest.mark.parametrize("action", ["rest", "wave", "gripper_open", "gripper_close", "grasp", "hover"])
def test_while_holding_moves_that_would_drop_it_are_refused(action):
    frames = [[0.0, dict(REST)], [0.1, dict(REST)]]
    robot = FakeRobot(REST, block={"gripper": (15.0, 100.0)})
    arm = srv.Arm(robot, _task_motions(wave={"fps": 30, "frames": frames}), FakeReach())
    arm.run("grasp", dict(SPOT))
    events = list(robot.bus.events)
    with pytest.raises(srv.Refused, match="holding the eraser: put it down first"):
        arm.run(action, dict(SPOT))
    assert robot.bus.events == events


def test_stop_while_holding_freezes_only_the_arm_joints():
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    c, replies = _start(arm)
    c.handle({"id": 1, "action": "stop"})
    _wait_for(replies, 1)
    assert robot.bus.events[-2:] == [("goal", {j: robot.bus.pos[j] for j in ARM}), ("on", list(ARM))]
    assert robot.bus.goal["gripper"] == pytest.approx(13.5)  # still squeezing


def test_a_fault_while_holding_keeps_the_grip_unless_the_gripper_tripped(monkeypatch):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    c, replies = _start(arm)
    robot.bus.status["shoulder_lift"] = 0x20
    _wait_for(replies, 1)
    assert replies[0]["event"] == "fault" and robot.bus.events[-1] == ("goal", {j: robot.bus.pos[j] for j in ARM})
    robot.bus.status["shoulder_lift"] = 0
    threading.Event().wait(0.05)
    robot.bus.status["gripper"] = 0x20  # the gripper itself: stop it pushing too
    _wait_for(replies, 2)
    assert "gripper" in robot.bus.events[-1][1]


def test_shutdown_while_holding_puts_it_back_then_rests():
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    robot.bus.block.pop("gripper")  # the object leaves the jaws once they open
    c, replies = _start(arm)
    assert c.handle({"id": 7, "action": "shutdown"}) is False
    _wait_for(replies, 1)
    assert replies == [{"id": 7, "ok": True}]
    assert arm.reach.calls[-1] == ("place", 0.30, 0.20, 0.01)  # back where it was picked up
    assert arm.holding is None and robot.bus.goal["gripper"] == pytest.approx(REST["gripper"])
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]


def test_shutdown_after_an_aborted_close_opens_the_gripper_then_rests_via_ready():
    arm, robot = _task_arm(block={})
    arm.hold()
    arm.move_to(HOVER)
    arm._out, arm._gripping = True, True  # stopped mid-close, before any contact
    arm.shutdown()
    opened = _first(robot.sent, lambda g: g.get("gripper") == pytest.approx(OPEN))
    ready = _first(robot.sent, lambda g: _at(g, READY), after=opened)
    assert opened < ready and robot.bus.events[-6:] == ALL_OFF and not arm._gripping


def test_a_fault_mid_close_takes_the_existing_fault_path(monkeypatch):
    monkeypatch.setattr(srv, "_FAULT_POLL_S", 0.0)

    def overload(robot):
        goals, _ = robot.sent[-1] if robot.sent else ({}, None)
        if set(goals) == {"gripper"} and goals["gripper"] < 30:
            robot.bus.status["gripper"] = 0x20

    arm, robot = _task_arm(block={})
    robot.on_action = overload
    c, replies = _start(arm)
    c.handle({"id": 1, "action": "grasp", "args": SPOT})
    _wait_for(replies, 2)
    grasp, event = replies
    assert not grasp["ok"] and grasp["faults"] == {"gripper": ["overload"]}
    assert event["event"] == "fault" and event["interrupted"]
    assert robot.bus.events[-1] == ("goal", dict(robot.bus.pos))  # every motor frozen where it was


@pytest.mark.parametrize("jaws", ["at their goal", "shut"])
def test_a_dropped_object_is_announced_and_a_hard_squeeze_eased(monkeypatch, jaws):
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    c, replies = _start(arm)
    robot.bus.load["gripper"] = 250  # squeezing too hard
    for _ in range(100):
        if robot.bus.goal["gripper"] > 13.5:
            break
        threading.Event().wait(0.01)
    assert robot.bus.goal["gripper"] > 13.5  # eased toward open
    robot.bus.load["gripper"] = 0
    robot.bus.block.pop("gripper")
    # It slipped out: the jaws close to their hold goal and stop there, unloaded (a real
    # object stops them far from the closed pose) — or, for a thin one, shut.
    robot.bus.pos["gripper"] = robot.bus.goal["gripper"] if jaws == "at their goal" else CLOSED
    _wait_for(replies, 1)
    assert replies == [{"event": "dropped", "arm": "arm", "label": "eraser"}]
    assert arm.holding is None


def test_a_drop_takes_slack_jaws_on_two_checks_in_a_row():
    arm, robot = _task_arm()
    arm.run("grasp", dict(SPOT))
    assert arm.check_grip() is None and arm.holding  # held: jaws at 15, goal 13.5, load 60
    robot.bus.block.pop("gripper")
    robot.bus.pos["gripper"] = robot.bus.goal["gripper"]  # slack at the hold goal ...
    assert arm.check_grip() is None
    robot.bus.pos["gripper"] = 15.0  # ... but back on the object: one odd reading
    assert arm.check_grip() is None
    robot.bus.pos["gripper"] = robot.bus.goal["gripper"]
    assert arm.check_grip() is None and arm.holding
    assert arm.check_grip() == "eraser" and arm.holding is None


@pytest.mark.parametrize("action, args", [("wave", None), ("hover", {"x": 0.3, "y": 0.2})])
def test_after_a_grasp_stopped_mid_close_a_gesture_or_hover_lets_go_and_parks_first(action, args):
    # Every frame of the wave closes the gripper fully: on the eraser, that's a full-force squeeze.
    frames = [[0.0, {**REST, "shoulder_pan": 20.0, "gripper": 0.0}],
              [0.1, {**REST, "shoulder_pan": -20.0, "gripper": 0.0}]]
    robot = FakeRobot(REST, block={"gripper": (15.0, 100.0)})
    arm = srv.Arm(robot, _task_motions(wave={"fps": 30, "frames": frames}), FakeReach())

    def stop_mid_close(r):  # the jaws are on the eraser at the bottom, before a contact is sure
        goals = r.sent[-1][0] if r.sent else {}
        if set(goals) == {"gripper"} and goals["gripper"] < 14.0 and r.bus.pos["shoulder_lift"] > DOWN[-1] - 1:
            arm.abort.set()

    robot.on_action = stop_mid_close
    with pytest.raises(srv.Stopped):
        arm.run("grasp", dict(SPOT))
    assert arm.holding is None and arm._gripping and arm._out
    arm.abort.clear()  # as the Controller does before the next request

    def leave_it(r):  # once the jaws open, the arm leaves the eraser on the table
        if r.sent and r.sent[-1][0].get("gripper", 0.0) >= OPEN - 1e-6:
            r.bus.block.pop("gripper", None)

    robot.on_action = leave_it
    n = len(robot.sent)
    arm.run(action, args)
    sent = robot.sent[n:]
    opened = _first(sent, lambda g: g.get("gripper") == pytest.approx(OPEN))
    moved = _first(sent, lambda g: any(j in g for j in ARM))
    assert opened < moved  # let go before any arm joint moved
    assert all(g["gripper"] >= 13.0 for g, _ in sent[:opened] if "gripper" in g)  # never squeezed harder
    assert robot.bus.pos == pytest.approx(REST) and robot.bus.events[-6:] == ALL_OFF


def test_fk_only_reads_from_the_arm(monkeypatch, capsys):
    import types

    bus = FakeBus(REST)
    bus.is_calibrated = True
    bus.connect = lambda: bus.events.append(("connect",))
    read = bus.sync_read
    polls = []

    def sync_read(name, motors=None, num_retry=0, normalize=True):
        if name == "Present_Position" and motors is None:
            polls.append(1)
            if len(polls) > 2:
                raise KeyboardInterrupt  # Ctrl-C
        return read(name, motors, num_retry, normalize)

    bus.sync_read = sync_read

    class Follower:
        def __init__(self, config):
            self.bus = bus

        def connect(self):
            raise AssertionError("robot.connect() runs configure(), which writes to the servos")

    mod = types.ModuleType("lerobot.robots.so_follower")
    mod.SO101Follower, mod.SO101FollowerConfig = Follower, lambda **kw: kw
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod)
    monkeypatch.setattr(srv, "_cli_reach", lambda cfg, need_mount: FakeReach())
    assert srv.fk({"port": "x", "id": "follower"}, registers=True) == 0
    assert bus.events == [("connect",), ("disconnect", False)]  # nothing written, torque left alone
    out = capsys.readouterr().out
    assert "gripper Overload_Torque" in out and out.count("joints ") == 2 and "T gripper" in out


def _try_setup(monkeypatch, answer):
    robot = FakeRobot(REST, block={"gripper": (15.0, 100.0)})
    reach = FakeReach()
    monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
    monkeypatch.setattr(srv, "_cli_reach", lambda cfg, need_mount: reach)
    monkeypatch.setattr(srv, "load_motions", lambda path: _task_motions())
    monkeypatch.setattr("builtins.input", lambda prompt="": answer(prompt, robot))
    return robot, reach


def test_try_grasp_shows_the_plan_and_waits_before_moving_then_puts_it_back(monkeypatch, capsys):
    prompts = []

    def answer(prompt, robot):
        prompts.append((prompt, robot.actions))
        if "put it back" in prompt:
            robot.bus.block.pop("gripper")  # it leaves the jaws once they open
        return ""

    robot, reach = _try_setup(monkeypatch, answer)
    assert srv.try_task({"port": "x", "id": "follower", "motions": None}, "grasp", 23.0, 22.0, None, 2.0) == 0
    assert prompts[0] == ("Press Enter to start (Ctrl-C to cancel)... ", 0)  # nothing moved before Enter
    assert "Plan: grasp at a 90° approach" in capsys.readouterr().out
    assert reach.calls[-1] == ("place", 0.23, 0.22, 0.01)  # put back where it was
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]


def test_ctrl_c_during_try_lets_go_and_parks_via_ready(monkeypatch):
    def answer(prompt, robot):
        if "put it back" in prompt:
            robot.bus.block.pop("gripper")
            raise KeyboardInterrupt
        return ""

    robot, reach = _try_setup(monkeypatch, answer)
    assert srv.try_task({"port": "x", "id": "follower", "motions": None}, "grasp", 23.0, 22.0, None, 2.0) == 0
    holding = max(i for i, (g, _) in enumerate(robot.sent) if _at(g, HOVER))
    opened = _first(robot.sent, lambda g: g.get("gripper") == pytest.approx(OPEN), after=holding)
    ready = _first(robot.sent, lambda g: _at(g, READY), after=opened)
    assert holding < opened < ready
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]


def test_ik_check_needs_no_arm_port(monkeypatch):
    monkeypatch.setattr(srv, "load_config", lambda: {"port": "", "reach": None, "reach_error": None})
    monkeypatch.setattr(srv, "ik_check", lambda cfg: 7)
    assert srv.main(["ik-check"]) == 7
    assert srv.main(["fk"]) == 1  # this one reads the arm: it needs ARM_PORT


def test_real_ik_solves_and_round_trips():
    """Runs only where LeRobot's kinematics (placo) and the SO-101 URDF are installed."""
    pytest.importorskip("placo")
    pytest.importorskip("lerobot.model.kinematics")
    urdf = Path(os.environ.get("ARM_URDF") or srv.REPO / "third_party" / "so101" / "so101_new_calib.urdf")
    if not urdf.is_file():
        pytest.skip(f"no SO-101 URDF at {urdf}")
    mount = {"x": 0.05, "y": 0.22, "z": 0.0, "yaw": 0.0}
    reach = srv.Reach(urdf, mount)
    rest = {"shoulder_pan": -5.05, "shoulder_lift": -107.03, "elbow_flex": 96.44, "wrist_flex": 74.81, "wrist_roll": 7.16}
    p = (0.23, 0.22, 0.03)  # 18 cm ahead of the arm's pan axis, 3 cm up
    for pitch, yaw in ((90.0, None), (90.0, 45.0), (60.0, 30.0)):
        q = reach.solve(p, pitch, yaw, reach.seed(p, rest))
        assert q is not None, (pitch, yaw)
        assert math.dist(reach.points(q)["gripper"], p) < 0.002
        assert reach.pitch(q) == pytest.approx(pitch, abs=3)
        assert abs(q["wrist_roll"]) <= 90
    plan = reach.plan_grasp(0.23, 0.22, None, rest)
    bottom = reach.points(plan["down"][-1])
    assert bottom["jaw tips"][2] == pytest.approx(0.004, abs=0.002)
    assert math.dist(bottom["gripper"][:2], (0.23, 0.22)) == pytest.approx(0.02, abs=0.002)  # jaw centre on the spot
    with pytest.raises(srv.Refused, match="out of reach"):
        reach.plan_grasp(0.45, 0.22, None, rest)
    with pytest.raises(srv.Refused, match="too close to Reachy"):
        reach.plan_grasp(0.10, 0.05, None, rest)
