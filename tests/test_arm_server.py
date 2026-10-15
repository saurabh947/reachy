"""Tests for the SO-101 arm server logic (arm_server.py) with a fake LeRobot bus — no arm."""

from __future__ import annotations

import json
import threading
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
    servo's Feetech status bits (0x20 = overload)."""

    def __init__(self, pos, torque=0, stuck=()):
        self.pos = dict(pos)
        self.torque = torque
        self.stuck = set(stuck)  # joints that can't move (blocked)
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
        return {m: v for m, v in self.pos.items() if motors is None or m in motors}

    def sync_write(self, name, values):
        self.events.append(("goal", dict(values)))
        self.pos.update({m: v for m, v in values.items() if m not in self.stuck})

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

    def configure(self):
        # Like LeRobot: torque off while writing settings, then back on.
        self.configured += 1
        self.bus.disable_torque()
        self.bus.enable_torque()

    def send_action(self, action):
        self.actions += 1
        if self.on_action:
            self.on_action(self)
        self.bus.sync_write("Goal_Position", {k.removesuffix(".pos"): v for k, v in action.items()})
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
    assert replies == [{"id": srv._QUEUE_MAX + 1, "ok": False,
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
    monkeypatch.setattr(srv.sys, "stdout", Out())
    assert srv.serve({"port": "x", "id": "follower", "motions": None}) == 0
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

    monkeypatch.setattr(srv.sys, "stdout", Out())
    assert srv.serve({"port": "x", "id": "follower", "motions": None}) == 1
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
    import io

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
    monkeypatch.setattr(srv.sys, "stdout", DeadPipe())
    assert srv.serve({"port": "x", "id": "follower", "motions": None}) == 0
    assert robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]  # parked at rest
    assert robot.bus.pos == pytest.approx(REST)


def test_ctrl_c_while_recording_parks_the_follower_and_saves_nothing(monkeypatch, tmp_path):
    import sys
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
