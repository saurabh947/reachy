"""Tests for the SO-101 arm server logic (arm_server.py) with a fake LeRobot bus — no arm."""

from __future__ import annotations

import threading

import pytest

from reachy_emotion import arm_server as srv

REST = {"shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
        "wrist_flex": 60.0, "wrist_roll": 0.0, "gripper": 5.0}


class FakeBus:
    """Joints follow their goals instantly; records the order of writes."""

    def __init__(self, pos, torque=0, stuck=()):
        self.pos = dict(pos)
        self.torque = torque
        self.stuck = set(stuck)  # joints that can't move (blocked)
        self.events = []

    def sync_read(self, name, motors=None, num_retry=0):
        if name == "Torque_Enable":
            return {m: self.torque for m in self.pos}
        if isinstance(motors, str):
            motors = [motors]
        return {m: v for m, v in self.pos.items() if motors is None or m in motors}

    def sync_write(self, name, values):
        self.events.append(("goal", dict(values)))
        self.pos.update({m: v for m, v in values.items() if m not in self.stuck})

    def enable_torque(self, motors=None):
        self.events.append(("on", motors))

    def disable_torque(self, motors=None):
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
    assert robot.bus.events[-1] == ("off", None)


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
    assert robot.bus.events[-1] == ("off", None)
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


def test_stop_interrupts_a_gesture_and_freezes_in_place():
    frames = [[i * 0.03, {**REST, "shoulder_pan": float(i)}] for i in range(200)]
    started = threading.Event()
    release = threading.Event()

    def on_action(robot):
        if robot.actions == 30:
            started.set()
            release.wait(2)

    robot = FakeRobot(REST, on_action)
    c, replies = _controller(robot, _motions(wave={"fps": 30, "frames": frames}))
    c.handle({"id": 1, "action": "wave"})
    assert started.wait(2)
    c.handle({"id": 2, "action": "wave"})  # one move at a time
    c.handle({"id": 3, "action": "stop"})
    release.set()
    _wait_for(replies, 3)
    assert {"id": 2, "ok": False, "error": "busy with another move; say stop first"} in replies
    assert {"id": 1, "ok": False, "error": "stopped"} in replies
    assert replies[-1] == {"id": 3, "ok": True}
    assert robot.bus.events[-1] == ("on", None)  # frozen: motors on where it stopped
    assert robot.bus.pos["shoulder_pan"] < 199  # it never finished the gesture


def test_shutdown_parks_at_rest_and_releases_the_port():
    robot = FakeRobot({**REST, "shoulder_lift": 0.0})
    c, replies = _controller(robot, _motions())
    assert c.handle({"id": 7, "action": "shutdown"}) is False
    _wait_for(replies, 1)
    assert replies == [{"id": 7, "ok": True}]
    assert robot.bus.pos == pytest.approx(REST)
    assert robot.bus.events[-2:] == [("off", None), ("disconnect", False)]


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
    with pytest.raises(RuntimeError, match="didn't reach rest"):
        srv.Arm(robot, _motions()).rest()
    assert ("off", None) not in robot.bus.events


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
    assert robot.bus.events[-2:] == [("off", None), ("disconnect", False)]  # parked at rest
    assert robot.bus.pos == pytest.approx(REST)
