"""Arm tasks end to end, without an arm: the app's ArmTask and ArmClient drive the real arm
server (serve() and its Controller) over its JSON lines, on the fake bus and fake IK from
test_arm_server.py, so both halves are held to the same protocol."""

from __future__ import annotations

import math
import os
import subprocess
import threading
from unittest.mock import MagicMock

import pytest

from reachy_emotion import arm_client
from reachy_emotion import arm_server as srv
from reachy_emotion import arm_task as at
from reachy_emotion import live_conversation as lc
from tests.test_arm_server import ALL_OFF, REST, FakeBus, FakeReach, FakeRobot, _task_motions
from tests.test_arm_task import _ER, _call, _camera_mini, _er_point, _Head, _results, _thought, _turn

_MOUNT = {"x": 0.05, "y": 0.22, "yaw_deg": 0.0}  # FakeReach.mount, as the app reads it from .env
_ERASER = (0.23, 0.20)  # on the table, 18 cm in front of the arm
_SPOT = (0.25, 0.28)    # a free spot to put it down
_BOTTOM = -10.0         # FakeReach's shoulder_lift with the jaw tips 4 mm above the table


class _ObjectBus(FakeBus):
    """An eraser lies where the arm reaches down: jaws closing at the bottom of FakeReach's
    descent stop on it (at 15 %), and it stays between them until they open again."""

    def __init__(self, pos):
        super().__init__(pos)
        self.held = False

    def sync_write(self, name, values):
        grip = values.get("gripper")
        if grip is not None and grip >= 30.0:
            self.held = False
        elif grip is not None and grip < 15.0 and self.pos["shoulder_lift"] >= _BOTTOM - 0.5:
            self.held = True
        self.block = {"gripper": (15.0, 100.0)} if self.held else {}
        super().sync_write(name, values)


@pytest.fixture
def arm_server(monkeypatch):
    """start(reach) -> (client, robot, done): an ArmClient whose "child process" is
    arm_server.serve() on a thread, talking over pipes as the real one does over its
    stdin/stdout, with a fake arm (every task pose taught) and *reach* as its IK."""
    opened = []

    def start(reach):
        robot = FakeRobot(REST)
        robot.bus = _ObjectBus(REST)
        monkeypatch.setattr(srv.time, "sleep", lambda s: None)
        monkeypatch.setattr(srv, "connect_follower", lambda port, rid: robot)
        monkeypatch.setattr(srv, "_load_reach", lambda cfg, motions: reach)
        monkeypatch.setattr(srv, "load_motions", lambda path: _task_motions())
        requests_r, requests_w = os.pipe()
        replies_r, replies_w = os.pipe()
        server_in = os.fdopen(requests_r)
        monkeypatch.setattr(srv.sys, "stdin", server_in)
        done = threading.Event()

        def run():
            out = os.fdopen(replies_w, "w", buffering=1)
            try:
                srv.serve({"port": "fake", "id": "follower", "motions": None}, out=out)
            finally:
                out.close()  # the client sees the server exit
                done.set()

        class Child:
            """subprocess.Popen, as ArmClient.start() uses it."""

            def __init__(self, cmd, **kw):
                self.stdin = os.fdopen(requests_w, "w", buffering=1)
                self.stdout = os.fdopen(replies_r)
                opened.extend([self.stdin, self.stdout, server_in])
                threading.Thread(target=run, daemon=True).start()

            def poll(self):
                return 0 if done.is_set() else None

            def wait(self, timeout=None):
                if not done.wait(timeout):
                    raise subprocess.TimeoutExpired("arm_server.py", timeout)
                return 0

            def kill(self):
                self.stdin.close()

        monkeypatch.setattr(arm_client.subprocess, "Popen", Child)
        client = arm_client.ArmClient(["arm_server.py", "serve"], ready_timeout=10)
        client.start()
        clients.append(client)
        return client, robot, done

    clients: list = []
    yield start
    for client in clients:
        client.close(timeout=10)  # parks the arm if a test stopped early
    for f in opened:
        f.close()


def _view_pose():
    """The head pose of the task's first view (where ER points are taken from)."""
    head = _Head()
    head.turn(at.work_view_deg(_MOUNT), at._TILT_DEG[at._WORK_TILT])
    return head.pose


def _task(er, client, mode="on"):
    head = _Head()
    task = at.ArmTask(er, "er-test", client, _camera_mini(), head.turn, head.grab, _MOUNT, mode)
    return task.run("put the pink eraser over there", threading.Event())


def _grasp_call(pose, **extra):
    return _call("g", "grasp", point=_er_point((*_ERASER, 0.005), pose), frame=1, label="pink eraser", **extra)


def test_a_pick_and_place_runs_through_the_real_arm_server(arm_server):
    reach = FakeReach()
    client, robot, done = arm_server(reach)
    assert client.name == "follower" and {"hover", "grasp", "place"} <= set(client.actions)  # the app's gate
    pose = _view_pose()
    # The eraser lies at 30° to Reachy's forward: the jaws close across it, at 120° → -60°.
    a, b = math.radians(30), 0.02
    ends = [_er_point((_ERASER[0] + s * b * math.cos(a), _ERASER[1] + s * b * math.sin(a), 0.005), pose)
            for s in (-1, 1)]
    er = _ER(
        _turn(_thought(), _grasp_call(pose, ends=ends)),
        _turn(_call("p", "place", frame=2, target="the table", release="on_table",
                    point=_er_point((*_SPOT, 0.0), pose))),
        _turn(_call("f", "finish", outcome="done", say="I put the pink eraser down.")),
    )
    result = _task(er, client)
    assert result == {"outcome": "done", "say": "I put the pink eraser down.", "holding": None,
                      "did": ["picked up the pink eraser", "put the pink eraser down on the table"]}
    # What the server got: T metres, the jaw yaw in T degrees, the release height in metres.
    (grasp, gx, gy, yaw), (place, px, py, height) = reach.calls
    assert (grasp, place) == ("grasp", "place")
    assert (gx, gy) == (pytest.approx(_ERASER[0], abs=0.003), pytest.approx(_ERASER[1], abs=0.003))
    assert yaw == pytest.approx(-60.0, abs=2.0)
    assert (px, py, height) == (pytest.approx(_SPOT[0], abs=0.003), pytest.approx(_SPOT[1], abs=0.003), 0.02)
    # What ER heard: the server's holding, as the reply and as the status in the next caption.
    (_, res, content), = _results(er, 1)
    assert res["ok"] is True and res["executed"] is True and res["holding"] is True
    assert content[2]["text"].endswith("Holding: pink eraser.")
    (_, res, content), = _results(er, 2)
    assert res["ok"] is True and res["holding"] is False and content[2]["text"].endswith("Holding: nothing.")
    assert client.call("status", timeout=5)["holding"] is None
    assert robot.bus.events[-6:] == ALL_OFF  # parked at rest, motors off
    client.close()
    assert done.is_set() and robot.bus.events[-1] == ("disconnect", False)


def test_holding_blocks_other_moves_stop_keeps_the_grip_and_the_session_end_puts_it_back(arm_server):
    reach = FakeReach()
    client, robot, done = arm_server(reach)
    er = _ER(
        _turn(_grasp_call(_view_pose())),
        _turn(_call("f", "finish", outcome="done", say="I picked up the pink eraser.")),
    )
    result = _task(er, client)
    assert result["outcome"] == "done" and result["holding"] == "pink eraser"
    assert reach.calls[0][3] is None  # no ends: no yaw_deg in the request
    assert client.call("status", timeout=5)["holding"] == "pink eraser"
    # move_arm while it holds the eraser: refused before anything moves.
    events = len(robot.bus.events)
    assert client.call("rest", timeout=5) == {
        "ok": False, "executed": False, "error": "holding the pink eraser: put it down first",
    }
    assert len(robot.bus.events) == events
    # move_arm(stop): only the arm joints freeze; the gripper keeps its squeeze.
    assert client.call("stop", timeout=5) == {"ok": True}
    assert robot.bus.events[-1] == ("on", list(srv.ARM_JOINTS))
    assert robot.bus.goal["gripper"] == pytest.approx(13.5) and robot.bus.held
    # The session ends: close() → shutdown puts it back where it was picked up, then rests.
    client.close()
    assert done.is_set()
    assert reach.calls[-1] == ("place", reach.calls[0][1], reach.calls[0][2], srv._PUT_BACK_Z)
    assert not robot.bus.held and robot.bus.events[-7:] == [*ALL_OFF, ("disconnect", False)]


def test_a_refusal_reaches_er_as_not_executed_and_hover_mode_only_hovers(arm_server):
    reach = FakeReach(refuse="out of reach: 35 cm from the arm's base (max 31)")
    client, robot, _ = arm_server(reach)
    sent = len(robot.sent)
    er = _ER(
        _turn(_grasp_call(_view_pose())),
        _turn(_call("f", "finish", outcome="infeasible", say="It's out of my arm's reach.")),
    )
    result = _task(er, client)
    (_, res, _), = _results(er, 1)
    assert res == {"ok": False, "executed": False, "error": "out of reach: 35 cm from the arm's base (max 31)"}
    assert len(robot.sent) == sent  # nothing moved
    assert result["outcome"] == "infeasible" and result["holding"] is None

    reach.refuse = None
    er = _ER(_turn(_grasp_call(_view_pose())), _turn(_call("f", "finish", outcome="done", say="Hovered.")))
    result = _task(er, client, mode="hover")
    assert reach.calls[-1] == ("hover", *reach.calls[0][1:3], None)  # a grasp's hover: no height
    (_, res, _), = _results(er, 1)
    assert res["note"] == "test mode: hovered only" and res["holding"] is False
    assert robot.bus.events[-6:] == ALL_OFF and not robot.bus.held


def test_a_fault_while_er_thinks_ends_the_task_before_its_next_move(arm_server, monkeypatch):
    # The arm holds the eraser at the hover; while ER plans the place, a shoulder overload
    # trips. The server freezes the arm and announces it; the task must not send the place.
    monkeypatch.setattr(srv, "_IDLE_POLL_S", 0.01)
    reach = FakeReach()
    client, robot, _ = arm_server(reach)
    tasks = lc._ArmTasks(client, "on", _MOUNT)
    tasks.busy.set()  # as the receive loop marks it
    client.on_event = lc._arm_event_handler(MagicMock(), MagicMock(), set(), None, tasks)
    pose = _view_pose()

    class _ThinkingER(_ER):
        def create(self, **kw):
            if len(self.calls) == 1:  # planning the place
                robot.bus.status["shoulder_lift"] = 0x20
                assert tasks.stop.wait(5)  # the fault event reached the task
                robot.bus.status["shoulder_lift"] = 0
            return super().create(**kw)

    er = _ThinkingER(
        _turn(_grasp_call(pose)),
        _turn(_call("p", "place", frame=2, target="the table", release="on_table", point=_er_point((*_SPOT, 0.0), pose))),
        _turn(_call("f", "finish", outcome="done", say="I put the pink eraser down.")),
    )
    head = _Head()
    result = at.ArmTask(er, "er-test", client, _camera_mini(), head.turn, head.grab, _MOUNT, "on").run(
        "put the pink eraser over there", tasks.stop
    )
    assert [c[0] for c in reach.calls] == ["grasp"]  # the place was never sent
    assert result["outcome"] == "stopped" and result["holding"] == "pink eraser"
    assert len(er.calls) == 2
