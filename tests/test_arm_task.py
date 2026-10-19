"""Tests for arm tasks (arm_task.py): the ER 2 function-calling loop, with a scripted ER
client, a fake arm and a fake head — no network, no robot."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_emotion import arm_task as at

# Reachy Mini's head→camera transform (from the SDK) and a 640x480 camera, no distortion.
_T_HEAD_CAM = np.array([[0, 0, 1, 0.0437], [-1, 0, 0, 0], [0, -1, 0, 0.0512], [0, 0, 0, 1]], float)
_K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
# The arm's base 10 cm ahead of Reachy's foot, reaching straight ahead: the first view
# looks straight ahead, 30° down, so the middle of frame 1 is the table ~41 cm ahead.
_MOUNT = {"x": 0.10, "y": 0.0, "yaw_deg": 0.0}


def _camera_mini():
    mini = MagicMock()
    mini.media.camera.resolution = (640, 480)
    mini.media.camera.K = _K
    mini.media.camera.D = np.zeros(5)
    mini.T_head_cam = _T_HEAD_CAM
    return mini


class _Head:
    """turn() / grab() stand-ins: each frame comes with the pose the head was last turned to."""

    def __init__(self):
        self.turns, self.pose = [], None

    def turn(self, yaw, tilt):
        from reachy_mini.utils import create_head_pose

        self.turns.append((round(yaw, 1), tilt))
        self.pose = create_head_pose(yaw=yaw, pitch=tilt, degrees=True)

    def grab(self):
        return np.zeros((480, 640, 3), np.uint8), self.pose


class _Arm:
    """An ArmClient stand-in. submit() answers at once with replies[action] (a dict, a
    function of the args, or None for never), else {"ok": True}; status reports what the
    gripper holds."""

    def __init__(self, replies=None, holding=None):
        self.replies, self.holding, self.sent = dict(replies or {}), holding, []

    def submit(self, action, args=None):
        self.sent.append((action, args))
        reply = self.replies.get(action, {"ok": True})
        reply = reply(args) if callable(reply) else reply
        future = Future()
        if reply is not None:
            if action == "grasp" and reply.get("holding"):
                self.holding = args["label"]
            if action == "place" and reply.get("ok"):
                self.holding = None
            future.set_result(reply)
        return future

    def call(self, action, args=None, timeout=None):
        assert action == "status" and timeout
        return {"ok": True, "busy": False, "queued": 0, "holding": self.holding}

    def moves(self):
        return [a for a, _ in self.sent]


class _Step(SimpleNamespace):
    """An Interactions step: attributes + model_dump(mode="json")."""

    def model_dump(self, mode="python"):
        return copy.deepcopy(vars(self))


def _thought(signature="c2lnLTE="):
    return _Step(type="thought", signature=signature, summary=[{"type": "text", "text": "Find the eraser."}])


def _call(call_id, name, **arguments):
    return _Step(type="function_call", id=call_id, name=name, arguments=arguments)


def _turn(*steps, status="requires_action"):
    return list(steps), status


class _ER:
    """client.interactions answering each create() with the next scripted turn; then a
    completed interaction with no calls. Records a deep copy of every request."""

    def __init__(self, *turns):
        self.turns, self.calls = list(turns), []

    @property
    def interactions(self):
        return self

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        steps, status = self.turns.pop(0) if self.turns else ([], "completed")
        return SimpleNamespace(status=status, steps=steps, output_text=None if steps else "I think it's done.")


def _run(er, arm=None, mode="on", mount=_MOUNT, stop=None, goal="pick up the pink eraser"):
    arm, head = arm or _Arm(), _Head()
    task = at.ArmTask(er, "er-test", arm, _camera_mini(), head.turn, head.grab, mount, mode)
    return task.run(goal, stop or threading.Event()), arm, head


def _results(er, n):
    """The function_results sent with ER call *n* for the turn before it: (call id, result, content)."""
    new = er.calls[n]["input"][len(er.calls[n - 1]["input"]):]
    return [
        (s["call_id"], json.loads(s["result"][0]["text"]), s["result"]) for s in new if s["type"] == "function_result"
    ]


def _er_point(p_T, pose):
    """The [y, x] (0-1000) ER would give for a T-frame point seen at *pose*."""
    cam = pose @ _T_HEAD_CAM
    x, y, z = np.linalg.inv(cam)[:3] @ [p_T[0], p_T[1], p_T[2] - 0.177, 1.0]
    u, v = _K[0, 0] * x / z + _K[0, 2], _K[1, 1] * y / z + _K[1, 2]
    return [round(v / 480 * 1000), round(u / 640 * 1000)]


_FINISH = _turn(_call("f", "finish", outcome="done", say="I picked up the pink eraser."))


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("reachy_emotion.conversation_app._load_env", lambda: None)
    for key in ("ARM_TASKS", "ARM_BASE_X_CM", "ARM_BASE_Y_CM", "ARM_BASE_YAW_DEG"):
        monkeypatch.delenv(key, raising=False)


def test_arm_tasks_are_off_unless_set(_no_dotenv, monkeypatch):
    assert at.arm_tasks_mode() == "off"
    monkeypatch.setenv("ARM_TASKS", " Hover ")
    assert at.arm_tasks_mode() == "hover"
    monkeypatch.setenv("ARM_TASKS", "yes")
    assert at.arm_tasks_mode() == "off"  # not a mode: off


def test_the_mount_needs_all_three_keys_as_numbers(_no_dotenv, monkeypatch):
    monkeypatch.setenv("ARM_BASE_X_CM", "12")
    monkeypatch.setenv("ARM_BASE_Y_CM", "20.5")
    assert at.arm_mount_from_env() is None  # no yaw
    monkeypatch.setenv("ARM_BASE_YAW_DEG", "-15")
    assert at.arm_mount_from_env() == {"x": pytest.approx(0.12), "y": pytest.approx(0.205), "yaw_deg": -15.0}
    monkeypatch.setenv("ARM_BASE_Y_CM", "left")
    assert at.arm_mount_from_env() is None


def test_the_first_view_looks_down_at_the_table_in_front_of_the_arm():
    mount = {"x": 0.10, "y": 0.20, "yaw_deg": 0.0}  # 20 cm to Reachy's left, reaching ahead
    assert at.work_view_deg(mount) == pytest.approx(np.degrees(np.arctan2(0.20, 0.28)))
    _, _, head = _run(_ER(_FINISH), mount=mount)
    assert head.turns[0] == (35.5, 30)  # toward (28, 20) cm, tilted down
    instruction = at.arm_instruction(mount)
    assert "in front of Reachy, on its left, about 22 cm" in instruction
    assert "hand or a person" in instruction and "a result is not proof, the frame is" in instruction
    assert "into an open container up to 9 cm tall" in instruction  # what the IK can lift to over most spots


# --------------------------------------------------------------------------- #
# The loop                                                                      #
# --------------------------------------------------------------------------- #

def test_every_er_call_carries_the_tools_instruction_and_low_thinking_and_isnt_stored():
    er = _ER(_turn(_thought(), _call("c1", "look", tilt="down")), _FINISH)
    result, _, _ = _run(er)
    assert len(er.calls) == 2
    for kw in er.calls:
        assert kw["model"] == "er-test" and kw["store"] is False
        assert kw["tools"] == at.ARM_TOOLS and [t["name"] for t in kw["tools"]] == ["look", "grasp", "place", "finish"]
        assert kw["system_instruction"] == at.arm_instruction(_MOUNT)
        assert kw["generation_config"] == {"thinking_level": "low"}
    first = er.calls[0]["input"]
    assert len(first) == 1 and first[0]["type"] == "user_input"
    image, text = first[0]["content"]
    assert image["type"] == "image" and image["mime_type"] == "image/jpeg" and image["data"]
    assert text["text"] == (
        "Frame 1: head facing ahead, tilted down 30°. Holding: nothing.\nTask: pick up the pink eraser"
    )
    assert result == {"outcome": "done", "say": "I picked up the pink eraser.", "holding": None,
                      "did": ["looked ahead, tilted down 30°"]}


def test_model_steps_go_back_exactly_as_received():
    thought, look = _thought("b3BhcXVl"), _call("c1", "look", facing="on my left")
    er = _ER(_turn(thought, look), _FINISH)
    _run(er)
    sent = er.calls[1]["input"]
    assert sent[1] == thought.model_dump(mode="json")  # the signature goes back untouched
    assert sent[2] == {"type": "function_call", "id": "c1", "name": "look", "arguments": {"facing": "on my left"}}
    assert sent[3]["type"] == "function_result" and sent[3]["call_id"] == "c1" and sent[3]["name"] == "look"


def test_each_call_is_answered_once_and_the_new_frame_rides_on_the_last_result():
    er = _ER(
        _turn(
            _call("c1", "look", tilt="down"), _call("c1", "look", tilt="down"), _call("c2", "look", facing="on my left"),
        ),
        _FINISH,
    )
    _, _, head = _run(er)
    results = _results(er, 1)
    assert [r[0] for r in results] == ["c1", "c2"]  # the repeated id once
    (_, first, first_content), (_, second, second_content) = results
    assert first == {"ok": True, "executed": True} and len(first_content) == 1  # text only
    assert second == at._ONE_ACTION
    image, caption = second_content[1:]
    assert image["type"] == "image" and image["mime_type"] == "image/jpeg"
    assert caption == {"type": "text", "text": "Frame 2: head facing ahead, tilted down 30°. Holding: nothing."}
    assert head.turns == [(0.0, 30), (0.0, 30)]  # the work view, then the one look


def test_a_second_physical_call_in_a_turn_is_refused():
    er = _ER(
        _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pink eraser"),
              _call("p", "place", frame=1, target="the table", release="on_table", point=[600, 500])),
        _FINISH,
    )
    _, arm, _ = _run(er, _Arm({"grasp": {"ok": True, "holding": True}}))
    assert arm.moves() == ["grasp"]
    assert _results(er, 1)[1][1] == {
        "ok": False, "executed": False, "error": "One action per turn: check the new frame first.",
    }


def test_a_grasp_aims_at_the_table_point_and_its_result_is_not_proof():
    er = _ER(_turn(_call("g", "grasp", point=[500, 500], frame=1, label="pink eraser")), _FINISH)
    result, arm, _ = _run(er, _Arm({"grasp": {"ok": True, "holding": True}}))
    (action, args), = arm.sent
    # The middle of the view, 30° down: 40.9 cm ahead on the table; aimed 5 mm up, a bit nearer.
    assert action == "grasp" and args["label"] == "pink eraser" and "yaw_deg" not in args
    assert args["x"] == pytest.approx(0.409 - 0.005 / np.tan(np.radians(30)), abs=0.002)
    assert args["y"] == pytest.approx(0.0, abs=1e-3)
    (_, res, content), = _results(er, 1)
    assert res == {"ok": True, "executed": True, "holding": True,
                   "check": "Claim success only if the new frame shows the pink eraser in the gripper, off the table."}
    assert content[2]["text"] == "Frame 2: head facing ahead, tilted down 30°. Holding: pink eraser."
    assert result["holding"] == "pink eraser" and result["did"] == ["picked up the pink eraser"]


def test_a_point_on_an_old_frame_is_refused_and_nothing_moves():
    er = _ER(_turn(_call("g", "grasp", point=[500, 500], frame=0, label="eraser")), _FINISH)
    _, arm, head = _run(er)
    assert arm.sent == []
    (_, res, content), = _results(er, 1)
    assert res == {"ok": False, "executed": False, "error": "frame must be the latest frame, 1: point on that one"}
    assert len(content) == 1  # nothing moved: no new frame
    assert len(head.turns) == 1


def test_bad_points_are_refused():
    bad = ([500, 500, 3], [500, 1200], "middle", [True, 500], None)
    er = _ER(*[_turn(_call(f"g{i}", "grasp", point=p, frame=1, label="eraser")) for i, p in enumerate(bad)], _FINISH)
    _, arm, _ = _run(er)
    assert arm.sent == []
    for n in range(1, len(bad) + 1):
        (_, res, _), = _results(er, n)
        assert res["ok"] is False and "point must be [y, x]" in res["error"]


def test_a_point_too_far_off_to_aim_at_is_refused():
    er = _ER(_turn(_call("g", "grasp", point=[5, 500], frame=1, label="eraser")), _FINISH)  # top of the view
    _, arm, _ = _run(er)
    assert arm.sent == [] and "too far off" in _results(er, 1)[0][1]["error"]


def test_a_motor_fault_ends_the_task_without_asking_er_again():
    fault = {"ok": False, "error": "motor fault: wrist_roll overload; arm frozen in place",
             "faults": {"wrist_roll": ["overload"]}}
    er = _ER(_turn(_call("g", "grasp", point=[500, 500], frame=1, label="pink eraser")), _FINISH)
    result, _, _ = _run(er, _Arm({"grasp": fault}))
    assert len(er.calls) == 1
    assert result["outcome"] == "stopped" and "wrist_roll overload" in result["say"]


def test_a_stop_ends_the_task_and_says_what_it_still_holds():
    stop = threading.Event()

    def stopped(args):
        stop.set()  # move_arm(stop): the event, then the server cancels the move
        return {"ok": False, "error": "stopped", "cancelled": True}

    er = _ER(_turn(_call("p", "place", frame=1, target="the table", release="on_table", point=[600, 500])), _FINISH)
    result, _, _ = _run(er, _Arm({"place": stopped}, holding="pink eraser"), stop=stop)
    assert len(er.calls) == 1
    assert result["outcome"] == "stopped" and result["say"] == "I stopped. I'm still holding the pink eraser."
    assert result["holding"] == "pink eraser"

    already = threading.Event()
    already.set()
    er = _ER(_FINISH)
    result, _, _ = _run(er, stop=already)
    assert er.calls == [] and result["outcome"] == "stopped"


def test_an_arm_that_doesnt_answer_is_stopped(monkeypatch):
    monkeypatch.setattr(at, "_ARM_TIMEOUT_S", 0.05)
    er = _ER(_turn(_call("g", "grasp", point=[500, 500], frame=1, label="eraser")), _FINISH)
    result, arm, _ = _run(er, _Arm({"grasp": None}))
    assert arm.moves() == ["grasp", "stop"] and len(er.calls) == 1
    assert result["outcome"] == "failed" and "stopped it" in result["say"]


def test_requires_action_continues_and_any_other_status_fails():
    er = _ER(_turn(_call("c1", "look")), _turn(_call("c2", "look"), status="failed"))
    result, _, _ = _run(er)
    assert len(er.calls) == 2 and result["outcome"] == "failed" and "failed" in result["say"]
    result, _, _ = _run(_ER(_turn(status="incomplete")))
    assert result["outcome"] == "failed"


def test_er_ending_without_a_call_ends_the_task_with_what_it_said():
    er = _ER(_turn(_call("c1", "look")))  # then: completed, no calls
    result, _, _ = _run(er)
    assert len(er.calls) == 2 and result["outcome"] == "unclear" and result["say"] == "I think it's done."


def test_the_task_stops_after_eight_er_turns():
    er = _ER(*[_turn(_call(f"c{i}", "look")) for i in range(12)])
    result, _, _ = _run(er)
    assert len(er.calls) == 8
    assert result["outcome"] == "failed" and "ran out of steps" in result["say"]


def test_finish_in_the_same_turn_as_an_action_waits_for_the_new_frame():
    er = _ER(_turn(_call("c1", "look"), _call("f1", "finish", outcome="done", say="Done.")), _FINISH)
    result, _, _ = _run(er)
    assert len(er.calls) == 2
    assert "check the new frame first" in _results(er, 1)[1][1]["error"]
    assert result["outcome"] == "done"


def test_hover_mode_only_hovers_over_each_target():
    er = _ER(
        _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pink eraser")),
        _turn(_call("p", "place", frame=2, target="the table", release="on_table", point=[600, 500])),
        _FINISH,
    )
    _, arm, _ = _run(er, mode="hover")
    assert arm.moves() == ["hover", "hover"]  # never grasp or place
    assert set(arm.sent[0][1]) == {"x", "y"} and arm.sent[0][1]["x"] == pytest.approx(0.400, abs=0.002)
    assert arm.sent[1][1]["height"] == 0.02  # a place's hover goes as high as the place would start down
    (_, res, _), = _results(er, 1)
    assert res["note"] == "test mode: hovered only" and res["holding"] is False


def test_ends_turn_the_jaws_across_the_long_side():
    across = _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pen", ends=[[500, 400], [500, 600]]))
    _, arm, _ = _run(_ER(across, _FINISH))
    assert arm.sent[0][1]["yaw_deg"] == pytest.approx(0.0, abs=0.5)  # lies left-right: jaws close front-back
    along = _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pen", ends=[[450, 500], [550, 500]]))
    _, arm, _ = _run(_ER(along, _FINISH))
    assert abs(arm.sent[0][1]["yaw_deg"]) == pytest.approx(90.0, abs=0.5)  # lies front-back: jaws close sideways
    bad = _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pen", ends=[[450, 500]]))
    _, arm, _ = _run(_ER(bad, _FINISH))
    assert arm.sent == []


def test_dropping_into_a_cup_aims_over_the_middle_of_its_opening():
    from reachy_mini.utils import create_head_pose

    pose = create_head_pose(pitch=30, degrees=True)  # frame 1
    front, rim, radius = np.array([0.36, 0.02, 0.0]), 0.08, 0.04  # an 8 cm cup, 8 cm across
    into = dict(
        frame=1, target="the white cup", release="into",
        base=_er_point(front, pose),
        rim_near=_er_point(front + [0, 0, rim], pose),
        rim_far=_er_point(front + [2 * radius, 0, rim], pose),
    )
    er = _ER(_turn(_call("p", "place", **into)), _FINISH)
    _, arm, _ = _run(er, _Arm(holding="pink eraser"))
    (action, args), = arm.sent
    assert action == "place"
    assert args["x"] == pytest.approx(0.40, abs=0.005) and args["y"] == pytest.approx(0.02, abs=0.005)
    assert args["height"] == pytest.approx(rim + 0.03, abs=0.004)  # 3 cm above the rim

    # Hover mode: the hover keeps that height (clear of the rim), not the usual ~9 cm of a hover.
    _, arm, _ = _run(_ER(_turn(_call("p", "place", **into)), _FINISH), mode="hover")
    (action, hover), = arm.sent
    assert action == "hover" and hover == {k: args[k] for k in ("x", "y", "height")}

    for cm in (11, 14):  # taller than the arm can lift a release over
        tall = dict(into, rim_near=_er_point(front + [0, 0, cm / 100], pose),
                    rim_far=_er_point(front + [0.08, 0, cm / 100], pose))
        er = _ER(_turn(_call("p", "place", **tall)), _FINISH)
        _, arm, _ = _run(er, _Arm(holding="pink eraser"))
        error = _results(er, 1)[0][1]["error"]
        assert arm.sent == [] and f"{cm} cm tall" in error and "containers up to 9 cm" in error


def test_putting_down_on_the_table_releases_2_cm_up():
    er = _ER(_turn(_call("p", "place", frame=1, target="the table", release="on_table", point=[500, 500])), _FINISH)
    result, arm, _ = _run(er, _Arm(holding="pink eraser"))
    (action, args), = arm.sent
    assert action == "place" and args["height"] == 0.02 and args["x"] == pytest.approx(0.409, abs=0.002)
    assert result["holding"] is None and result["did"] == ["put the pink eraser down on the table"]


def test_grasp_and_place_need_the_right_gripper_state():
    grasp = _turn(_call("g", "grasp", point=[500, 500], frame=1, label="pen"))
    _, arm, _ = _run(_ER(grasp, _FINISH), _Arm(holding="pink eraser"))
    assert arm.sent == []
    place = _turn(_call("p", "place", frame=1, target="the table", release="on_table", point=[500, 500]))
    er = _ER(place, _FINISH)
    _, arm, _ = _run(er)
    assert arm.sent == [] and "gripper is empty" in _results(er, 1)[0][1]["error"]


def test_look_turns_the_head_and_the_caption_says_where_it_faces():
    er = _ER(_turn(_call("c1", "look", facing="on my left")), _turn(_call("c2", "look", facing="sideways")), _FINISH)
    result, _, head = _run(er)
    assert head.turns == [(0.0, 30), (90.0, 30)]  # kept the tilt
    (_, _, content), = _results(er, 1)
    assert content[2]["text"] == "Frame 2: head facing on my left, tilted down 30°. Holding: nothing."
    (_, res, _), = _results(er, 2)
    assert res["ok"] is False and res["error"].startswith("facing must be one of: ahead,")
    assert head.turns[-1] == (90.0, 30) and result["did"] == ["looked on my left, tilted down 30°"]


def test_no_new_er_call_once_the_time_is_up(monkeypatch):
    monkeypatch.setattr(at, "_BUDGET_S", 0.0)
    er = _ER(_FINISH)
    result, _, _ = _run(er)
    assert er.calls == [] and result["outcome"] == "failed" and "out of time" in result["say"]


def test_a_failed_er_call_ends_the_task_and_says_what_it_holds():
    class _Down(_ER):
        def create(self, **kw):
            raise RuntimeError("503 overloaded")

    result, _, _ = _run(_Down(), _Arm(holding="pink eraser"))
    assert result == {"outcome": "failed", "say": "My planner didn't answer, so I stopped. I'm still holding the pink eraser.",
                      "holding": "pink eraser", "did": []}
