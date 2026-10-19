"""Tests for SceneBrain (Gemini Robotics ER 2 scene tools) — no network, no robot."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_emotion import scene_brain as sb

_FRAME = np.zeros((48, 64, 3), np.uint8)


# --------------------------------------------------------------------------- #
# Parsing                                                                       #
# --------------------------------------------------------------------------- #

def test_parse_json_strips_code_fences():
    text = '```json\n[{"point": [500, 250], "label": "red circle"}]\n```'
    assert sb._parse_json(text) == [{"point": [500, 250], "label": "red circle"}]


def test_parse_json_plain_and_invalid():
    assert sb._parse_json("[1, 2]") == [1, 2]
    assert sb._parse_json("not json") is None
    assert sb._parse_json("") is None
    assert sb._parse_json(None) is None


def test_valid_objects_filters_malformed_before_capping():
    parsed = [
        {"point": [1, 2, 3], "label": "bad point"},
        {"point": [100, 100], "label": ""},
        {"label": "no point"},
        "not a dict",
        *[{"point": [10, 10 + i], "label": f"o{i}"} for i in range(20)],
    ]
    out = sb._valid_objects(parsed)
    # malformed entries don't consume the cap: we still get a full set of valid ones
    assert len(out) == sb._MAX_OBJECTS
    assert out[0] == {"label": "o0", "point": [10.0, 10.0]}
    assert sb._valid_objects({"not": "a list"}) == []


# --------------------------------------------------------------------------- #
# Geometry                                                                      #
# --------------------------------------------------------------------------- #

def test_points_to_pixels_maps_and_clamps_strictly_inside():
    px = sb.points_to_pixels([[500, 250], [0, 0], [1000, 1000]], width=640, height=480)
    assert px[0] == (160, 240)
    assert px[1] == (1, 1)        # look_at_image asserts 0 < u < width
    assert px[2] == (639, 479)


def test_body_direction_like_a_person_from_the_default_facing():
    said = {deg: sb.body_direction(deg) for deg in (0, 20, -20, 45, -45, 90, -90, 135, -135, 170, -170, 180)}
    assert said == {
        0: "ahead", 20: "ahead", -20: "ahead",
        45: "ahead on my left", -45: "ahead on my right",
        90: "on my left", -90: "on my right",
        135: "behind me on my left", -135: "behind me on my right",
        170: "behind me", -170: "behind me", 180: "behind me",
    }
    assert sb.body_direction(270) == "on my right"  # wraps
    # well below or above eye level
    assert sb.body_direction(0, -35) == "ahead, down low" and sb.body_direction(90, 25) == "on my left, up high"
    assert sb.body_direction(0, -10) == "ahead"


# Reachy Mini's head→camera transform (from the SDK) and a 640x480 camera with a ~65° view.
_T_HEAD_CAM = np.array([[0, 0, 1, 0.0437], [-1, 0, 0, 0], [0, -1, 0, 0.0512], [0, 0, 0, 1]], float)


def _camera_mini():
    mini = MagicMock()
    mini.media.camera.K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    mini.media.camera.D = np.zeros(5)
    mini.T_head_cam = _T_HEAD_CAM
    return mini


def test_bearing_is_from_the_default_facing_whatever_the_head_did():
    from reachy_mini.utils import create_head_pose

    mini = _camera_mini()
    ahead = create_head_pose(degrees=True)
    assert sb.bearing_deg(mini, (320, 240), ahead) == pytest.approx(0.0, abs=1e-6)
    assert sb.bearing_deg(mini, (0 + 1, 240), ahead) == pytest.approx(32.6, abs=0.1)  # image left = my left
    assert sb.bearing_deg(mini, (639, 240), ahead) == pytest.approx(-32.5, abs=0.1)
    turned = create_head_pose(yaw=120, degrees=True)  # seen while turned to +120°
    assert sb.bearing_deg(mini, (320, 240), turned) == pytest.approx(120.0, abs=1e-6)
    looking_down = create_head_pose(pitch=25, yaw=-90, degrees=True)
    assert sb.bearing_deg(mini, (320, 100), looking_down) == pytest.approx(-90.0, abs=1e-6)


def test_bearing_falls_back_to_the_heading_without_a_camera_model():
    from reachy_mini.utils import create_head_pose

    mini = MagicMock()
    mini.media.camera.K = None  # camera specs not set
    assert sb.bearing_deg(mini, (10, 240), create_head_pose(yaw=60, degrees=True)) == pytest.approx(60.0)
    assert sb.bearing_deg(mini, (10, 240), None) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Table points (arm targets, T frame: on the table under the foot, x ahead, y left)  #
# --------------------------------------------------------------------------- #

def _to_pixel(p_T, pose):
    """Project a T-frame point into the image of _camera_mini (no distortion)."""
    cam = pose @ _T_HEAD_CAM
    p_head = np.asarray(p_T, float) - [0, 0, sb.HEAD_ABOVE_TABLE_M]
    x, y, z = np.linalg.inv(cam)[:3] @ [*p_head, 1.0]
    K = _camera_mini().media.camera.K
    return (K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2])


def test_table_point_straight_down_the_optical_axis():
    from reachy_mini.utils import create_head_pose

    mini = _camera_mini()
    pose = create_head_pose(pitch=30, degrees=True)  # + = down
    # Expected from T_head_cam itself: the camera's optical axis (its z column) from the
    # camera's position, down to the table.
    cam = pose @ _T_HEAD_CAM
    origin, axis = cam[:3, 3] + [0, 0, sb.HEAD_ABOVE_TABLE_M], cam[:3, 2]
    expected = origin - origin[2] / axis[2] * axis
    point = sb.table_point(mini, (320, 240), pose)
    assert point == pytest.approx(expected, abs=1e-6)
    assert point == pytest.approx([0.409, 0.0, 0.0], abs=0.002)  # camera 19.95 cm up, 6.3 cm ahead, 30° down
    assert sb.plane_point(mini, (320, 240), pose, 0.0) == pytest.approx(expected, abs=1e-6)


def test_table_point_is_in_the_foot_frame_whatever_the_body_did():
    from reachy_mini.utils import create_head_pose

    mini = _camera_mini()
    left = sb.table_point(mini, (320, 240), create_head_pose(pitch=30, yaw=90, degrees=True))
    assert left == pytest.approx([0.0, 0.409, 0.0], abs=0.002)  # turned to its left: same distance, on +y


def test_a_target_never_comes_from_a_missing_pose():
    mini = _camera_mini()
    assert sb.plane_point(mini, (320, 240), None, 0.0) is None
    assert sb.table_point(mini, (320, 240), None) is None
    assert sb.height_above(mini, np.array([0.3, 0.0, 0.0]), (320, 240), None) is None


def test_level_upward_and_shallow_rays_give_no_table_point():
    from reachy_mini.utils import create_head_pose

    mini = _camera_mini()
    level = create_head_pose(degrees=True)
    assert sb.table_point(mini, (320, 240), level) is None       # level
    assert sb.plane_point(mini, (320, 240), level, 0.0) is None
    assert sb.table_point(mini, (320, 10), level) is None        # above the horizon
    assert sb.plane_point(mini, (320, 10), level, 0.0) is None
    shallow = create_head_pose(pitch=5, degrees=True)  # 5° down: hits the table ~2.6 m away
    assert sb.table_point(mini, (320, 240), shallow) is None
    assert sb.plane_point(mini, (320, 240), shallow, 0.0)[0] > 2.0
    # a plane above the camera isn't "down" the ray
    assert sb.plane_point(mini, (320, 240), create_head_pose(pitch=30, degrees=True), 0.5) is None


def test_a_point_above_the_table_lands_nearer_the_camera():
    from reachy_mini.utils import create_head_pose

    mini, pose = _camera_mini(), create_head_pose(pitch=30, degrees=True)
    on_table = sb.table_point(mini, (320, 300), pose)
    raised = sb.table_point(mini, (320, 300), pose, height_m=0.02)
    assert raised[2] == pytest.approx(0.02) and raised[0] < on_table[0]


def test_height_above_finds_a_rim_over_the_base():
    from reachy_mini.utils import create_head_pose

    mini, pose = _camera_mini(), create_head_pose(pitch=30, degrees=True)
    base = np.array([0.36, 0.03, 0.0])  # where a cup's front touches the table
    for rim in (0.04, 0.08):
        pixel = tuple(round(c) for c in _to_pixel(base + [0, 0, rim], pose))
        assert sb.height_above(mini, base, pixel, pose) == pytest.approx(rim, abs=0.003)
    tall = tuple(round(c) for c in _to_pixel(base + [0, 0, 0.18], pose))
    assert sb.height_above(mini, base, tall, pose) == pytest.approx(0.15)  # clipped


def test_camera_size_prefers_camera_resolution_then_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    mini = MagicMock()
    mini.media.camera.resolution = (1280, 720)
    assert sb.camera_size(mini, frame) == (1280, 720)
    no_cam = MagicMock()
    no_cam.media.camera = None
    assert sb.camera_size(no_cam, frame) == (640, 480)


# --------------------------------------------------------------------------- #
# Gaze                                                                          #
# --------------------------------------------------------------------------- #

def _gaze_mini(home):
    mini = MagicMock()
    mini.get_current_head_pose.return_value = home
    targets = [np.full((4, 4), 1.0), np.full((4, 4), 2.0)]
    mini.look_at_image.side_effect = targets
    return mini, targets


def test_look_at_objects_computes_all_targets_before_moving(monkeypatch):
    # look_at_image rays come from the head's *current* pose, so every target must be
    # computed (perform_movement=False) before the head moves — else they stack.
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    home = np.eye(4)
    mini, targets = _gaze_mini(home)
    sb.look_at_objects(mini, [(100, 200), (300, 400)], capture_pose=home)

    names = [c[0] for c in mini.mock_calls if c[0] in ("look_at_image", "goto_target")]
    assert names == ["look_at_image", "look_at_image", "goto_target", "goto_target", "goto_target"]
    for c in mini.look_at_image.call_args_list:
        assert c.kwargs["perform_movement"] is False
    heads = [c.kwargs["head"] for c in mini.goto_target.call_args_list]
    assert heads[0] is targets[0] and heads[1] is targets[1]
    assert [c.kwargs["body_yaw"] for c in mini.goto_target.call_args_list] == [None, None, 0.0]
    _assert_neutral(mini.goto_target.call_args, body_yaw=0.0)  # tour ends at neutral


def _assert_neutral(call, body_yaw):
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS
    from reachy_mini.utils import create_head_pose

    assert np.allclose(call.kwargs["head"], create_head_pose(yaw=body_yaw, degrees=False))
    assert call.kwargs["body_yaw"] == pytest.approx(body_yaw)
    assert call.kwargs["antennas"] == INIT_ANTENNAS_JOINT_POSITIONS


def test_look_at_objects_returns_to_capture_pose_first(monkeypatch):
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    home, capture = np.eye(4), np.eye(4) * 3.0  # head moved since the frame was taken
    mini, _ = _gaze_mini(home)
    sb.look_at_objects(mini, [(100, 200)], capture_pose=capture)

    first = [c for c in mini.mock_calls if c[0] in ("look_at_image", "goto_target")][0]
    assert first[0] == "goto_target" and first.kwargs["head"] is capture


def test_look_at_objects_stops_on_failure_but_still_returns_to_neutral(monkeypatch):
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    mini = MagicMock()
    mini.get_current_head_pose.return_value = np.eye(4)
    mini.look_at_image.side_effect = RuntimeError("Camera specs not set.")
    sb.look_at_objects(mini, [(100, 200), (300, 400)])
    assert mini.look_at_image.call_count == 1
    mini.goto_target.assert_called_once()  # just the return to neutral
    _assert_neutral(mini.goto_target.call_args, body_yaw=0.0)


def test_heading_reads_the_yaw_of_a_head_pose():
    from reachy_mini.utils import create_head_pose

    assert sb.heading(create_head_pose(yaw=30, degrees=True)) == pytest.approx(np.radians(30))
    assert sb.heading(create_head_pose(pitch=10, yaw=-45, degrees=True)) == pytest.approx(np.radians(-45))


def test_face_toward_turns_the_body_to_the_person_with_the_head_neutral():
    from reachy_mini.utils import create_head_pose

    mini = MagicMock()
    mini.get_current_head_pose.return_value = np.eye(4)
    mini.look_at_image.return_value = create_head_pose(pitch=12, yaw=40, degrees=True)
    sb.face_toward(mini, (500, 200), capture_pose=np.eye(4))
    mini.goto_target.assert_called_once()  # already at the capture pose: one turn
    _assert_neutral(mini.goto_target.call_args, body_yaw=np.radians(40))  # level, not pitched

    mini.goto_target.reset_mock()
    mini.look_at_image.return_value = create_head_pose(yaw=175, degrees=True)
    sb.face_toward(mini, (500, 200))
    _assert_neutral(mini.goto_target.call_args, body_yaw=np.radians(150))  # body limit


def test_face_toward_a_person_just_past_180_does_not_spin_the_other_way():
    from reachy_mini.utils import create_head_pose

    at_150 = create_head_pose(yaw=150, degrees=True)
    mini = MagicMock()
    mini.get_current_head_pose.return_value = at_150
    mini.look_at_image.return_value = create_head_pose(yaw=185, degrees=True)  # reads as -175°
    sb.face_toward(mini, (100, 200), capture_pose=at_150)
    _assert_neutral(mini.goto_target.call_args, body_yaw=np.radians(150))  # stays at +150, not -150


def test_look_at_objects_skips_targets_past_180(monkeypatch):
    from reachy_mini.utils import create_head_pose

    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    at_150 = create_head_pose(yaw=150, degrees=True)
    mini = MagicMock()
    mini.get_current_head_pose.return_value = at_150
    behind, ok = create_head_pose(yaw=185, degrees=True), create_head_pose(yaw=160, degrees=True)
    mini.look_at_image.side_effect = [behind, ok]
    sb.look_at_objects(mini, [(100, 200), (300, 200)], capture_pose=at_150)
    heads = [c.kwargs["head"] for c in mini.goto_target.call_args_list]
    assert len(heads) == 2 and heads[0] is ok  # 185° skipped; then neutral


# --------------------------------------------------------------------------- #
# SceneBrain (fake ER client)                                                   #
# --------------------------------------------------------------------------- #

class _FakeInteractions:
    """``client.interactions`` answering every create() with *text* (status *status*)."""

    def __init__(self, text, status="completed"):
        self.text, self.status = text, status
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(status=self.status, output_text=self.text)


def _brain(text, status="completed"):
    client = MagicMock()
    client.interactions = _FakeInteractions(text, status)
    return sb.SceneBrain(client, model="er-test"), client.interactions


def _prompt(call):
    return call["input"][1]["text"]


def test_locate_parses_er_points_and_passes_query():
    brain, er = _brain('```json\n[{"point": [500, 250], "label": "cup"}]\n```')
    assert brain.locate(_FRAME, "the cup") == [{"label": "cup", "point": [500.0, 250.0]}]
    call = er.calls[0]
    assert call["model"] == "er-test" and "the cup" in _prompt(call)
    assert call["input"][0]["type"] == "image" and call["input"][0]["mime_type"] == "image/jpeg"
    # pointing needs little thinking, and a room search makes one call per heading
    assert call["generation_config"] == {"thinking_level": "low"}
    # JSON in Google's pointing shape
    assert call["response_format"] == {"type": "text", "mime_type": "application/json", "schema": sb._POINTS_SCHEMA}
    # Interactions keeps requests unless told not to: these are camera frames of a home
    assert call["store"] is False
    # only things really there, named so two of a kind can be told apart
    assert "actually present" in _prompt(call) and "identifying name" in _prompt(call)


def test_the_er_client_times_out_and_retries_once():
    interactions = sb.er_client("test-key").interactions.sdk_configuration
    # a stalled ER request must not hang (google-genai's default is no timeout) ...
    assert interactions.timeout_ms == sb._ER_TIMEOUT_MS
    # ... and a busy preview model gets one retry: checked on what the SDK will do, since
    # Interactions reads HttpRetryOptions.attempts as retries, not tries
    assert interactions.retry_config.max_retries == 1


def test_an_unfinished_answer_is_an_error():
    brain, _ = _brain('{"items": []}', status="incomplete")
    with pytest.raises(RuntimeError, match="incomplete"):
        brain.locate(_FRAME, "cup")


def test_locate_reads_the_structured_items_shape():
    brain, _ = _brain('{"items": [{"point": [500, 250], "label": "red mug"}]}')
    assert brain.locate(_FRAME, "mug") == [{"label": "red mug", "point": [500.0, 250.0]}]


def test_an_unreadable_reply_is_an_error_not_nothing_found():
    for reply in ('Here it is: [{"point": [1, 2], "label": "cup"}]', None, '{"oops": true}'):
        brain, _ = _brain(reply)
        with pytest.raises(RuntimeError):
            brain.locate(_FRAME, "cup")
    for reply in ("garbled", '{"oops": true}', '{"feasible": true}'):
        brain, _ = _brain(reply)
        with pytest.raises(RuntimeError):
            brain.plan(_FRAME, "clear the table")


def test_locate_defaults_to_all_objects():
    brain, er = _brain('{"items": []}')
    assert brain.locate(_FRAME) == []
    assert "distinct objects" in _prompt(er.calls[0])


def test_plan_thinks_hard_and_knows_the_arm():
    brain, er = _brain('{"feasible": true, "steps": []}')
    brain.plan(_FRAME, "clear the table")
    call = er.calls[0]
    assert call["generation_config"] == {"thinking_level": "high"} and call["store"] is False
    assert call["response_format"]["schema"] is sb._PLAN_SCHEMA
    prompt = _prompt(call)
    assert "camera in its head" in prompt and "SO-101" in prompt and "on its left side" in prompt


def test_plan_returns_steps_and_keeps_last_plan():
    brain, _ = _brain('{"feasible": true, "steps": [{"step": 1, "action": "grasp", "object": "cup"}]}')
    result = brain.plan(_FRAME, "clear the table")
    assert result == {"steps": [{"step": 1, "action": "grasp", "object": "cup"}]}
    assert brain.last_plan == {"goal": "clear the table", **result}


def test_plan_can_say_it_cant_be_done():
    brain, _ = _brain('{"feasible": false, "reason": "the cup is out of reach", "steps": []}')
    assert brain.plan(_FRAME, "hand me the cup") == {"feasible": False, "reason": "the cup is out of reach"}
