"""Tests for the Gemini Live conversation helpers (W4-C1).

Covers the pure audio/format + tool logic, and drives the async receive loop
via asyncio.run (no pytest-asyncio needed). The full streaming session needs a
real API key + audio hardware and is out of scope here.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import numpy as np

from reachy_emotion import live_conversation as lc


# --------------------------------------------------------------------------- #
# Pure helpers                                                                  #
# --------------------------------------------------------------------------- #

def test_to_mono_downmixes():
    stereo = np.ones((10, 2), dtype=np.float32)
    assert lc._to_mono(stereo).shape == (10,)


def test_resample_linear_changes_length_and_is_identity_when_equal():
    x = np.linspace(0, 1, 300).astype(np.float32)
    assert lc._resample_linear(x, 24000, 16000).shape[0] == 200
    same = lc._resample_linear(x, 16000, 16000)
    assert same.shape[0] == 300


def test_pcm16_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    back = lc._pcm16_to_float(lc._float_to_pcm16(x))
    assert np.allclose(back, x, atol=1e-3)


def test_mic_to_mono16k_downmixes_stereo_no_resample():
    sample = np.ones((320, 2), dtype=np.float32)  # stereo @ 16k
    mono = lc._mic_to_mono16k(sample, src_rate=16000)
    assert mono.shape == (320,)
    assert np.allclose(mono, 1.0)


def test_mic_to_mono16k_resamples_when_needed():
    sample = np.zeros(480, dtype=np.float32)  # 480 @ 48k -> 160 @ 16k
    mono = lc._mic_to_mono16k(sample, src_rate=48000)
    assert mono.shape == (160,)


def test_audio_ring_accumulates_then_drains_once():
    ring = lc._AudioRing()
    assert ring.drain() is None
    ring.append(np.ones(10, dtype=np.float32))
    ring.append(np.ones(5, dtype=np.float32))
    out = ring.drain()
    assert out.shape == (15,)
    assert ring.drain() is None  # cleared after drain


def test_audio_ring_bounded_to_max_seconds():
    ring = lc._AudioRing(max_seconds=0.001, rate=16000)  # cap = 16 samples
    ring.append(np.ones(100, dtype=np.float32))
    assert ring.drain().shape == (16,)


def test_audio_ring_stays_bounded_without_drain():
    # With emotion disabled (or no frames) nothing drains the ring — the mic pump
    # appends forever, so memory must stay capped on append, not only on drain.
    ring = lc._AudioRing(max_seconds=0.1, rate=16000)  # cap = 1600 samples
    for _ in range(1000):
        ring.append(np.ones(320, dtype=np.float32))
    assert sum(c.shape[0] for c in ring._chunks) <= 1600
    assert ring.drain().shape == (1600,)


def test_live_pcm_to_speaker_resamples_24k_to_16k_shape():
    data = np.zeros(240, dtype="<i2").tobytes()  # 240 samples @ 24k
    out = lc._live_pcm_to_speaker(data)
    assert out.shape == (160, 1)  # 240 * 16000/24000
    assert out.dtype == np.float32


def test_encode_jpeg_returns_bytes():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    jpeg = lc._encode_jpeg(frame)
    assert isinstance(jpeg, bytes) and len(jpeg) > 0


def test_emotion_tool_result_dispatch():
    class _FC:
        def __init__(self, name):
            self.name = name
            self.id = "x"

    client = MagicMock()
    client.detect_emotion.return_value = {"dominant_emotion": "sad"}
    assert lc._emotion_tool_result(_FC("detect_emotion"), client) == {"dominant_emotion": "sad"}
    assert "error" in lc._emotion_tool_result(_FC("other_tool"), client)
    assert "error" in lc._emotion_tool_result(_FC("detect_emotion"), None)


# --------------------------------------------------------------------------- #
# Async receive loop                                                            #
# --------------------------------------------------------------------------- #

class _FC:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.args = {}


class _ToolCall:
    def __init__(self, fcs):
        self.function_calls = fcs


class _Msg:
    def __init__(self, data=None, tool_call=None, server_content=None):
        self.data = data
        self.tool_call = tool_call
        self.server_content = server_content


class _FakeSession:
    """Serves one turn's worth of messages, then reports the session idle/closed
    (receive() yields nothing) — mirroring the real turn-then-end behaviour."""

    def __init__(self, msgs):
        self._msgs = msgs
        self._served = False
        self.tool_responses = []

    async def receive(self):
        if self._served:
            return
        self._served = True
        for m in self._msgs:
            yield m

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)


def test_receive_loop_plays_audio_and_answers_tool(caplog):
    mini = MagicMock()
    emotion_client = MagicMock()
    emotion_client.detect_emotion.return_value = {"dominant_emotion": "happy", "confidence": 0.9}
    emotion_client.last_age = 0.4

    audio = np.zeros(240, dtype="<i2").tobytes()
    session = _FakeSession([
        _Msg(data=audio),
        _Msg(tool_call=_ToolCall([_FC("id1", "detect_emotion")])),
    ])

    with caplog.at_level("INFO"):
        asyncio.run(lc._receive_loop(session, mini, threading.Event(), emotion_client))

    mini.media.push_audio_sample.assert_called_once()
    assert len(session.tool_responses) == 1
    fr = session.tool_responses[0][0]
    assert fr.name == "detect_emotion"
    assert fr.response == {"dominant_emotion": "happy", "confidence": 0.9}
    emotion_client.detect_emotion.assert_called_once()
    assert "(read 0.4s ago)" in caplog.text  # the reading's age, for diagnosing flips
    mini.play_move.assert_not_called()  # emotion no longer drives robot moves


def test_receive_loop_stops_when_event_set():
    mini = MagicMock()
    stop = threading.Event()
    stop.set()  # already stopped → first message breaks out
    session = _FakeSession([_Msg(data=np.zeros(10, dtype="<i2").tobytes())])

    asyncio.run(lc._receive_loop(session, mini, stop, MagicMock()))

    mini.media.push_audio_sample.assert_not_called()


class _Content:
    def __init__(self, interrupted=False, heard=None, said=None):
        self.interrupted = interrupted
        self.input_transcription = MagicMock(text=heard) if heard else None
        self.output_transcription = MagicMock(text=said) if said else None


def test_receive_loop_logs_user_and_reachy_transcripts(caplog):
    session = _FakeSession([_Msg(server_content=_Content(heard="where is the cup")),
                            _Msg(server_content=_Content(said="On the left."))])

    with caplog.at_level("INFO"):
        asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None))

    assert "User   → where is the cup" in caplog.text
    assert "Reachy → On the left." in caplog.text


def test_receive_loop_flushes_speaker_on_barge_in():
    # Live streams audio faster than real time; when the user interrupts, the
    # already-queued reply must be dropped instead of talking over them.
    mini = MagicMock()
    session = _FakeSession([_Msg(server_content=_Content(interrupted=False)),
                            _Msg(server_content=_Content(interrupted=True))])

    asyncio.run(lc._receive_loop(session, mini, threading.Event(), None))

    mini.media.audio.clear_player.assert_called_once()


def test_run_live_conversation_logs_why_a_task_died(monkeypatch, caplog):
    import reachy_emotion.scene_brain as sb

    class _Connect:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *exc):
            return False

    client = MagicMock()
    client.aio.live.connect.return_value = _Connect()
    monkeypatch.setattr("google.genai.Client", lambda api_key: client)
    monkeypatch.setattr(lc, "_load_api_key", lambda: "k")
    monkeypatch.setattr(sb, "_load_er_model", lambda: "er")
    monkeypatch.setattr("reachy_emotion.arm_client.arm_from_env", lambda: None)  # no arm

    async def _dies(*a):
        raise RuntimeError("1011 server closed")

    async def _idle(*a):
        await asyncio.sleep(10)

    monkeypatch.setattr(lc, "_audio_pump", _dies)
    monkeypatch.setattr(lc, "_emotion_video_loop", _idle)
    monkeypatch.setattr(lc, "_receive_loop", _idle)
    mini = MagicMock()
    mini.media.get_input_audio_samplerate.return_value = 16000

    with caplog.at_level("ERROR"):
        asyncio.run(lc.run_live_conversation(mini, threading.Event(), "p", "m"))

    assert "Live conversation failed: 1011 server closed" in caplog.text


def test_ctrl_c_stops_session_tasks_before_the_socket_closes(monkeypatch):
    # Ctrl-C cancels run_live_conversation mid-sleep. The session tasks must be
    # cancelled and collected before the Live socket closes — otherwise they die on
    # the closed socket and asyncio logs "Task exception was never retrieved".
    import reachy_emotion.scene_brain as sb

    events = []

    class _Connect:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *exc):
            events.append("socket closed")
            return False

    client = MagicMock()
    client.aio.live.connect.return_value = _Connect()
    monkeypatch.setattr("google.genai.Client", lambda api_key: client)
    monkeypatch.setattr(lc, "_load_api_key", lambda: "k")
    monkeypatch.setattr(sb, "_load_er_model", lambda: "er")
    monkeypatch.setattr("reachy_emotion.arm_client.arm_from_env", lambda: None)  # no arm

    def _runs_until_cancelled(name):
        async def _task(*a):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                events.append(f"{name} cancelled")
                raise
        return _task

    for name in ("_audio_pump", "_emotion_video_loop", "_receive_loop"):
        monkeypatch.setattr(lc, name, _runs_until_cancelled(name))
    mini = MagicMock()
    mini.media.get_input_audio_samplerate.return_value = 16000
    stop = threading.Event()

    async def _ctrl_c():
        main = asyncio.create_task(lc.run_live_conversation(mini, stop, "p", "m"))
        await asyncio.sleep(0.2)
        main.cancel()  # what asyncio.run's SIGINT handler does
        try:
            await main
        except asyncio.CancelledError:
            pass

    asyncio.run(_ctrl_c())

    assert events[-1] == "socket closed"
    assert sorted(events[:-1]) == [
        "_audio_pump cancelled", "_emotion_video_loop cancelled", "_receive_loop cancelled",
    ]
    mini.media.stop_recording.assert_called_once()  # cleanup still ran
    assert stop.is_set()  # ends a room search still running in a worker thread
    assert mini.goto_target.call_args_list[0].kwargs["body_yaw"] == 0.0  # started from neutral


class _CollectSession:
    def __init__(self):
        self.sent = []

    async def send_realtime_input(self, **kw):
        self.sent.append(kw)


def test_emotion_video_loop_warms_model_and_sends_video():
    # A continuous warm loop must feed the local model AND stream video to Gemini.
    stop = threading.Event()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    mini = MagicMock()
    mini.media.get_frame.return_value = frame

    emotion_client = MagicMock()
    emotion_client.process.side_effect = lambda f, a: stop.set()  # exit after 1 pass

    ring = lc._AudioRing()
    ring.append(np.ones(8, dtype=np.float32))
    session = _CollectSession()

    asyncio.run(lc._emotion_video_loop(session, mini, stop, emotion_client, ring))

    emotion_client.process.assert_called_once()          # model kept warm
    assert any("video" in kw for kw in session.sent)     # first frame past throttle → sent


# --------------------------------------------------------------------------- #
# ER 2 scene tools                                                              #
# --------------------------------------------------------------------------- #

class _FakeBrain:
    def __init__(self, objects=None, steps=None, exc=None):
        self.objects, self.steps, self.exc = objects or [], steps or [], exc
        self.query = "unset"

    def locate(self, frame, query=None):
        if self.exc:
            raise self.exc
        self.query = query
        return self.objects

    def plan(self, frame, goal):
        if self.exc:
            raise self.exc
        return self.steps


class _ArgFC:
    def __init__(self, name, args=None):
        self.name, self.args, self.id = name, args, "id1"


_POSE = np.eye(4)


def _box():
    box = lc._LatestFrame()
    box.set(np.zeros((480, 640, 3), np.uint8), _POSE)
    return box


def test_build_config_declares_emotion_and_scene_tools():
    cfg = lc._build_config("hi")
    names = [f.name for t in cfg.tools for f in t.function_declarations]
    assert names == ["detect_emotion", "look_at_scene", "plan_task"]
    assert cfg.input_audio_transcription is not None  # the user's speech shows in the log


def test_move_arm_is_declared_only_when_the_arm_is_connected():
    cfg = lc._build_config("hi", arm=True)
    decl = {f.name: f for t in cfg.tools for f in t.function_declarations}["move_arm"]
    assert decl.parameters.properties["action"].enum == [
        "wave", "gripper_open", "gripper_close", "rest", "stop",
    ]


def test_receive_loop_sends_move_arm_to_the_arm():
    arm = MagicMock()
    arm.call.return_value = {"ok": True}
    session = _LiveSession(
        [_Msg(tool_call=_ToolCall([_ArgFC("move_arm", {"action": "wave"})]))], MagicMock()
    )
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, arm=arm))
    arm.call.assert_called_once_with("wave")
    fr = session.tool_responses[0][0]
    assert fr.name == "move_arm" and fr.response == {"ok": True}


# Reachy Mini's head→camera transform (from the SDK) and a 640x480 camera with a ~65° view.
_T_HEAD_CAM = np.array([[0, 0, 1, 0.0437], [-1, 0, 0, 0], [0, -1, 0, 0.0512], [0, 0, 0, 1]], float)


def _camera_mini():
    mini = MagicMock()
    mini.media.camera.resolution = (640, 480)
    mini.media.camera.K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    mini.media.camera.D = np.zeros(5)
    mini.T_head_cam = _T_HEAD_CAM
    return mini


def test_scene_tool_look_at_scene_returns_labels_and_gaze_from_capture_pose():
    import reachy_emotion.scene_brain as sb

    mini = _camera_mini()
    brain = _FakeBrain(objects=[
        {"label": "cup", "point": [500.0, 100.0]},
        {"label": "book", "point": [500.0, 900.0]},
    ])
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "stuff"}), brain, _box(), mini
    )
    assert result == {  # ±27° off the middle of the view, facing straight ahead
        "objects": [{"label": "cup", "where": "ahead on my left"},
                    {"label": "book", "where": "ahead on my right"}],
        "count": 2,
    }
    fn, m, pixels, pose = motion
    assert fn is sb.look_at_objects and m is mini
    assert pixels == [(64, 240), (576, 240)]
    assert pose is _POSE  # gaze is computed from the pose the frame was taken at
    assert brain.query == "stuff"
    mini.goto_target.assert_not_called()  # found in view: no search


def test_scene_tool_plan_task():
    steps = [{"step": 1, "action": "grasp", "object": "cup"}]
    result, motion = lc._scene_tool_result(
        _ArgFC("plan_task", {"goal": "clear the table"}), _FakeBrain(steps=steps), _box(), MagicMock()
    )
    assert result == {"goal": "clear the table", "steps": steps}
    assert motion is None


def test_scene_tool_errors_are_reported_not_raised():
    no_frame, _ = lc._scene_tool_result(
        _ArgFC("look_at_scene"), _FakeBrain(), lc._LatestFrame(), MagicMock()
    )
    assert "error" in no_frame
    empty_goal, _ = lc._scene_tool_result(
        _ArgFC("plan_task", {"goal": "  "}), _FakeBrain(), _box(), MagicMock()
    )
    assert "error" in empty_goal
    boom, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene"), _FakeBrain(exc=RuntimeError("403")), _box(), MagicMock()
    )
    assert "error" in boom and motion is None


class _SeqBrain(_FakeBrain):
    """ER fake answering each locate() with the next result in turn."""

    def __init__(self, results):
        super().__init__()
        self.results, self.calls = list(results), 0

    def locate(self, frame, query=None):
        self.query, self.calls = query, self.calls + 1
        return self.results.pop(0) if self.results else []


class _FakeFrames:
    """Frame box whose wait_newer() returns at once, with the head pose the search just
    turned to (from the last goto_target on *mini*)."""

    def __init__(self, mini=None):
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.mini = mini
        self.poses = []

    def snapshot(self):
        return self.frame, _POSE

    def wait_newer(self, after, timeout=2.0):
        from reachy_mini.utils import create_head_pose

        yaw = self.mini.goto_target.call_args.kwargs["body_yaw"] if self.mini else 0.0
        self.poses.append(create_head_pose(yaw=yaw, degrees=False))
        return self.frame, self.poses[-1]


def _body_yaws(mini):
    return [round(float(np.degrees(c.kwargs["body_yaw"]))) for c in mini.goto_target.call_args_list]


def test_is_person_query():
    assert lc._is_person_query("the person") and lc._is_person_query("me")
    assert not lc._is_person_query("the red cup") and not lc._is_person_query(None)
    assert not lc._is_person_query("the book next to me")  # an object, mentioning the user


def test_look_at_scene_searches_the_room_until_the_object_is_found():
    import reachy_emotion.scene_brain as sb

    mini = _camera_mini()
    cup = {"label": "cup", "point": [500.0, 500.0]}
    brain, frames = _SeqBrain([[], [], [cup]]), _FakeFrames(mini)  # not here, not at +60°, at +120°

    result, motion = lc._scene_tool_result(_ArgFC("look_at_scene", {"query": "cup"}), brain, frames, mini)

    # Found while turned to +120°: said from Reachy's default facing, not "ahead" of the view.
    assert result == {"objects": [{"label": "cup", "where": "behind me on my left"}], "count": 1}
    assert _body_yaws(mini) == [60, 120]  # stopped turning at the first hit
    fn, _, pixels, pose = motion
    assert fn is sb.look_at_objects and pixels == [(320, 240)]
    assert pose is frames.poses[-1]  # gaze from the pose that view was taken at


def test_look_at_scene_search_that_finds_nothing_returns_to_neutral():
    mini = MagicMock()
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "elephant"}), _SeqBrain([]), _FakeFrames(), mini
    )
    assert result == {"objects": [], "count": 0} and motion is None
    # full circle — crossing from +150 to -60 via the front, never across the back — then neutral
    assert _body_yaws(mini) == [60, 120, 150, 0, -60, -120, -150, 0]


def test_search_from_a_turned_robot_checks_straight_ahead_first():
    from reachy_mini.utils import create_head_pose

    class _Turned(_FakeFrames):
        def snapshot(self):
            return self.frame, create_head_pose(yaw=-150, degrees=True)  # left facing -150°

    mini = MagicMock()
    lc._scene_tool_result(_ArgFC("look_at_scene", {"query": "cup"}), _SeqBrain([]), _Turned(), mini)
    # -150 → 0 → 60 …: straight ahead first, no step across the back, -150 (already seen) skipped
    assert _body_yaws(mini) == [0, 60, 120, 150, 0, -60, -120, 0]


def test_a_failed_check_mid_search_still_ends_at_neutral():
    class _Flaky(_SeqBrain):
        def locate(self, frame, query=None):
            if self.calls:
                raise RuntimeError("ER timeout")
            return super().locate(frame, query)

    mini = MagicMock()
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "cup"}), _Flaky([]), _FakeFrames(), mini
    )
    assert "error" in result and motion is None
    assert _body_yaws(mini) == [60, 0]  # turned once, check failed, back to neutral


def test_full_scene_look_and_stop_do_not_search():
    mini = MagicMock()
    lc._scene_tool_result(_ArgFC("look_at_scene"), _SeqBrain([]), _FakeFrames(), mini)
    mini.goto_target.assert_not_called()  # no query: nothing to search for

    stop = threading.Event()
    stop.set()
    lc._scene_tool_result(_ArgFC("look_at_scene", {"query": "cup"}), _SeqBrain([]), _FakeFrames(), mini, stop)
    assert _body_yaws(mini) == [0]  # shutting down: straight back to neutral


def test_person_search_uses_faces_and_ends_facing_them(monkeypatch):
    import reachy_emotion.scene_brain as sb

    mini = _camera_mini()
    person = [{"label": "person", "point": [400.0, 750.0]}]
    faces = iter([[], [], person])  # not in the current frame, not at +60°, at +120°
    monkeypatch.setattr(lc, "_face_objects", lambda frame: next(faces))
    brain, frames = _SeqBrain([]), _FakeFrames(mini)

    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "the person"}), brain, frames, mini
    )

    # Right of the middle of the +120° view: about +102° from Reachy's default facing.
    assert result == {"objects": [{"label": "person", "where": "on my left"}], "count": 1}
    assert brain.calls == 1  # ER only for the current view; the search is by face
    assert _body_yaws(mini) == [60, 120]
    fn, _, pixel, pose = motion
    assert fn is sb.face_toward and pixel == (480, 192) and pose is frames.poses[-1]


class _LiveSession:
    """Like a real Live session: stays open after serving its turn until every
    pending tool answer has been sent (or a short timeout), then closes. Records
    the order of audio vs tool-response events."""

    def __init__(self, msgs, mini, expected_responses=1):
        self._msgs, self._served = msgs, False
        self._expected = expected_responses
        self._done = None
        self.tool_responses, self.events = [], []
        mini.media.push_audio_sample.side_effect = lambda s: self.events.append("audio")

    async def receive(self):
        if self._done is None:
            self._done = asyncio.Event()
        if self._served:
            try:
                await asyncio.wait_for(self._done.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            return
        self._served = True
        for m in self._msgs:
            yield m

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)
        self.events.append("tool")
        if len(self.tool_responses) >= self._expected:
            self._done.set()


def test_receive_loop_answers_scene_tool_in_background(monkeypatch, caplog):
    import reachy_emotion.scene_brain as sb

    monkeypatch.setattr(sb.time, "sleep", lambda s: None)  # background gaze holds are instant
    mini = MagicMock()
    mini.media.camera.resolution = (640, 480)
    brain = _FakeBrain(objects=[{"label": "cup", "point": [500.0, 500.0]}])
    fc = _ArgFC("look_at_scene", {"query": "the cup"})
    session = _LiveSession([_Msg(tool_call=_ToolCall([fc]))], mini)

    with caplog.at_level("INFO"):
        asyncio.run(lc._receive_loop(session, mini, threading.Event(), None, brain, _box()))

    fr = session.tool_responses[0][0]
    assert fr.name == "look_at_scene" and fr.id == "id1"
    assert fr.response == {"objects": [{"label": "cup", "where": "ahead"}], "count": 1}
    assert "look_at_scene({'query': 'the cup'}) →" in caplog.text  # what Gemini asked for
    assert "ER locate 'the cup' @ +0° →" in caplog.text  # ER's raw reply + where the head was


def test_slow_scene_tool_does_not_block_model_audio(monkeypatch):
    # ER 2 takes seconds; audio that arrives meanwhile must still reach the speaker
    # before the scene answer, instead of waiting behind the ER call.
    import time as _time

    class _SlowBrain(_FakeBrain):
        def plan(self, frame, goal):
            _time.sleep(0.3)
            return [{"step": 1, "action": "locate", "object": "cup"}]

    mini = MagicMock()
    audio = np.zeros(240, dtype="<i2").tobytes()
    session = _LiveSession([
        _Msg(tool_call=_ToolCall([_ArgFC("plan_task", {"goal": "tidy up"})])),
        _Msg(data=audio),
    ], mini)

    asyncio.run(lc._receive_loop(session, mini, threading.Event(), None, _SlowBrain(), _box()))

    assert session.events == ["audio", "tool"]


def test_emotion_video_loop_publishes_frame_with_its_head_pose():
    stop = threading.Event()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mini = MagicMock()
    mini.media.get_frame.return_value = frame
    mini.get_current_head_pose.return_value = _POSE
    emotion_client = MagicMock()
    emotion_client.process.side_effect = lambda f, a: stop.set()
    box = lc._LatestFrame()

    asyncio.run(lc._emotion_video_loop(_CollectSession(), mini, stop, emotion_client, lc._AudioRing(), box))

    got_frame, got_pose = box.snapshot()
    assert got_frame is frame and got_pose is _POSE


def test_grab_frame_and_pose_tolerates_missing_pose():
    mini = MagicMock()
    mini.media.get_frame.return_value = np.zeros((4, 4, 3), np.uint8)
    mini.get_current_head_pose.side_effect = AssertionError("No head pose received yet.")
    frame, pose = lc._grab_frame_and_pose(mini)
    assert frame is not None and pose is None
