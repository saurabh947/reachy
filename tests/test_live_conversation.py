"""Tests for the Gemini Live conversation helpers (W4-C1).

Covers the pure audio/format + tool logic, and drives the async receive loop
via asyncio.run (no pytest-asyncio needed). The full streaming session needs a
real API key + audio hardware and is out of scope here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

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


@pytest.fixture(autouse=True)
def _memory_in_tmp(monkeypatch, tmp_path):
    """A conversation creates its memory file: keep it out of the repo."""
    monkeypatch.setenv("REACHY_MEMORY", str(tmp_path / "reachy_memory.md"))


def test_run_live_conversation_logs_why_a_task_died(monkeypatch, caplog):
    import reachy_emotion.scene_brain as sb

    class _Connect:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *exc):
            return False

    client = MagicMock()
    client.aio.live.connect.return_value = _Connect()
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kw: client)
    monkeypatch.setattr(lc, "_load_api_key", lambda: "k")
    monkeypatch.setattr(sb, "_load_er_model", lambda: "er")
    monkeypatch.setattr("reachy_emotion.arm_client.arms_from_env", lambda: {})  # no arm

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
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kw: client)
    monkeypatch.setattr(lc, "_load_api_key", lambda: "k")
    monkeypatch.setattr(sb, "_load_er_model", lambda: "er")
    monkeypatch.setattr("reachy_emotion.arm_client.arms_from_env", lambda: {})  # no arm

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
        return {"steps": self.steps}


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


def test_every_tool_says_whether_gemini_waits_for_it():
    # Instant question: Gemini waits (can't guess the mood). Slow ones: Gemini keeps
    # talking, and is told the answer comes later.
    from google.genai import types

    cfg = lc._build_config("hi", arms=["follower"])
    decl = {f.name: f for t in cfg.tools for f in t.function_declarations}
    assert decl["detect_emotion"].behavior == types.Behavior.BLOCKING
    for name in ("look_at_scene", "plan_task", "move_arm"):
        assert decl[name].behavior == types.Behavior.NON_BLOCKING
    for name in ("look_at_scene", "plan_task"):
        assert "arrives later" in decl[name].description
    assert "you do NOT know it" in lc._SCENE_TOOLS_PROMPT
    # Run 4: the tool said "on my left" and Gemini told the user "on your left" (wrong side)
    assert "never turn them into the user's left or right" in lc._SCENE_TOOLS_PROMPT
    assert "never as the user's left or right" in decl["look_at_scene"].description
    # Run 4: "open and close your gripper" made only one move_arm call
    assert "call move_arm once for each, in order" in lc._arm_prompt(["follower"])


def test_move_arm_is_declared_only_when_an_arm_is_connected():
    cfg = lc._build_config("hi", arms=["follower"])
    decl = {f.name: f for t in cfg.tools for f in t.function_declarations}["move_arm"]
    assert decl.parameters.properties["action"].enum == [
        "wave", "gripper_open", "gripper_close", "rest", "stop",
    ]
    assert "arm" not in decl.parameters.properties  # one arm: nothing to choose
    assert "one robot arm" in lc._arm_prompt(["follower"])


def test_two_arms_add_an_arm_choice():
    cfg = lc._build_config("hi", arms=["left", "right"])
    decl = {f.name: f for t in cfg.tools for f in t.function_declarations}["move_arm"]
    assert decl.parameters.properties["arm"].enum == ["left", "right"]
    assert decl.parameters.required == ["action", "arm"]
    assert "2 robot arms beside you: left, right" in lc._arm_prompt(["left", "right"])


def _fake_arm(result):
    """An ArmClient stand-in whose submit() answers at once with *result*."""
    from concurrent.futures import Future

    def submit(action):
        future = Future()
        future.set_result(result)
        return future

    arm = MagicMock()
    arm.submit.side_effect = submit
    return arm


def _arm_reply(result, args=None, arms=None):
    """Run one move_arm call through the receive loop; returns (arm mocks, FunctionResponse)."""
    if arms is None:
        arms = {"follower": _fake_arm(result)}
    session = _LiveSession(
        [_Msg(tool_call=_ToolCall([_ArgFC("move_arm", args or {"action": "wave"})]))], MagicMock()
    )
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, arms=arms))
    return arms, session.tool_responses[0][0]


def test_a_move_that_went_through_is_answered_silently():
    # Gemini already said what it's doing; a result that prompted a reply made it repeat itself.
    from google.genai import types

    arms, fr = _arm_reply({"ok": True})
    arms["follower"].submit.assert_called_once_with("wave")
    assert fr.name == "move_arm" and fr.response == {"ok": True}
    assert fr.scheduling == types.FunctionResponseScheduling.SILENT


def test_moves_in_one_turn_reach_the_arm_in_order():
    from concurrent.futures import Future

    sent, pending = [], []

    def submit(action):  # answers come later, like a queued move
        sent.append(action)
        pending.append(Future())
        return pending[-1]

    arm = MagicMock()
    arm.submit.side_effect = submit
    calls = [_ArgFC("move_arm", {"action": a}) for a in ("gripper_open", "gripper_close", "rest")]
    session = _LiveSession([_Msg(tool_call=_ToolCall(calls))], MagicMock(), expected_responses=3)

    async def main():
        loop = asyncio.create_task(lc._receive_loop(session, MagicMock(), threading.Event(), None, arms={"f": arm}))
        await asyncio.sleep(0.05)
        for future in pending:
            future.set_result({"ok": True})
        await loop

    asyncio.run(main())
    assert sent == ["gripper_open", "gripper_close", "rest"]  # all sent before any finished


def test_a_cancelled_move_is_silent_and_a_failure_prompts_a_reply():
    from google.genai import types

    _, cancelled = _arm_reply({"ok": False, "error": "cancelled: stop", "cancelled": True})
    assert cancelled.scheduling == types.FunctionResponseScheduling.SILENT
    _, failed = _arm_reply({"ok": False, "error": "motor fault: wrist_roll overload; arm frozen in place"})
    assert failed.scheduling == types.FunctionResponseScheduling.WHEN_IDLE  # Gemini tells the user


def test_move_arm_goes_to_the_named_arm():
    left, right = _fake_arm({"ok": True}), _fake_arm({"ok": True})
    _arm_reply(None, {"action": "wave", "arm": "right"}, arms={"left": left, "right": right})
    right.submit.assert_called_once_with("wave")
    left.submit.assert_not_called()
    _, fr = _arm_reply(None, {"action": "wave", "arm": "middle"}, arms={"left": left, "right": right})
    assert fr.response == {"ok": False, "error": "no arm named 'middle' (have: left, right)"}
    _, fr = _arm_reply(None, arms={})
    assert fr.response == {"ok": False, "error": "the arm is not connected"}


class _NoticeSession:
    def __init__(self):
        self.notices = []

    async def send_client_content(self, turns, turn_complete):
        self.notices.append((turns.parts[0].text, turn_complete))


def test_a_fault_while_idle_is_told_to_gemini(caplog):
    session = _NoticeSession()
    event = {"event": "fault", "arm": "follower", "faults": {"gripper": ["overload"]},
             "interrupted": False, "cancelled": 0, "detail": "arm frozen in place"}
    with caplog.at_level("WARNING"):
        asyncio.run(lc._on_arm_event(session, event))
    (text, complete), = session.notices
    assert "gripper overload" in text and "arm frozen in place" in text and complete
    assert "Arm follower fault: gripper overload" in caplog.text


def test_an_arm_that_stops_responding_is_told_to_gemini(caplog):
    session = _NoticeSession()
    with caplog.at_level("WARNING"):
        asyncio.run(lc._on_arm_event(session, {"event": "unreachable", "arm": "follower",
                                               "detail": "There is no status packet!"}))
    (text, _), = session.notices
    assert "stopped responding" in text and "Arm follower not responding" in caplog.text


def test_a_fault_that_interrupted_a_move_is_not_told_twice():
    # The interrupted move_arm call's error already carries it to Gemini.
    session = _NoticeSession()
    event = {"event": "fault", "arm": "follower", "faults": {"wrist_roll": ["overload"]},
             "interrupted": True, "cancelled": 2, "detail": "arm frozen in place"}
    asyncio.run(lc._on_arm_event(session, event))
    assert session.notices == []


def test_arm_events_from_the_reader_thread_reach_the_session():
    session = _NoticeSession()
    event = {"event": "fault", "arm": "follower", "faults": {"gripper": ["overheat"]},
             "interrupted": False, "cancelled": 0, "detail": "arm frozen in place"}

    async def main():
        tasks = set()
        handle = lc._arm_event_handler(session, asyncio.get_running_loop(), tasks)
        threading.Thread(target=handle, args=(event,)).start()  # like ArmClient's reader
        for _ in range(100):
            if session.notices:
                break
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert len(session.notices) == 1 and "gripper overheat" in session.notices[0][0]


def test_an_idle_fault_notice_waits_for_gemini_to_finish_its_sentence():
    session, speaking = _NoticeSession(), lc._Speaking()
    speaking.heard()  # mid-reply
    event = {"event": "fault", "arm": "follower", "faults": {"gripper": ["overload"]},
             "interrupted": False, "cancelled": 0, "detail": "arm frozen in place"}

    async def main():
        notice = asyncio.create_task(lc._on_arm_event(session, event, speaking))
        await asyncio.sleep(0.3)
        assert session.notices == []  # sending now would cut Gemini off
        speaking.done()
        await notice

    asyncio.run(main())
    assert len(session.notices) == 1


def test_receive_loop_tracks_when_gemini_is_speaking():
    from types import SimpleNamespace

    speaking, seen, mini = lc._Speaking(), [], MagicMock()
    mini.media.push_audio_sample.side_effect = lambda s: seen.append(speaking.active)
    session = _FakeSession([
        _Msg(data=np.zeros(240, dtype="<i2").tobytes()),
        _Msg(server_content=SimpleNamespace(turn_complete=True)),
    ])
    asyncio.run(lc._receive_loop(session, mini, threading.Event(), None, speaking=speaking))
    assert seen == [True] and not speaking.active


class _CancellingSession:
    """Sends one tool call, then the server's cancellation of it (the user talked over
    it); stays open briefly so the tool can finish, then closes."""

    def __init__(self, fc, before_cancel=None, after_cancel=None):
        self.fc, self.before, self.after = fc, before_cancel, after_cancel
        self.served, self.tool_responses = False, []

    async def receive(self):
        from types import SimpleNamespace

        if self.served:
            await asyncio.sleep(0.5)
            return
        self.served = True
        yield _Msg(tool_call=_ToolCall([self.fc]))
        if self.before:
            await asyncio.to_thread(self.before)
        yield SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=[self.fc.id]))
        if self.after:
            self.after()

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)


def test_a_cancelled_room_search_stops_and_gets_no_answer(caplog):
    mini, started = _camera_mini(), threading.Event()

    class _SlowSeq(_SeqBrain):
        def locate(self, frame, query=None):
            started.set()
            threading.Event().wait(0.05)
            return super().locate(frame, query)

    brain = _SlowSeq([])  # never finds it
    session = _CancellingSession(_ArgFC("look_at_scene", {"query": "fridge"}), lambda: started.wait(2))
    with caplog.at_level("INFO"):
        asyncio.run(lc._receive_loop(session, mini, threading.Event(), None, brain, _FakeFrames(mini)))
    assert session.tool_responses == []
    assert _body_yaws(mini) == [0]  # no turning round the room: straight back to neutral
    assert "Gemini cancelled tool call(s): id1" in caplog.text


def test_a_cancelled_search_that_already_found_it_still_turns_back(monkeypatch):
    import reachy_emotion.scene_brain as sb

    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    mini, at_60 = _camera_mini(), threading.Event()
    cup = {"label": "cup", "point": [500.0, 500.0]}

    class _FoundAt60(_SeqBrain):
        def locate(self, frame, query=None):
            if self.calls == 2:  # the check at +60° is in flight when the user talks over it
                at_60.set()
                threading.Event().wait(0.2)
            return super().locate(frame, query)

    session = _CancellingSession(_ArgFC("look_at_scene", {"query": "cup"}), lambda: at_60.wait(2))
    asyncio.run(lc._receive_loop(session, mini, threading.Event(), None, _FoundAt60([[], [], [cup]]), _FakeFrames(mini)))
    assert session.tool_responses == []
    assert _body_yaws(mini) == [0, 60, 0]  # table glance, found it at +60°, answer dropped, back to neutral


def test_a_cancelled_move_that_faults_is_still_told_to_gemini():
    from concurrent.futures import Future

    future, arm = Future(), MagicMock()
    arm.submit.return_value = future
    session = _CancellingSession(
        _ArgFC("move_arm", {"action": "wave"}),
        after_cancel=lambda: future.set_result(
            {"ok": False, "error": "motor fault: wrist_roll overload; arm frozen in place"}),
    )
    notices = []

    async def send_client_content(turns, turn_complete):
        notices.append(turns.parts[0].text)

    session.send_client_content = send_client_content
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, arms={"follower": arm}))
    assert session.tool_responses == []  # the call id is gone...
    assert len(notices) == 1 and "wrist_roll overload" in notices[0]  # ...but the fault isn't


def test_the_notice_waits_for_queued_speech_to_finish_playing():
    # Live sends audio faster than real time: the turn can look over while the speaker
    # still has half a second to play.
    speaking = lc._Speaking()
    speaking.heard(8000)  # 0.5 s at 16 kHz
    speaking.done()

    async def timed():
        start = time.monotonic()
        await speaking.wait_quiet()
        return time.monotonic() - start

    assert asyncio.run(timed()) >= 0.4


def test_memory_goes_into_the_prompt_with_a_remember_tool(tmp_path):
    from google.genai import types

    from reachy_emotion.memory import Memory

    memory = Memory(tmp_path / "m.md")
    memory.remember("user", "name", "Sam")
    assert "- name: Sam" in lc._memory_prompt(memory) and "## MEMORY" in lc._memory_prompt(memory)
    decl = {f.name: f for t in lc._build_config("hi", memory=True).tools for f in t.function_declarations}
    assert decl["remember"].behavior == types.Behavior.BLOCKING
    assert decl["remember"].parameters.properties["section"].enum == ["places", "user", "reachy", "notes"]
    assert "remember" not in {f.name for t in lc._build_config("hi").tools for f in t.function_declarations}
    # Answer "where is X" from memory instead of always searching the room again.
    assert "answer from memory" in lc._SCENE_TOOLS_PROMPT
    assert "even if you already know" not in lc._SCENE_TOOLS_PROMPT
    assert "say where things are" not in lc._SCENE_TOOLS_PROMPT  # no ALWAYS rule against it


def test_receive_loop_saves_what_gemini_asks_to_remember(tmp_path):
    from reachy_emotion.memory import Memory

    memory = Memory(tmp_path / "m.md")
    fc = _ArgFC("remember", {"section": "user", "item": "name", "detail": "Sam"})
    session = _FakeSession([_Msg(tool_call=_ToolCall([fc]))])
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, memory=memory))
    assert session.tool_responses[0][0].response == {"ok": True, "saved": "name: Sam"}
    assert memory.recall("user", "name") == "Sam" and "- name: Sam" in memory.path.read_text()


def test_look_at_scene_keeps_places_in_memory(tmp_path):
    from reachy_emotion.memory import Memory

    memory = Memory(tmp_path / "m.md")
    mini, cup = _camera_mini(), {"label": "cup", "point": [500.0, 500.0]}
    lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "the cup"}), _SeqBrain([[], [], [], [cup]]), _FakeFrames(mini), mini,
        memory=memory,
    )
    assert memory.recall("places", "cup").startswith("behind me on my left (+120°) · seen ")

    # Searched the whole room and it's gone: memory no longer says where it was, and Live is told.
    result, _ = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "cup"}), _SeqBrain([]), _FakeFrames(mini), mini, memory=memory
    )
    assert memory.recall("places", "cup") is None
    assert result == {"objects": [], "count": 0, "searched": "whole room", "memory": "forgot where cup was last seen"}


def test_people_pets_and_groups_arent_kept_as_places(tmp_path, monkeypatch):
    from reachy_emotion.memory import Memory

    memory = Memory(tmp_path / "m.md")
    mini = _camera_mini()
    person = [{"label": "person", "point": [500.0, 500.0]}]
    monkeypatch.setattr(lc, "_face_objects", lambda frame: person)
    lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "the person"}), _SeqBrain([]), _FakeFrames(mini), mini, memory=memory
    )
    for query, found in (
        ("Mom", [{"label": "woman", "point": [500.0, 500.0]}]),
        ("the dog", [{"label": "small dog", "point": [500.0, 500.0]}]),
        ("my keys and my phone", [{"label": "keys", "point": [500.0, 300.0]}, {"label": "phone", "point": [500.0, 700.0]}]),
    ):
        lc._scene_tool_result(_ArgFC("look_at_scene", {"query": query}), _SeqBrain([found]), _FakeFrames(mini), mini, memory=memory)
    assert memory.facts() == 0


def test_a_search_cut_short_doesnt_forget_a_place(tmp_path):
    from reachy_emotion.memory import Memory

    memory = Memory(tmp_path / "m.md")
    memory.note_place("sofa", "on my right", -90)
    mini = _camera_mini()
    stopped = threading.Event()
    stopped.set()  # the user talked over it
    lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "sofa"}), _SeqBrain([]), _FakeFrames(mini), mini, stopped, memory
    )
    result, _ = lc._scene_tool_result(  # the camera stalled after the first heading
        _ArgFC("look_at_scene", {"query": "sofa"}), _SeqBrain([]), _StalledFrames(mini), mini, memory=memory
    )
    assert result == {"objects": [], "count": 0, "searched": "part of the room"}

    class _Garbled(_SeqBrain):  # ER's reply at the second heading can't be read
        def locate(self, frame, query=None):
            if self.calls == 1:
                raise RuntimeError("ER 2 didn't answer in JSON")
            return super().locate(frame, query)

    result, _ = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "sofa"}), _Garbled([]), _FakeFrames(mini), mini, memory=memory
    )
    assert "error" in result  # not "not found"
    assert memory.recall("places", "sofa").startswith("on my right")


def test_a_cancelled_move_still_runs_but_gets_no_answer():
    from concurrent.futures import Future

    future, arm = Future(), MagicMock()
    arm.submit.return_value = future
    session = _CancellingSession(
        _ArgFC("move_arm", {"action": "wave"}), after_cancel=lambda: future.set_result({"ok": True})
    )
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, arms={"follower": arm}))
    arm.submit.assert_called_once_with("wave")  # only move_arm(stop) stops the arm
    assert session.tool_responses == []


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

    class _NoBrain(_FakeBrain):
        def plan(self, frame, goal):
            return {"feasible": False, "reason": "out of the arm's reach"}

    result, _ = lc._scene_tool_result(_ArgFC("plan_task", {"goal": "hand me the cup"}), _NoBrain(), _box(), MagicMock())
    assert result == {"goal": "hand me the cup", "feasible": False, "reason": "out of the arm's reach"}


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

        # the head pose just commanded (turned and tilted), as the daemon would report it
        self.poses.append(self.mini.goto_target.call_args.kwargs["head"] if self.mini else create_head_pose())
        return self.frame, self.poses[-1]


class _StalledFrames(_FakeFrames):
    """The camera stops sending frames once the search starts turning."""

    def wait_newer(self, after, timeout=2.0):
        return None, None


def _body_yaws(mini):
    return [round(float(np.degrees(c.kwargs["body_yaw"]))) for c in mini.goto_target.call_args_list]


def _tilts(mini):
    """Head tilt of each move, in degrees (+ = down)."""
    from scipy.spatial.transform import Rotation

    return [
        round(float(Rotation.from_matrix(c.kwargs["head"][:3, :3]).as_euler("xyz", degrees=True)[1]))
        for c in mini.goto_target.call_args_list
    ]


def test_is_person_query():
    assert lc._is_person_query("the person") and lc._is_person_query("me")
    assert not lc._is_person_query("the red cup") and not lc._is_person_query(None)
    assert not lc._is_person_query("the book next to me")  # an object, mentioning the user


def test_look_at_scene_searches_the_room_until_the_object_is_found():
    import reachy_emotion.scene_brain as sb

    mini = _camera_mini()
    cup = {"label": "cup", "point": [500.0, 500.0]}
    brain, frames = _SeqBrain([[], [], [], [cup]]), _FakeFrames(mini)  # not here, below, at +60°; at +120°

    result, motion = lc._scene_tool_result(_ArgFC("look_at_scene", {"query": "cup"}), brain, frames, mini)

    # Found while turned to +120°: said from Reachy's default facing, not "ahead" of the view.
    assert result == {"objects": [{"label": "cup", "where": "behind me on my left"}], "count": 1}
    assert _body_yaws(mini) == [0, 60, 120]  # glanced down, then turned; stopped at the first hit
    assert _tilts(mini) == [30, 0, 0]
    fn, _, pixels, pose = motion
    assert fn is sb.look_at_objects and pixels == [(320, 240)]
    assert pose is frames.poses[-1]  # gaze from the pose that view was taken at


def test_a_search_turns_from_where_the_robot_is_now():
    from reachy_mini.utils import create_head_pose

    mini = MagicMock()  # an earlier search left it facing behind, on its right
    mini.get_current_head_pose.return_value = create_head_pose(yaw=-150, degrees=True)
    lc._search_room(lambda f, p: [], mini, _FakeFrames(mini), None, 0.0)
    assert _body_yaws(mini) == [0, 60, 120, 150, 0, -60, -120, -150, 0]  # via the front, not across the back


def test_look_at_me_falls_back_to_faces_when_er_fails(monkeypatch):
    person = [{"label": "person", "point": [500.0, 500.0]}]
    monkeypatch.setattr(lc, "_face_objects", lambda frame: person)
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "the person"}), _FakeBrain(exc=RuntimeError("not JSON")), _box(),
        _camera_mini(),
    )
    assert result["count"] == 1 and motion is not None


def test_look_at_scene_glances_down_at_the_table_first():
    import reachy_emotion.scene_brain as sb

    mini = _camera_mini()
    eraser = {"label": "black eraser", "point": [500.0, 500.0]}
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "eraser"}), _SeqBrain([[], [eraser]]), _FakeFrames(mini), mini
    )
    assert result == {"objects": [{"label": "black eraser", "where": "ahead, down low"}], "count": 1}
    assert _body_yaws(mini) == [0] and _tilts(mini) == [30]  # head tilted down; no room search
    assert motion[0] is sb.look_at_objects


def test_a_directed_look_looks_there_only():
    mini = _camera_mini()
    brain = _SeqBrain([])
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "eraser", "facing": "on my left", "tilt": "down"}), brain,
        _FakeFrames(mini), mini,
    )
    assert result == {"objects": [], "count": 0, "searched": "that view"} and motion is None
    assert brain.calls == 1  # not the current view, no room search
    assert _body_yaws(mini) == [90, 0] and _tilts(mini) == [30, 0]  # looked down on its left, then back

    mini, cup = _camera_mini(), {"label": "cup", "point": [500.0, 500.0]}
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "cup", "tilt": "up"}), _SeqBrain([[cup]]), _FakeFrames(mini), mini
    )
    assert result == {"objects": [{"label": "cup", "where": "ahead, up high"}], "count": 1}
    assert _tilts(mini) == [-25] and motion is not None


def test_look_at_scene_search_that_finds_nothing_returns_to_neutral():
    mini = MagicMock()
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "elephant"}), _SeqBrain([]), _FakeFrames(), mini
    )
    assert result == {"objects": [], "count": 0, "searched": "whole room"} and motion is None
    # the table in front, then a full circle — crossing from +150 to -60 via the front, never across the back — then neutral
    assert _body_yaws(mini) == [0, 60, 120, 150, 0, -60, -120, -150, 0]


def test_search_from_a_turned_robot_checks_straight_ahead_first():
    from reachy_mini.utils import create_head_pose

    class _Turned(_FakeFrames):
        def snapshot(self):
            return self.frame, create_head_pose(yaw=-150, degrees=True)  # left facing -150°

    mini = MagicMock()
    lc._scene_tool_result(_ArgFC("look_at_scene", {"query": "cup"}), _SeqBrain([]), _Turned(), mini)
    # table glance at -150, then 0 → 60 …: straight ahead first, no step across the back,
    # -150 (already seen) skipped
    assert _body_yaws(mini) == [-150, 0, 60, 120, 150, 0, -60, -120, 0]


def test_a_failed_check_mid_search_still_ends_at_neutral():
    class _Flaky(_SeqBrain):
        def locate(self, frame, query=None):
            if self.calls >= 2:  # the view and the table glance are fine; +60° fails
                raise RuntimeError("ER timeout")
            return super().locate(frame, query)

    mini = MagicMock()
    result, motion = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "cup"}), _Flaky([]), _FakeFrames(), mini
    )
    assert "error" in result and motion is None
    assert _body_yaws(mini) == [0, 60, 0]  # glanced down, turned once, check failed, back to neutral


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
    assert fr.scheduling == "WHEN_IDLE"  # spoken once Gemini finishes its sentence
    assert "look_at_scene({'query': 'the cup'}) →" in caplog.text  # what Gemini asked for
    assert "ER locate 'the cup' @ +0° →" in caplog.text  # ER's raw reply + where the head was


def test_slow_scene_tool_does_not_block_model_audio(monkeypatch):
    # ER 2 takes seconds; audio that arrives meanwhile must still reach the speaker
    # before the scene answer, instead of waiting behind the ER call.
    import time as _time

    class _SlowBrain(_FakeBrain):
        def plan(self, frame, goal):
            _time.sleep(0.3)
            return {"steps": [{"step": 1, "action": "locate", "object": "cup"}]}

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


# --------------------------------------------------------------------------- #
# Arm tasks                                                                     #
# --------------------------------------------------------------------------- #

_MOUNT = {"x": 0.10, "y": 0.20, "yaw_deg": 0.0}


class _IdFC:
    def __init__(self, id, name, args=None):
        self.id, self.name, self.args = id, name, args


def test_with_arm_tasks_on_arm_task_replaces_plan_task():
    from google.genai import types

    decl = {f.name: f for t in lc._build_config("hi", arms=["follower"], arm_tasks=True).tools
            for f in t.function_declarations}
    assert "plan_task" not in decl and "move_arm" in decl
    task = decl["arm_task"]
    assert task.behavior == types.Behavior.NON_BLOCKING
    assert task.parameters.required == ["goal"]
    assert "only when the user asks" in task.description and "arrives later" in task.description
    assert "arm_task" not in {f.name for t in lc._build_config("hi").tools for f in t.function_declarations}


def test_the_prompts_dont_contradict_themselves_in_either_mode():
    off = lc._scene_prompt(False) + lc._arm_prompt(["follower"])
    on = lc._scene_prompt(True) + lc._arm_prompt(["follower"], arm_tasks=True)
    assert lc._scene_prompt(False) == lc._SCENE_TOOLS_PROMPT
    # Off: physical tasks still go to plan_task. On the robot, "can't carry out plans yet,
    # so never claim you did it" made Gemini refuse the eraser task without calling it.
    assert "ALWAYS call plan_task, then describe the plan and say you can't carry it out yet" in off
    assert "call plan_task, describe the plan, and say you can't carry it out yet" in off
    assert "never claim" not in off and "arm_task" not in off
    # On: arm_task, only when asked; nothing says the arm can't do it.
    assert "plan_task" not in on and "can't carry" not in on and "can't pick things up yet" not in on
    assert "call arm_task — only when the user asks; never on your own" in on
    assert "say you're on it; never say it's done" in on
    assert "move_arm only accepts stop" in on
    for prompt in (off, on):  # the robot said "I don't see an eraser" without looking
        assert "Before saying something is or isn't there, call look_at_scene" in prompt
        assert "you do NOT know it" in prompt


def test_arm_tasks_need_the_mode_an_arm_that_can_grasp_and_the_mount(monkeypatch):
    import reachy_emotion.arm_task as at

    arm = MagicMock(actions=["rest", "gripper_open", "hover", "grasp", "place", "stop"])
    old = MagicMock(actions=["rest", "wave", "stop"])  # no IK, mount or ready pose on the server
    monkeypatch.setattr(at, "arm_tasks_mode", lambda: "hover")
    monkeypatch.setattr(at, "arm_mount_from_env", lambda: _MOUNT)
    tasks = lc._arm_tasks_for({"follower": arm})
    assert tasks.arm is arm and tasks.mode == "hover" and tasks.mount == _MOUNT
    assert not tasks.busy.is_set()
    assert lc._arm_tasks_for({}) is None
    assert lc._arm_tasks_for({"follower": old}) is None
    monkeypatch.setattr(at, "arm_mount_from_env", lambda: None)
    assert lc._arm_tasks_for({"follower": arm}) is None
    monkeypatch.setattr(at, "arm_mount_from_env", lambda: _MOUNT)
    monkeypatch.setattr(at, "arm_tasks_mode", lambda: "off")
    assert lc._arm_tasks_for({"follower": arm}) is None


def test_while_an_arm_task_runs_only_stop_gets_through_and_it_ends_the_task(monkeypatch):
    from google.genai import types

    seen = {}

    def task(goal, tasks, brain, frame_box, mini, session_stop=None):
        seen["goal"] = goal
        seen["stopped"] = tasks.stop.wait(5)  # until move_arm(stop)
        return {"outcome": "stopped", "say": "I stopped.", "holding": None, "did": []}

    monkeypatch.setattr(lc, "_run_arm_task", task)
    arm = _fake_arm({"ok": True})
    tasks = lc._ArmTasks(arm, "on", _MOUNT)
    session = _LiveSession([
        _Msg(tool_call=_ToolCall([_IdFC("t1", "arm_task", {"goal": "pick up the pink eraser"})])),
        _Msg(tool_call=_ToolCall([_IdFC("w1", "move_arm", {"action": "wave"})])),
        _Msg(tool_call=_ToolCall([_IdFC("l1", "look_at_scene", {"query": "cup"}),
                                  _IdFC("t2", "arm_task", {"goal": "put it back"})])),
        _Msg(tool_call=_ToolCall([_IdFC("s1", "move_arm", {"action": "stop"})])),
    ], MagicMock(), expected_responses=5)
    asyncio.run(lc._receive_loop(
        session, MagicMock(), threading.Event(), None, _FakeBrain(), _box(), arms={"follower": arm}, arm_tasks=tasks,
    ))
    replies = {fr.id: fr for batch in session.tool_responses for fr in batch}
    assert replies["w1"].response == {"ok": False, "error": "busy with an arm task"}
    assert replies["l1"].response == {"error": "busy with an arm task; ask again after it"}
    assert replies["t2"].response == {"error": "already doing an arm task"}
    assert replies["s1"].response == {"ok": True}
    arm.submit.assert_called_once_with("stop")  # the wave never reached the arm
    assert seen == {"goal": "pick up the pink eraser", "stopped": True}
    assert replies["t1"].response["outcome"] == "stopped"
    for fr in (replies["t1"], replies["w1"], replies["l1"]):
        assert fr.scheduling == types.FunctionResponseScheduling.WHEN_IDLE
    assert not tasks.busy.is_set()  # the next task can run


def test_a_stop_or_wave_made_before_an_arm_task_in_the_same_turn_goes_to_the_arm(monkeypatch):
    # "Stop that and pick up the pink eraser": [move_arm(stop), arm_task] in one tool call.
    # The stop is for the wave, not for the task that starts after it.
    seen = []

    def task(goal, tasks, *a, **k):
        threading.Event().wait(0.2)  # the move_arm call made before it has run by now
        seen.append(tasks.stop.is_set())
        return {"outcome": "done", "say": "Done.", "holding": None, "did": []}

    monkeypatch.setattr(lc, "_run_arm_task", task)
    for action in ("stop", "wave"):
        arm = _fake_arm({"ok": True})
        tasks = lc._ArmTasks(arm, "on", _MOUNT)
        session = _LiveSession([_Msg(tool_call=_ToolCall([
            _IdFC("m1", "move_arm", {"action": action}), _IdFC("t1", "arm_task", {"goal": "pick up the pink eraser"}),
        ]))], MagicMock(), expected_responses=2)
        asyncio.run(lc._receive_loop(
            session, MagicMock(), threading.Event(), None, _FakeBrain(), _box(), arms={"follower": arm}, arm_tasks=tasks,
        ))
        replies = {fr.id: fr for batch in session.tool_responses for fr in batch}
        assert replies["m1"].response == {"ok": True}  # it went to the arm ...
        arm.submit.assert_called_once_with(action)
        assert replies["t1"].response["outcome"] == "done"
    assert seen == [False, False]  # ... and didn't stop the new task


def test_a_look_in_the_same_turn_as_a_new_arm_task_is_refused(monkeypatch):
    # It would wait for the head until the task ends, then look with stale data.
    done = {"outcome": "done", "say": "Done.", "holding": None, "did": []}
    monkeypatch.setattr(lc, "_run_arm_task", lambda *a, **k: done)
    tasks = lc._ArmTasks(_fake_arm({"ok": True}), "on", _MOUNT)
    session = _LiveSession([_Msg(tool_call=_ToolCall([
        _IdFC("l1", "look_at_scene", {"query": "pink eraser"}), _IdFC("t1", "arm_task", {"goal": "pick up the pink eraser"}),
    ]))], MagicMock(), expected_responses=2)
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, _FakeBrain(), _box(), arm_tasks=tasks))
    replies = {fr.id: fr for batch in session.tool_responses for fr in batch}
    assert replies["l1"].response == {"error": "busy with an arm task; ask again after it"}
    assert replies["t1"].response == done


def test_a_fault_or_a_lost_arm_during_an_arm_task_stops_it():
    # E.g. while ER thinks about the next step: the task must not send it to a tripped arm.
    arm = MagicMock()
    arm.name = "follower"
    tasks = lc._ArmTasks(arm, "on", _MOUNT)
    loop = MagicMock()  # the notice to Gemini is covered above
    handle = lc._arm_event_handler(_NoticeSession(), loop, set(), None, tasks)
    fault = {"event": "fault", "arm": "follower", "faults": {"shoulder_lift": ["overload"]},
             "interrupted": False, "cancelled": 0, "detail": "arm frozen in place"}
    handle(fault)
    assert not tasks.stop.is_set()  # no task running
    tasks.busy.set()
    handle({"event": "dropped", "arm": "follower", "label": "pink eraser"})
    handle({**fault, "arm": "right"})
    assert not tasks.stop.is_set()  # not a fault, or another arm
    handle(fault)
    assert tasks.stop.is_set()
    tasks.stop.clear()
    handle({"event": "unreachable", "arm": "follower", "detail": "There is no status packet!"})
    assert tasks.stop.is_set()
    assert loop.call_soon_threadsafe.call_count == 5  # each still goes on to Gemini


def test_an_arm_task_result_is_spoken_when_gemini_is_idle(monkeypatch):
    from google.genai import types

    done = {"outcome": "done", "say": "It's in the cup.", "holding": None, "did": ["picked up the pink eraser"]}
    monkeypatch.setattr(lc, "_run_arm_task", lambda *a, **k: done)
    session = _LiveSession(
        [_Msg(tool_call=_ToolCall([_IdFC("t1", "arm_task", {"goal": "put the pink eraser in the white cup"})]))],
        MagicMock(),
    )
    tasks = lc._ArmTasks(_fake_arm({"ok": True}), "on", _MOUNT)
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, _FakeBrain(), _box(), arm_tasks=tasks))
    (fr,), = session.tool_responses
    assert fr.name == "arm_task" and fr.response == done
    assert fr.scheduling == types.FunctionResponseScheduling.WHEN_IDLE


def test_a_barge_in_doesnt_stop_an_arm_task_and_its_result_comes_as_a_notice(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def task(goal, tasks, *a, **k):
        started.set()
        release.wait(5)
        return {"outcome": "done", "say": "I picked it up.", "holding": "pink eraser", "did": []}

    monkeypatch.setattr(lc, "_run_arm_task", task)
    tasks = lc._ArmTasks(_fake_arm({"ok": True}), "on", _MOUNT)
    session = _CancellingSession(
        _IdFC("t1", "arm_task", {"goal": "pick up the pink eraser"}),
        before_cancel=lambda: started.wait(2), after_cancel=release.set,
    )
    notices = []

    async def send_client_content(turns, turn_complete):
        notices.append(turns.parts[0].text)

    session.send_client_content = send_client_content
    asyncio.run(lc._receive_loop(session, MagicMock(), threading.Event(), None, _FakeBrain(), _box(), arm_tasks=tasks))
    assert session.tool_responses == []  # the call id is gone ...
    assert len(notices) == 1 and "Your arm task (pick up the pink eraser) ended: done. I picked it up." in notices[0]
    assert not tasks.stop.is_set()  # ... and the user talking didn't stop the task


def test_an_arm_task_holds_the_head_and_ends_at_neutral(monkeypatch):
    import reachy_emotion.arm_task as at

    seen = {}

    class _Task:
        def __init__(self, client, model, arm, mini, turn, grab, mount, mode):
            seen["made"] = (client, model, arm, mount, mode)
            self.turn, self.grab = turn, grab

        def run(self, goal, stop):
            seen["locked"] = lc._MOTION_LOCK.locked()  # no gaze tour or search can move the head
            self.turn(45.0, 30)  # the task's own look goes straight through (no second lock)
            seen["frame"] = self.grab()[0] is not None
            seen["stop"] = stop.is_set()
            return {"outcome": "done", "say": "Done.", "holding": None, "did": []}

    monkeypatch.setattr(at, "ArmTask", _Task)
    mini, brain = _camera_mini(), MagicMock(model="er")
    tasks = lc._ArmTasks("arm", "on", _MOUNT)
    session_over = threading.Event()
    session_over.set()
    result = lc._run_arm_task("pick up the eraser", tasks, brain, _FakeFrames(mini), mini, session_over)
    assert result["outcome"] == "done" and not lc._MOTION_LOCK.locked()
    assert seen == {"made": (brain.client, "er", "arm", _MOUNT, "on"), "locked": True, "frame": True, "stop": True}
    assert _body_yaws(mini) == [45, 0] and _tilts(mini) == [30, 0]  # its look, then back to neutral


def test_a_directed_look_that_finds_nothing_goes_back_to_neutral_before_letting_go_of_the_head():
    mini, locked = _camera_mini(), []
    mini.goto_target.side_effect = lambda **kw: locked.append(lc._MOTION_LOCK.locked())
    result, _ = lc._scene_tool_result(
        _ArgFC("look_at_scene", {"query": "eraser", "facing": "on my left", "tilt": "down"}), _SeqBrain([]),
        _FakeFrames(mini), mini,
    )
    assert result["searched"] == "that view"
    assert _body_yaws(mini) == [90, 0] and locked == [True, True]  # an arm task can't slip in between


def test_a_drop_while_holding_is_told_to_gemini():
    session = _NoticeSession()
    asyncio.run(lc._on_arm_event(session, {"event": "dropped", "arm": "follower", "label": "pink eraser"}))
    (text, complete), = session.notices
    assert "Your arm (follower) dropped the pink eraser" in text and complete


def test_a_session_with_arm_tasks_on_offers_arm_task(monkeypatch):
    import reachy_emotion.arm_task as at
    import reachy_emotion.scene_brain as sb

    seen = {}

    class _Connect:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *exc):
            return False

    def connect(model, config):
        seen["config"] = config
        return _Connect()

    client = MagicMock()
    client.aio.live.connect.side_effect = connect
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kw: client)
    monkeypatch.setattr(lc, "_load_api_key", lambda: "k")
    monkeypatch.setattr(sb, "_load_er_model", lambda: "er")
    arm = MagicMock(actions=["rest", "grasp", "place", "stop"])
    monkeypatch.setattr("reachy_emotion.arm_client.arms_from_env", lambda: {"follower": arm})
    monkeypatch.setattr(at, "arm_tasks_mode", lambda: "on")
    monkeypatch.setattr(at, "arm_mount_from_env", lambda: _MOUNT)

    async def _receive(*args):
        seen["arm_tasks"] = args[9]

    async def _idle(*a):
        await asyncio.sleep(10)

    monkeypatch.setattr(lc, "_audio_pump", _idle)
    monkeypatch.setattr(lc, "_emotion_video_loop", _idle)
    monkeypatch.setattr(lc, "_receive_loop", _receive)
    mini = MagicMock()
    mini.media.get_input_audio_samplerate.return_value = 16000
    asyncio.run(lc.run_live_conversation(mini, threading.Event(), "p", "m"))

    names = [f.name for t in seen["config"].tools for f in t.function_declarations]
    assert "arm_task" in names and "plan_task" not in names
    assert "call arm_task" in str(seen["config"].system_instruction)
    assert seen["arm_tasks"].arm is arm and seen["arm_tasks"].mode == "on"
    arm.close.assert_called_once()  # the session's end parks the arm (putting back what it holds)
