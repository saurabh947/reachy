"""Tests for GeminiBridge: the Interactions conversation, tool calls, tool execution."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for Interactions responses
# ---------------------------------------------------------------------------

class _Step(SimpleNamespace):
    """An Interactions step; model_dump() gives it back as sent (what history stores)."""

    def model_dump(self):
        return dict(vars(self))


def _call(name: str = "detect_emotion", call_id: str = "c1") -> _Step:
    return _Step(type="function_call", name=name, id=call_id, arguments={})


def _asks(*calls) -> SimpleNamespace:
    """Gemini asks for tool calls (the interaction waits on them)."""
    return SimpleNamespace(status="requires_action", steps=[_Step(type="thought"), *calls], output_text=None)


def _says(text: str) -> SimpleNamespace:
    """Gemini's final reply."""
    content = [{"type": "text", "text": text}]
    return SimpleNamespace(status="completed", steps=[_Step(type="model_output", content=content)], output_text=text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNCLEAR_RESULT = {
    "dominant_emotion": "unclear",
    "confidence": 0.0,
    "note": "No result from emotion-cloud within timeout",
}


def _make_emotion_client(result=None) -> MagicMock:
    """Return a mock emotion client with a pre-configured detect_emotion return value."""
    client = MagicMock()
    client.detect_emotion.return_value = result if result is not None else _UNCLEAR_RESULT
    return client


def _make_bridge(emotion_client=None):
    from reachy_emotion.gemini_bridge import GeminiBridge
    return GeminiBridge(
        api_key="test-key",
        emotion_client=emotion_client or _make_emotion_client(),
        model="gemini-test",
    )


# ---------------------------------------------------------------------------
# _run_emotion_detection
# ---------------------------------------------------------------------------

def test_run_emotion_detection_returns_error_when_no_client():
    from reachy_emotion.gemini_bridge import GeminiBridge
    bridge = GeminiBridge(api_key="key", emotion_client=None, model="test")
    result = bridge._run_emotion_detection()
    assert "error" in result


def test_run_emotion_detection_returns_unclear_when_no_result():
    """detect_emotion() returning unclear is passed through as-is."""
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=None))
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "unclear"
    assert result["confidence"] == 0.0
    assert "note" in result
    # unclear result must NOT be stored as last_emotion_result
    assert bridge._last_emotion_result is None


def test_run_emotion_detection_returns_cloud_result():
    cloud_result = {
        "dominant_emotion": "happy",
        "confidence": 0.91,
        "confidence_scores": {"happy": 0.91, "neutral": 0.05},
        "stress": 0.1,
        "engagement": 0.88,
        "arousal": 0.65,
    }
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=cloud_result))
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "happy"
    assert result["confidence"] == pytest.approx(0.91, abs=0.01)
    assert bridge._last_emotion_result is cloud_result


def test_run_emotion_detection_stores_last_result():
    cloud_result = {"dominant_emotion": "sad", "confidence": 0.75}
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=cloud_result))
    bridge._run_emotion_detection()
    assert bridge._last_emotion_result == cloud_result


# ---------------------------------------------------------------------------
# chat — top-level orchestration
# ---------------------------------------------------------------------------

def test_chat_raises_if_not_initialized():
    bridge = _make_bridge()
    with pytest.raises(RuntimeError, match="initialize"):
        bridge.chat("Hello!")


def _ready(*interactions, emotion_client=None):
    """A bridge whose client answers create() with *interactions*, in turn."""
    bridge = _make_bridge(emotion_client)
    bridge._client = MagicMock()
    bridge._client.interactions.create.side_effect = list(interactions)
    return bridge, bridge._client.interactions.create


def _user(text):
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def test_chat_returns_text_when_no_tool_calls():
    bridge, create = _ready(_says("Nice to meet you!"))

    text, emotion = bridge.chat("Hello Reachy!")

    assert text == "Nice to meet you!"
    assert emotion is None
    kw = create.call_args.kwargs
    assert kw["input"] == [_user("Hello Reachy!")] and kw["model"] == "gemini-test"
    assert kw["store"] is False  # nothing kept on Google's side
    assert kw["system_instruction"] and kw["tools"][0]["name"] == "detect_emotion"


def test_chat_keeps_the_conversation():
    bridge, create = _ready(_says("Hi Sam!"), _says("You're Sam."))
    bridge.chat("I'm Sam")
    bridge.chat("Who am I?")
    sent = create.call_args.kwargs["input"]
    assert sent[0] == _user("I'm Sam") and sent[1]["type"] == "model_output" and sent[2] == _user("Who am I?")


def test_chat_resets_last_emotion_result_each_turn():
    bridge, _ = _ready(_says("Hi"))
    bridge._last_emotion_result = {"dominant_emotion": "sad"}  # leftover from previous turn

    _, emotion = bridge.chat("Hello")

    assert emotion is None  # reset at start of turn


def test_chat_handles_detect_emotion_tool_call():
    """Full tool-call loop: Gemini calls detect_emotion, gets the result, returns text."""
    cloud_result = {
        "dominant_emotion": "happy",
        "confidence": 0.88,
        "confidence_scores": {"happy": 0.88},
        "stress": 0.1,
        "engagement": 0.9,
        "arousal": 0.6,
    }
    bridge, create = _ready(_asks(_call()), _says("You look happy!"),
                            emotion_client=_make_emotion_client(result=cloud_result))

    text, emotion = bridge.chat("How am I feeling?")

    assert text == "You look happy!"
    assert emotion == cloud_result
    assert create.call_count == 2
    # the second call carries the whole turn: question, Gemini's steps as received, the result
    sent = create.call_args.kwargs["input"]
    assert [s["type"] for s in sent] == ["user_input", "thought", "function_call", "function_result"]
    assert sent[3]["call_id"] == "c1" and sent[3]["name"] == "detect_emotion"
    assert json.loads(sent[3]["result"][0]["text"]) == cloud_result


def test_chat_handles_unknown_tool_gracefully():
    """If Gemini calls an unknown tool, the bridge sends an error result and continues."""
    bridge, create = _ready(_asks(_call("nonexistent_tool")), _says("Okay."))

    text, emotion = bridge.chat("Do something weird")

    assert text == "Okay."
    assert emotion is None
    assert "error" in json.loads(create.call_args.kwargs["input"][-1]["result"][0]["text"])


def test_chat_respects_max_tool_call_depth():
    """If the model keeps returning tool calls, the loop exits after _MAX_TOOL_CALL_DEPTH."""
    from reachy_emotion.gemini_bridge import _MAX_TOOL_CALL_DEPTH

    bridge, create = _ready(*[_asks(_call()) for _ in range(1 + _MAX_TOOL_CALL_DEPTH)],
                            emotion_client=_make_emotion_client(result=None))

    bridge.chat("Loop forever?")

    # 1 (user message) + _MAX_TOOL_CALL_DEPTH (tool results)
    assert create.call_count == 1 + _MAX_TOOL_CALL_DEPTH
    assert bridge._history == []  # its last tool call is unanswered: the turn isn't kept


def test_an_answer_after_the_last_tool_round_is_kept():
    from reachy_emotion.gemini_bridge import _MAX_TOOL_CALL_DEPTH

    bridge, _ = _ready(*[_asks(_call()) for _ in range(_MAX_TOOL_CALL_DEPTH)], _says("You seem calm, Sam."))
    assert bridge.chat("I'm Sam. Keep checking how I feel.")[0] == "You seem calm, Sam."
    assert bridge._history[0] == _user("I'm Sam. Keep checking how I feel.")
    assert bridge._history[-1]["type"] == "model_output"


def test_a_failed_turn_leaves_the_conversation_as_it_was():
    bridge, _ = _ready(_says("Hi Sam!"), _asks(_call()), RuntimeError("503"))
    bridge.chat("I'm Sam")
    before = list(bridge._history)
    with pytest.raises(RuntimeError):
        bridge.chat("How am I feeling?")
    assert bridge._history == before  # no half-answered tool call left behind


def test_an_unfinished_interaction_is_an_error():
    bridge, _ = _ready(SimpleNamespace(status="failed", steps=[], output_text=None))
    with pytest.raises(RuntimeError, match="failed"):
        bridge.chat("Hello")


# ---------------------------------------------------------------------------
# Lifecycle: initialize() and shutdown()
# ---------------------------------------------------------------------------

def test_shutdown_is_idempotent():
    """Calling shutdown() multiple times must not raise."""
    bridge = _make_bridge()
    bridge.shutdown()
    bridge.shutdown()


def test_api_key_cleared_after_initialize():
    """The API key must be erased from the instance after the Gemini client is created."""
    from reachy_emotion.gemini_bridge import GeminiBridge

    bridge = GeminiBridge(
        api_key="super-secret-key",
        emotion_client=_make_emotion_client(),
        model="test",
    )

    mock_client = MagicMock()
    fake_genai_module = MagicMock()
    fake_genai_module.Client.return_value = mock_client
    fake_google = MagicMock()
    fake_google.genai = fake_genai_module

    with patch.dict(sys.modules, {
        "google": fake_google,
        "google.genai": fake_genai_module,
        "google.genai.types": MagicMock(),
    }):
        bridge.initialize()

    assert bridge._GeminiBridge__api_key == ""
