"""SceneBrain: Gemini Robotics ER 2 as Reachy's embodied-reasoning "brain".

The Live conversation model (``gemini-3.8-live``) handles voice; ER 2 is called
as a tool when the conversation needs spatial reasoning:

  * ``locate`` — find objects in the current camera frame and return
    ``[{"label", "point": [y, x]}]`` with points normalised to 0-1000
    (the format documented for ER models);
  * ``plan`` — break a physical goal into ordered steps for the SO-101 arm (the
    input the M3 skill router will consume), or say why it can't be done.

Both go through the Interactions API with ``store=False`` (no camera frames of
the home are kept on Google's side), and ask for JSON in a fixed shape
(structured output), so a reply that can't be read is an error, never taken for
"nothing there".

``look_at_objects`` then turns Reachy's head toward each located object via
the SDK's ``look_at_image`` (verified: it needs the Reachy camera intrinsics,
which the daemon provides) and back to neutral; ``face_toward`` turns Reachy to
face a person and stays there.

ER 2 streaming (``gemini-robotics-er-2-streaming-preview``) was ruled out as the
voice model: it rejects AUDIO output (1007), so it can't replace Live.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ER_MODEL = "gemini-robotics-er-2-preview"

_MAX_OBJECTS = 8
_MAX_STEPS = 8
_ER_TIMEOUT_MS = 20_000  # plan_task measured ~5 s; bound it so a stalled call can't hang
# One retry when the preview model is busy (HTTP 408/409/429/5xx). On Interactions the SDK
# reads HttpRetryOptions.attempts as the number of RETRIES, not tries, and doesn't retry a
# timeout or dropped connection: a stalled call fails after one 20 s try, not two.
_ER_RETRIES = 1
_JPEG_QUALITY = 85  # about half the bytes of OpenCV's 95, at full resolution for small things
_PLAN_ACTIONS = ("locate", "grasp", "place", "move")

# Headings to check when searching the room: with the ~80° camera they cover the
# full circle (the body turns up to ±160°; ±150° keeps a margin).
SCAN_YAWS_DEG = (60, 120, 150, -60, -120, -150)
_MAX_BODY_YAW = np.radians(150)

_LOCATE_PROMPT = (
    "Point to {target}. The label returned should be an identifying name for the object "
    "detected, so similar ones can be told apart (e.g. 'red mug', 'white mug'). Only "
    "include objects that are actually present in the image; if there are none, return "
    "no items. The points are in [y, x] format normalized to 0-1000. At most {n} items."
)
# Google's ER 2 pointing shape (robotics-samples' PointResponse).
_POINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": _MAX_OBJECTS,
            "items": {
                "type": "object",
                "properties": {
                    "point": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[y, x] format normalized to 0-1000",
                    },
                    "label": {"type": "string", "description": "Identifying name for the object detected"},
                },
                "required": ["point", "label"],
            },
        }
    },
    "required": ["items"],
}

_PLAN_PROMPT = (
    "You plan for a small robot. The image is from the camera in its head. Its one arm, an "
    "SO-101 (five joints and a gripper), is mounted on its left side, like a left hand. "
    "Break the goal \"{goal}\" into short, ordered steps for that arm, using only these "
    "actions: {actions}. Only use objects you can see. If the goal is unclear, or the arm "
    "can't do it here (for example the object isn't in view or is too far from the arm), set "
    "feasible to false and say why in a few words, with no steps. At most {n} steps."
)
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string", "description": "Why not, in a few words (only when not feasible)"},
        "steps": {
            "type": "array",
            "maxItems": _MAX_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "action": {"type": "string", "enum": list(_PLAN_ACTIONS)},
                    "object": {"type": "string"},
                    "target": {"type": "string", "description": "Where to, if the action has a destination"},
                },
                "required": ["step", "action", "object"],
            },
        },
    },
    "required": ["feasible", "steps"],
}

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #

def _load_er_model() -> str:
    """Return GEMINI_ER_MODEL from env, or the default ER 2 model."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    return os.environ.get("GEMINI_ER_MODEL", "").strip() or DEFAULT_ER_MODEL


def _parse_json(text: Optional[str]) -> Any:
    """Parse model JSON, tolerating the ```json fences ER 2 wraps it in."""
    if not text:
        return None
    try:
        return json.loads(_FENCE.sub("", text.strip()))
    except (json.JSONDecodeError, ValueError):
        return None


def _valid_objects(parsed: Any) -> list[dict]:
    """Keep only well-formed ``{"label", "point": [y, x]}`` entries."""
    out: list[dict] = []
    if not isinstance(parsed, list):
        return out
    for item in parsed:
        if not isinstance(item, dict):
            continue
        point = item.get("point")
        label = str(item.get("label", "")).strip()
        if (
            label
            and isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(c, (int, float)) for c in point)
        ):
            out.append({"label": label, "point": [float(point[0]), float(point[1])]})
            if len(out) == _MAX_OBJECTS:
                break
    return out


def points_to_pixels(points: list, width: int, height: int) -> list[tuple[int, int]]:
    """Normalised ``[y, x]`` (0-1000) -> ``(u, v)`` pixels, clamped strictly inside
    the image (``look_at_image`` asserts ``0 < u < width`` and ``0 < v < height``)."""
    pixels = []
    for y, x in points:
        u = int(round(x / 1000.0 * width))
        v = int(round(y / 1000.0 * height))
        pixels.append((min(max(u, 1), width - 1), min(max(v, 1), height - 1)))
    return pixels


def _camera_ray(mini: Any, pixel: tuple[int, int], pose: Optional[np.ndarray]) -> np.ndarray:
    """Direction (world frame: x forward, y left, z up) of the camera ray through image
    *pixel*, for a frame taken with the head at *pose*: ``look_at_image``'s geometry,
    rotated by the *capture* pose, so it holds wherever the head and body point now.
    Raises without the camera model."""
    from reachy_mini.media.camera_utils import undistort_points

    head = pose if pose is not None else np.eye(4)
    camera = mini.media.camera
    x_n, y_n = undistort_points(pixel[0], pixel[1], camera.K, camera.D)
    return (head @ mini.T_head_cam)[:3, :3] @ np.array([x_n, y_n, 1.0])


def bearing_deg(mini: Any, pixel: tuple[int, int], pose: Optional[np.ndarray]) -> float:
    """Direction of image *pixel*, in degrees from Reachy's default facing (0° = straight
    ahead with the body unturned, + = its left), for a frame taken with the head at *pose*.
    Nothing moves. Without the camera model it falls back to the way the head faced."""
    try:
        ray = _camera_ray(mini, pixel, pose)
        return float(np.degrees(np.arctan2(ray[1], ray[0])))
    except Exception:
        return float(np.degrees(heading(pose if pose is not None else np.eye(4))))


def elevation_deg(mini: Any, pixel: tuple[int, int], pose: Optional[np.ndarray]) -> float:
    """How far above (+) or below (-) eye level image *pixel* is, in degrees; 0 without
    the camera model."""
    try:
        ray = _camera_ray(mini, pixel, pose)
        return float(np.degrees(np.arctan2(ray[2], np.hypot(ray[0], ray[1]))))
    except Exception:
        return 0.0


def body_direction(deg: float, elevation: float = 0.0) -> str:
    """Where a bearing is, said like a person would from where they stand (0° = straight
    ahead, + = left): 'ahead', 'ahead on my left', 'on my left', 'behind me on my left',
    'behind me' — and the same on the right. Things well below or above eye level (an
    *elevation* past ±20°) get ', down low' or ', up high'."""
    deg = (deg + 180.0) % 360.0 - 180.0
    side = "left" if deg > 0 else "right"
    off = abs(deg)
    if off <= 22.5:
        words = "ahead"
    elif off <= 67.5:
        words = f"ahead on my {side}"
    elif off <= 112.5:
        words = f"on my {side}"
    elif off <= 157.5:
        words = f"behind me on my {side}"
    else:
        words = "behind me"
    if elevation <= -20:
        return f"{words}, down low"
    if elevation >= 20:
        return f"{words}, up high"
    return words


def camera_size(mini: Any, frame: np.ndarray) -> tuple[int, int]:
    """``(width, height)`` used by ``look_at_image``'s bounds; fall back to the frame."""
    try:
        w, h = mini.media.camera.resolution
        return int(w), int(h)
    except Exception:
        return int(frame.shape[1]), int(frame.shape[0])


def heading(pose: np.ndarray) -> float:
    """World heading (yaw, radians) the head's forward (x) axis points along."""
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _unwrap(yaw: float, ref: float) -> float:
    """*yaw* expressed within ±π of *ref* (undoes atan2's wrap at ±180°)."""
    return ref + float(np.angle(np.exp(1j * (yaw - ref))))


def go_neutral(mini: Any, body_yaw: float = 0.0, duration: float = 0.8, pitch: float = 0.0) -> None:
    """Head straight (and level, unless tilted by *pitch*: + = down), antennas at rest,
    body at *body_yaw* (0 = facing forward). Radians.

    Blocking (``goto_target`` waits for the move).
    """
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS
    from reachy_mini.utils import create_head_pose

    mini.goto_target(
        head=create_head_pose(yaw=body_yaw, pitch=pitch, degrees=False),
        antennas=INIT_ANTENNAS_JOINT_POSITIONS,
        body_yaw=body_yaw,
        duration=duration,
    )


def _gaze_targets(
    mini: Any, pixels: list[tuple[int, int]], capture_pose: Optional[np.ndarray], duration: float
) -> list[np.ndarray]:
    """Head poses that look at each pixel of a frame taken at *capture_pose*.

    ``look_at_image`` builds its ray from the head's *current* pose, but every pixel
    comes from one frame taken at ``capture_pose``. So: go back to the capture pose
    (if the head has moved since) and work out every target up front with
    ``perform_movement=False`` — otherwise each look would be computed from the
    previous one and aim at the wrong spot.
    """
    try:
        now = mini.get_current_head_pose()
    except Exception:
        now = None
    if capture_pose is not None and (now is None or not np.allclose(now, capture_pose, atol=1e-3)):
        try:
            mini.goto_target(head=capture_pose, duration=duration, body_yaw=None)
        except Exception as exc:
            logger.warning("could not return to the frame's capture pose: %s", exc)
            return []

    targets = []
    for u, v in pixels:
        try:
            targets.append(mini.look_at_image(u, v, perform_movement=False))
        except Exception as exc:
            logger.warning("look_at_image(%s, %s) failed: %s", u, v, exc)
            break
    return targets


def look_at_objects(
    mini: Any,
    pixels: list[tuple[int, int]],
    capture_pose: Optional[np.ndarray] = None,
    duration: float = 0.8,
    hold_s: float = 0.5,
) -> None:
    """Turn the head toward each pixel in turn, then back to neutral.

    Blocking (each ``goto_target`` waits for its move) — run it off the event loop.
    ``body_yaw=None`` keeps the body's rotation during the tour (``look_at_world``
    would reset it to 0). Targets past ±180° are skipped: the head would reach them
    the short way round while the body spins the long way.
    """
    ref = heading(capture_pose) if capture_pose is not None else 0.0
    for pose in _gaze_targets(mini, pixels, capture_pose, duration):
        if abs(_unwrap(heading(pose), ref)) > np.pi:
            continue
        try:
            mini.goto_target(head=pose, duration=duration, body_yaw=None)
        except Exception as exc:
            logger.warning("gaze move failed: %s", exc)
            break
        time.sleep(hold_s)
    try:
        go_neutral(mini, duration=duration)
    except Exception as exc:
        logger.warning("returning to neutral failed: %s", exc)


def face_toward(
    mini: Any, pixel: tuple[int, int], capture_pose: Optional[np.ndarray] = None, duration: float = 0.8
) -> None:
    """Turn to face a person: body toward them, head neutral (straight and level). Stays there."""
    targets = _gaze_targets(mini, [pixel], capture_pose, duration)
    if not targets:
        return
    # Unwrap against the capture heading: a person at 185° seen from the +150° stop
    # stays at 185° (clipped to the body limit), not -175° (a spin the other way).
    ref = heading(capture_pose) if capture_pose is not None else heading(targets[0])
    yaw = float(np.clip(_unwrap(heading(targets[0]), ref), -_MAX_BODY_YAW, _MAX_BODY_YAW))
    try:
        go_neutral(mini, body_yaw=yaw, duration=duration)
    except Exception as exc:
        logger.warning("turning to face the person failed: %s", exc)


# --------------------------------------------------------------------------- #
# ER 2 client                                                                   #
# --------------------------------------------------------------------------- #

def er_client(api_key: str) -> Any:
    """A Gemini client for ER 2. Interactions takes its timeout and retries only from the
    client, so ER gets its own rather than sharing (and changing) the Live one."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=_ER_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=_ER_RETRIES)
        ),
    )


class SceneBrain:
    """Thin wrapper over Gemini Robotics ER 2 for pointing and task planning.

    Args:
        client: A ``google.genai.Client`` made by ``er_client``.
        model: ER model id (else GEMINI_ER_MODEL env / DEFAULT_ER_MODEL).
    """

    def __init__(self, client: Any, model: Optional[str] = None) -> None:
        self._client = client
        self.model = model or _load_er_model()
        self.last_plan: Optional[dict] = None  # kept for the M3 skill router
        self.last_raw = ""  # ER's last raw reply, for logging

    def _ask(self, frame_bgr: np.ndarray, prompt: str, schema: dict, thinking: str) -> Any:
        """One ER request on a frame, answered as JSON in *schema*'s shape, with *thinking*
        ("low" or "high"). Raises if ER didn't finish or the reply isn't JSON, so a garbled or
        blocked answer is an error rather than "nothing there"."""
        import cv2

        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            raise RuntimeError("could not encode camera frame")
        resp = self._client.interactions.create(
            model=self.model,
            input=[
                {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(jpg.tobytes()).decode("ascii")},
                {"type": "text", "text": prompt},
            ],
            generation_config={"thinking_level": thinking},
            response_format={"type": "text", "mime_type": "application/json", "schema": schema},
            store=False,  # Interactions keeps requests by default; these are frames of a home
        )
        self.last_raw = resp.output_text or ""
        if resp.status != "completed":
            raise RuntimeError(f"ER 2 didn't finish ({resp.status}): {self.last_raw[:120]!r}")
        parsed = _parse_json(self.last_raw)
        if parsed is None:
            raise RuntimeError(f"ER 2 didn't answer in JSON: {self.last_raw[:120]!r}")
        return parsed

    def locate(self, frame_bgr: np.ndarray, query: Optional[str] = None) -> list[dict]:
        """Find objects (all distinct ones, or those matching *query*). Pointing needs little
        thinking, and a room search makes one call per heading while the user waits."""
        target = query.strip() if query and query.strip() else "the distinct objects in view"
        parsed = self._ask(frame_bgr, _LOCATE_PROMPT.format(target=target, n=_MAX_OBJECTS), _POINTS_SCHEMA, "low")
        items = parsed.get("items") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            raise RuntimeError(f"ER 2 answered without a list of points: {self.last_raw[:120]!r}")
        return _valid_objects(items)

    def plan(self, frame_bgr: np.ndarray, goal: str) -> dict:
        """Break *goal* into ordered steps for the SO-101 arm: ``{"steps": [...]}``, or
        ``{"feasible": False, "reason": ...}`` when the goal is unclear or can't be done
        here. Plans get ER's full thinking (HIGH)."""
        parsed = self._ask(
            frame_bgr,
            _PLAN_PROMPT.format(goal=goal, actions=", ".join(_PLAN_ACTIONS), n=_MAX_STEPS),
            _PLAN_SCHEMA,
            "high",
        )
        if isinstance(parsed, dict) and parsed.get("feasible") is False:
            result: dict = {"feasible": False, "reason": str(parsed.get("reason") or "it can't be done here")}
        else:
            steps = parsed.get("steps") if isinstance(parsed, dict) else None
            if not isinstance(steps, list):
                raise RuntimeError(f"ER 2 answered without a plan: {self.last_raw[:120]!r}")
            result = {"steps": [s for s in steps[:_MAX_STEPS] if isinstance(s, dict)]}
        self.last_plan = {"goal": goal, **result}
        return result
