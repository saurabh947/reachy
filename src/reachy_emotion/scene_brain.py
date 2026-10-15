"""SceneBrain: Gemini Robotics ER 2 as Reachy's embodied-reasoning "brain".

The Live conversation model (``gemini-3.8-live``) handles voice; ER 2 is called
as a tool when the conversation needs spatial reasoning:

  * ``locate`` — find objects in the current camera frame and return
    ``[{"label", "point": [y, x]}]`` with points normalised to 0-1000
    (the format documented for ER models);
  * ``plan`` — break a physical goal into ordered robot steps (the input the
    M3 skill router will consume once the SO-101 arm is connected).

``look_at_objects`` then turns Reachy's head toward each located object via
the SDK's ``look_at_image`` (verified: it needs the Reachy camera intrinsics,
which the daemon provides) and back to neutral; ``face_toward`` turns Reachy to
face a person and stays there.

ER 2 streaming (``gemini-robotics-er-2-streaming-preview``) was ruled out as the
voice model: it rejects AUDIO output (1007), so it can't replace Live.
"""

from __future__ import annotations

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
_PLAN_ACTIONS = ("locate", "grasp", "place", "move")

# Headings to check when searching the room: with the ~80° camera they cover the
# full circle (the body turns up to ±160°; ±150° keeps a margin).
SCAN_YAWS_DEG = (60, 120, 150, -60, -120, -150)
_MAX_BODY_YAW = np.radians(150)

_LOCATE_PROMPT = (
    "Point to {target}. Answer only as JSON: "
    '[{{"point": [y, x], "label": "<short name>"}}] with points normalized to 0-1000. '
    "At most {n} items."
)

_PLAN_PROMPT = (
    "You control a robot arm that can see this scene. Break the goal \"{goal}\" into "
    "short, ordered steps using only these actions: {actions}. Only use objects you "
    "can actually see. Answer only as JSON: "
    '[{{"step": 1, "action": "<action>", "object": "<object>", "target": "<where, or null>"}}]. '
    "At most {n} steps."
)

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


def bearing_deg(mini: Any, pixel: tuple[int, int], pose: Optional[np.ndarray]) -> float:
    """Direction of image *pixel*, in degrees from Reachy's default facing (0° = straight
    ahead with the body unturned, + = its left), for a frame taken with the head at *pose*.

    The camera ray through the pixel (``look_at_image``'s geometry) rotated by the
    *capture* pose — so it holds wherever the head and body point now. Nothing moves.
    Without the camera model it falls back to the way the head faced.
    """
    from reachy_mini.media.camera_utils import undistort_points

    head = pose if pose is not None else np.eye(4)
    try:
        camera = mini.media.camera
        x_n, y_n = undistort_points(pixel[0], pixel[1], camera.K, camera.D)
        ray = (head @ mini.T_head_cam)[:3, :3] @ np.array([x_n, y_n, 1.0])
        return float(np.degrees(np.arctan2(ray[1], ray[0])))
    except Exception:
        return float(np.degrees(heading(head)))


def body_direction(deg: float) -> str:
    """Where a bearing is, said like a person would from where they stand (0° = straight
    ahead, + = left): 'ahead', 'ahead on my left', 'on my left', 'behind me on my left',
    'behind me' — and the same on the right."""
    deg = (deg + 180.0) % 360.0 - 180.0
    side = "left" if deg > 0 else "right"
    off = abs(deg)
    if off <= 22.5:
        return "ahead"
    if off <= 67.5:
        return f"ahead on my {side}"
    if off <= 112.5:
        return f"on my {side}"
    if off <= 157.5:
        return f"behind me on my {side}"
    return "behind me"


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


def go_neutral(mini: Any, body_yaw: float = 0.0, duration: float = 0.8) -> None:
    """Head straight and level, antennas at rest, body at *body_yaw* (0 = facing forward).

    Blocking (``goto_target`` waits for the move).
    """
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS
    from reachy_mini.utils import create_head_pose

    mini.goto_target(
        head=create_head_pose(yaw=body_yaw, degrees=False),
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

class SceneBrain:
    """Thin wrapper over Gemini Robotics ER 2 for pointing and task planning.

    Args:
        client: A ``google.genai.Client`` (shared with the Live session).
        model: ER model id (else GEMINI_ER_MODEL env / DEFAULT_ER_MODEL).
    """

    def __init__(self, client: Any, model: Optional[str] = None) -> None:
        self._client = client
        self.model = model or _load_er_model()
        self.last_plan: Optional[dict] = None  # kept for the M3 skill router
        self.last_raw = ""  # ER's last raw reply, for logging

    def _ask(self, frame_bgr: np.ndarray, prompt: str) -> Any:
        import cv2
        from google.genai import types

        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("could not encode camera frame")
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt],
            # Without an explicit timeout google-genai waits forever on a stalled request.
            config=types.GenerateContentConfig(http_options=types.HttpOptions(timeout=_ER_TIMEOUT_MS)),
        )
        self.last_raw = resp.text or ""
        return _parse_json(resp.text)

    def locate(self, frame_bgr: np.ndarray, query: Optional[str] = None) -> list[dict]:
        """Find objects (all distinct ones, or those matching *query*)."""
        target = query.strip() if query and query.strip() else "the distinct objects in view"
        parsed = self._ask(frame_bgr, _LOCATE_PROMPT.format(target=target, n=_MAX_OBJECTS))
        return _valid_objects(parsed)

    def plan(self, frame_bgr: np.ndarray, goal: str) -> list[dict]:
        """Break *goal* into ordered steps for the (future) SO-101 arm."""
        parsed = self._ask(
            frame_bgr,
            _PLAN_PROMPT.format(goal=goal, actions=", ".join(_PLAN_ACTIONS), n=_MAX_STEPS),
        )
        steps = [s for s in parsed[:_MAX_STEPS] if isinstance(s, dict)] if isinstance(parsed, list) else []
        self.last_plan = {"goal": goal, "steps": steps}
        return steps
