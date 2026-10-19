"""Arm tasks: Gemini Robotics ER 2 does a physical task with the SO-101 arm, step by step.

Live calls ``arm_task(goal)`` ("put the pink eraser in the white cup"); ``ArmTask.run``
then runs a stateless ER 2 function-calling loop (Interactions, ``store=False``) with
four tools: ``look`` (turn the head; a new frame comes back), ``grasp`` and ``place``
(fixed sequences the arm server runs, aimed at points ER picks on the latest numbered
frame) and ``finish``. After every look, grasp or place a fresh camera frame goes back
with the last function result of that turn: a result only says what the arm did, the
frame shows whether it worked.

ER's [y, x] points become points in the table frame T (``scene_brain.table_point``):
metres, origin on the table under the centre of Reachy's foot, x Reachy's default
facing, y its left, z up. The arm server owns everything arm-specific (the mount, IK,
the keep-out zone around Reachy) and refuses a target it can't reach before anything
moves.

``.env``: ARM_TASKS = off (the default) | hover (grasp and place only hover over the
target, a place as high as it would start down: a test of the mount and the camera,
without contact) | on; and the arm's mount,
ARM_BASE_X_CM, ARM_BASE_Y_CM and ARM_BASE_YAW_DEG (the app needs it for the first view
and to tell ER where the arm stands).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from reachy_emotion.live_conversation import _FACING_DEG, _TILT_DEG
from reachy_emotion.scene_brain import (
    _JPEG_QUALITY,
    body_direction,
    camera_size,
    height_above,
    plane_point,
    points_to_pixels,
    table_point,
)

logger = logging.getLogger(__name__)

MODES = ("off", "hover", "on")

_MAX_TURNS = 8          # ER calls per task (a pick-and-place takes 3-5)
_BUDGET_S = 150.0       # no new ER call after this long
_ARM_TIMEOUT_S = 90.0   # one grasp/place/hover; then the arm is stopped
_STATUS_TIMEOUT_S = 5.0
_GRASP_Z_M = 0.005      # aim half a small object's height above the table
_PLACE_HEIGHT_M = 0.02  # put down on the table: the jaw tips let go 2 cm up
# Container heights the arm can drop into. A place starts down with the jaw tips 6 cm over the
# rim: the IK gets there over about 2/3 of the graspable spots for a 9 cm rim, none for 12 cm.
_RIM_MIN_M, _RIM_MAX_M = 0.02, 0.09
_ABOVE_RIM_M = 0.03     # let go this far above the rim
_WORK_AHEAD_M = 0.18    # the first view: the table this far in front of the arm's base ...
_WORK_TILT = "down"     # ... looked at from above
_MAX_BODY_DEG = 150.0
_PHYSICAL = ("look", "grasp", "place")
_OUTCOMES = ("done", "failed", "need_clarification", "infeasible")

_ONE_ACTION = {"ok": False, "executed": False, "error": "One action per turn: check the new frame first."}
_FINISH_TOO_SOON = {
    "ok": False, "executed": False,
    "error": "A look, grasp or place ran this turn: check the new frame first, then finish.",
}


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

def arm_tasks_mode() -> str:
    """ARM_TASKS from .env: "off" (the default), "hover" or "on"."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    mode = os.environ.get("ARM_TASKS", "").strip().lower() or "off"
    if mode not in MODES:
        logger.warning("ARM_TASKS=%r isn't off, hover or on: arm tasks are off", mode)
        return "off"
    return mode


def arm_mount_from_env() -> Optional[dict]:
    """Where the arm stands, from .env: ``{"x", "y"}`` of its shoulder-pan axis (metres,
    T frame) and ``"yaw_deg"``, the way it reaches at shoulder_pan 0 (+ = toward Reachy's
    left). None if ARM_BASE_X_CM, ARM_BASE_Y_CM or ARM_BASE_YAW_DEG is missing or not a
    number."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    values = []
    for key in ("ARM_BASE_X_CM", "ARM_BASE_Y_CM", "ARM_BASE_YAW_DEG"):
        raw = os.environ.get(key, "").strip()
        try:
            value = float(raw)
        except ValueError:
            if raw:
                logger.warning("%s=%r isn't a number", key, raw)
            return None
        if not math.isfinite(value):
            logger.warning("%s=%r isn't a number", key, raw)
            return None
        values.append(value)
    return {"x": values[0] / 100.0, "y": values[1] / 100.0, "yaw_deg": values[2]}


def work_view_deg(mount: dict) -> float:
    """Heading (degrees from Reachy's default facing, + = left) of the table 18 cm in front
    of the arm's base: the first frame looks there, tilted down."""
    yaw = math.radians(mount["yaw_deg"])
    x = mount["x"] + _WORK_AHEAD_M * math.cos(yaw)
    y = mount["y"] + _WORK_AHEAD_M * math.sin(yaw)
    return float(np.clip(math.degrees(math.atan2(y, x)), -_MAX_BODY_DEG, _MAX_BODY_DEG))


# --------------------------------------------------------------------------- #
# What ER is told                                                               #
# --------------------------------------------------------------------------- #

_POINT = {
    "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
    "description": "[y, x] on that frame, normalized to 0-1000.",
}
_FRAME = {"type": "integer", "description": "Number of the frame the points are on: always the most recent."}

ARM_TOOLS = [
    {
        "type": "function", "name": "look",
        "description": "Turn Reachy's head to see more of the table; a new frame comes back.",
        "parameters": {"type": "object", "properties": {
            "facing": {"type": "string", "enum": list(_FACING_DEG),
                       "description": "Which way to face, from Reachy's own body. Leave out to keep facing this way."},
            "tilt": {"type": "string", "enum": list(_TILT_DEG),
                     "description": "'down' for the table. Leave out to keep this tilt."},
        }},
    },
    {
        "type": "function", "name": "grasp",
        "description": (
            "Pick up one object from above: open the gripper, move over it, go down, close, lift. "
            "The result says what the arm did, not whether it worked: check the new frame."
        ),
        "parameters": {"type": "object", "properties": {
            "point": {**_POINT, "description": "[y, x] of the centre of the object where it rests on the table."},
            "frame": _FRAME,
            "label": {"type": "string", "description": "The object, e.g. 'pink eraser'."},
            "ends": {"type": "array", "items": _POINT, "minItems": 2, "maxItems": 2,
                     "description": "For a long object: [y, x] of the two ends of its long side."},
        }, "required": ["point", "frame", "label"]},
    },
    {
        "type": "function", "name": "place",
        "description": (
            "Put the held object down on the table, or drop it into an open container, then "
            "park the arm. The result says what the arm did, not whether it worked: check the new frame."
        ),
        "parameters": {"type": "object", "properties": {
            "frame": _FRAME,
            "target": {"type": "string", "description": "Where, e.g. 'the table' or 'the white cup'."},
            "release": {"type": "string", "enum": ["on_table", "into"]},
            "point": {**_POINT, "description": "For on_table: [y, x] of the centre of a free spot."},
            "base": {**_POINT, "description": (
                "For into: [y, x] where the container's front touches the table (its point nearest the camera).")},
            "rim_near": {**_POINT, "description": "For into: [y, x] of the near point of the container's rim."},
            "rim_far": {**_POINT, "description": (
                "For into: [y, x] of the far point of the rim, along your line of sight from rim_near.")},
        }, "required": ["frame", "target", "release"]},
    },
    {
        "type": "function", "name": "finish",
        "description": "End the task, with one or two short sentences for Reachy to say, in the first person.",
        "parameters": {"type": "object", "properties": {
            "outcome": {"type": "string", "enum": list(_OUTCOMES)},
            "say": {"type": "string"},
        }, "required": ["outcome", "say"]},
    },
]

_INSTRUCTION = (
    "You plan for Reachy Mini, a small robot head standing on a table, and its one SO-101 robot "
    "arm (five joints and a two-finger gripper), which stands on the same table {where}, about "
    "{distance} cm from the centre of Reachy's foot. The images come from the camera in Reachy's "
    "head; the head can turn and tilt (look); the arm has no camera.\n"
    "WHAT THE ARM CAN DO: pick up one small, light, solid object at a time from above (under 3 cm "
    "tall, narrow enough for the gripper), put it down on the table, or drop it into an open "
    "container up to {rim_cm} cm tall. It reaches table points about 10-26 cm from its own base, and "
    "never close to Reachy.\n"
    "WHAT IT CAN'T DO: push, pour, twist, open lids, stack, or hand things to people. It must never "
    "touch liquids, food, sharp, hot or fragile things, people or pets.\n"
    "POINTS: each image is a numbered frame. Points are [y, x] integers normalized to 0-1000 on the "
    "most recent frame, and frame is that frame's number; never compute other coordinates. For "
    "grasp, point to the centre of the object where it rests on the table; for a long object also "
    "give ends, the two ends of its long side. To place on the table, point to the centre of a "
    "free spot. To place into a container, give base, where the container's front touches the "
    "table (its point nearest the camera), and rim_near and rim_far, the near and far points of "
    "its rim along your line of sight. Facing directions are from Reachy's own body: 'on my left' "
    "is Reachy's left.\n"
    "HOW TO WORK: call one look, grasp or place per turn, then check the new frame that comes back "
    "before the next: a result is not proof, the frame is. If the target isn't in view, look (down "
    "at the table first). Before each grasp or place, check the newest frame: if a hand or a person "
    "is in the arm's workspace, don't move; finish with need_clarification and say why. If the "
    "goal is unclear, several things match, or it is out of reach or against the rules above, "
    "don't move: finish with need_clarification or infeasible and say why. Finish when the task is "
    "done, or after a second failed try."
)


def arm_instruction(mount: dict) -> str:
    """ER's system instruction: Reachy, where the arm stands (from *mount*), what it can do."""
    bearing = math.degrees(math.atan2(mount["y"], mount["x"]))
    side = "left" if bearing >= 0 else "right"
    if abs(bearing) <= 67.5:
        where = f"in front of Reachy, on its {side}"
    elif abs(bearing) <= 112.5:
        where = f"on Reachy's {side}"
    else:
        where = f"behind Reachy, on its {side}"
    distance = round(math.hypot(mount["x"], mount["y"]) * 100)
    return _INSTRUCTION.format(where=where, distance=distance, rim_cm=round(_RIM_MAX_M * 100))


def _tilt_words(tilt: float) -> str:
    if tilt > 0:
        return f"tilted down {tilt:g}°"
    if tilt < 0:
        return f"tilted up {-tilt:g}°"
    return "level"


def _valid_point(point: Any) -> bool:
    """[y, x], two numbers in 0..1000."""
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1000 for v in point)
    )


def _jaw_yaw(a: np.ndarray, b: np.ndarray) -> float:
    """Direction (degrees in T, + = counter-clockwise from Reachy's facing) the jaws close
    along to grip across a long object whose ends are *a* and *b*: its long axis + 90°.
    The jaws are symmetric, so it's folded into [-90, 90)."""
    yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + 90.0
    return (yaw + 90.0) % 180.0 - 90.0


def _fault_words(faults: dict) -> str:
    """'wrist_roll overload, gripper overheat' from a reply's faults."""
    return ", ".join(f"{motor} {'/'.join(kinds)}" for motor, kinds in faults.items())


@dataclass
class _Frame:
    number: int
    image: np.ndarray
    pose: Optional[np.ndarray]
    jpeg_b64: str
    caption: str


# --------------------------------------------------------------------------- #
# The loop                                                                      #
# --------------------------------------------------------------------------- #

class ArmTask:
    """One arm task: ER 2 plans it a step at a time through the tools, the arm server moves
    the arm, Reachy's head camera checks each step.

    Args:
        client: The ER client (``scene_brain.er_client``).
        model: ER model id.
        arm: The arm's ArmClient (``submit`` / ``call``).
        mini: The ReachyMini, for the camera model (the head moves only through *turn*).
        turn: ``turn(heading_deg, tilt_deg)`` turns the body and head (+ tilt = down);
            returns once the move is done.
        grab: ``grab() -> (frame, pose)``: a camera frame taken from now on, with the head
            pose it was taken at; (None, None) if none came.
        mount: The arm's mount (``arm_mount_from_env``).
        mode: "on", or "hover" to only hover over each grasp/place target.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        arm: Any,
        mini: Any,
        turn: Callable[[float, float], None],
        grab: Callable[[], tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        mount: dict,
        mode: str = "on",
    ) -> None:
        self.client, self.model, self.arm, self.mini = client, model, arm, mini
        self.turn, self.grab, self.mount, self.mode = turn, grab, mount, mode
        self.instruction = arm_instruction(mount)
        self.frame: Optional[_Frame] = None
        self.facing = 0.0  # where the head faces now (degrees, + = left) ...
        self.tilt = 0.0    # ... and its tilt (+ = down)
        self.holding: Optional[str] = None  # what the gripper holds, as last known
        self.did: list[str] = []  # what happened, in short, for Live
        self._end_now: Optional[tuple[str, Optional[str]]] = None  # an arm reply that ends the task

    def run(self, goal: str, stop: Any) -> dict:
        """Do *goal*; returns ``{"outcome", "say", "holding", "did"}`` for Live. *stop* (anything
        with ``is_set()``) ends the task at its next check. Never raises."""
        try:
            return self._run(goal, stop)
        except Exception as exc:
            logger.exception("arm task failed")
            return self._end("failed", f"Something went wrong with my arm task ({exc}).")

    def _run(self, goal: str, stop: Any) -> dict:
        start = time.monotonic()
        self.holding = self._status_holding()
        self._turn(work_view_deg(self.mount), _TILT_DEG[_WORK_TILT])
        if not self._capture():
            return self._end("failed", "I couldn't get a picture from my camera.")
        history: list[dict] = [{"type": "user_input", "content": [
            self._image(), {"type": "text", "text": f"{self.frame.caption}\nTask: {goal}"},
        ]}]
        answered: set = set()
        for turn in range(1, _MAX_TURNS + 1):
            if stop.is_set():
                return self._end("stopped", None)
            if time.monotonic() - start > _BUDGET_S:
                return self._end("failed", "I ran out of time before finishing.")
            t0 = time.monotonic()
            try:
                it = self.client.interactions.create(
                    model=self.model, store=False, input=history, tools=ARM_TOOLS,
                    system_instruction=self.instruction, generation_config={"thinking_level": "low"},
                )
            except Exception as exc:
                logger.warning("arm task: ER call failed: %s", exc)
                return self._end("failed", "My planner didn't answer, so I stopped.")
            steps = list(getattr(it, "steps", None) or [])
            calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
            logger.info(
                "arm task turn %d (%.1f s): %s %s", turn, time.monotonic() - t0, it.status,
                [(c.name, c.arguments) for c in calls],
            )
            if it.status not in ("requires_action", "completed"):
                return self._end("failed", f"My planner stopped ({it.status}).")
            history += [s.model_dump(mode="json") for s in steps]  # thoughts carry signatures: as received
            if not calls:  # it ended without finish: pass on what it said
                text = (getattr(it, "output_text", None) or "").strip()
                return self._end("unclear", text or "My planner stopped without saying how it went.")
            if stop.is_set():
                return self._end("stopped", None)

            physical = any(c.name in _PHYSICAL for c in calls)
            results: list[tuple[Any, dict]] = []
            acted = moved = False
            finish: Optional[tuple[str, str]] = None
            for c in calls:
                if c.id in answered:
                    continue
                answered.add(c.id)
                args = dict(c.arguments or {})
                if c.name == "finish":
                    if physical:
                        res = dict(_FINISH_TOO_SOON)
                    else:
                        outcome = str(args.get("outcome") or "failed")
                        finish = (outcome, str(args.get("say") or "").strip() or f"Task {outcome}.")
                        res = {"ok": True}
                elif c.name in _PHYSICAL:
                    if acted:
                        res = dict(_ONE_ACTION)
                    else:
                        acted = True
                        res, moved = self._execute(c.name, args)
                else:
                    res = {"ok": False, "executed": False, "error": f"no tool named {c.name}"}
                results.append((c, res))
                if self._end_now is not None:
                    break
            if self._end_now is not None:  # a fault, a stop or no answer: no more ER calls
                return self._end(*self._end_now)
            if stop.is_set():
                return self._end("stopped", None)
            if finish is not None:
                return self._end(*finish)
            if moved and not self._capture():
                return self._end("failed", "I lost my camera picture.")
            for i, (c, res) in enumerate(results):
                content: list[dict] = [{"type": "text", "text": json.dumps(res)}]
                if moved and i == len(results) - 1:  # one fresh frame per turn, with the last result
                    content += [self._image(), {"type": "text", "text": self.frame.caption}]
                history.append({"type": "function_result", "call_id": c.id, "name": c.name, "result": content})
        return self._end("failed", "I ran out of steps before finishing.")

    # -- tools -------------------------------------------------------------- #

    def _execute(self, name: str, args: dict) -> tuple[dict, bool]:
        """Run one look, grasp or place: (result for ER, whether the head or the arm was
        sent anywhere, so a new frame is due)."""
        if name == "look":
            return self._look(args)
        error = self._check_frame(args)
        if error is None:
            target = self._grasp_target(args) if name == "grasp" else self._place_target(args)
            if isinstance(target, str):
                error = target
        if error is not None:
            logger.info("arm task: %s refused: %s", name, error)
            return {"ok": False, "executed": False, "error": error}, False
        return self._arm(*target), True

    def _look(self, args: dict) -> tuple[dict, bool]:
        facing, tilt = args.get("facing"), args.get("tilt")
        for name, value, choices in (("facing", facing, _FACING_DEG), ("tilt", tilt, _TILT_DEG)):
            if value is not None and (not isinstance(value, str) or value not in choices):
                return {"ok": False, "executed": False, "error": f"{name} must be one of: {', '.join(choices)}"}, False
        yaw = float(_FACING_DEG.get(facing, self.facing))
        pitch = float(_TILT_DEG.get(tilt, self.tilt))
        try:
            self._turn(yaw, pitch)
        except Exception as exc:
            logger.warning("arm task: look failed: %s", exc)
            return {"ok": False, "executed": True, "error": f"the head didn't move: {exc}"}, True
        self.did.append(f"looked {body_direction(yaw)}, {_tilt_words(pitch)}")
        return {"ok": True, "executed": True}, True

    def _check_frame(self, args: dict) -> Optional[str]:
        number = args.get("frame")
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number != self.frame.number:
            return f"frame must be the latest frame, {self.frame.number}: point on that one"
        if self.frame.pose is None:
            return f"frame {self.frame.number} has no head pose, so its points can't be used: look again"
        return None

    def _pixel(self, point: Any, name: str) -> Any:
        """Image pixel of an ER point, or an error string."""
        if not _valid_point(point):
            return f"{name} must be [y, x]: two integers 0-1000 on frame {self.frame.number}"
        width, height = camera_size(self.mini, self.frame.image)
        return points_to_pixels([point], width, height)[0]

    def _table(self, point: Any, height_m: float, name: str) -> Any:
        """T point (at *height_m* above the table) of an ER point, or an error string."""
        pixel = self._pixel(point, name)
        if isinstance(pixel, str):
            return pixel
        p = table_point(self.mini, pixel, self.frame.pose, height_m)
        if p is None:
            return f"{name} {list(point)} is too far off to aim at from this view: look down at it first"
        return p

    def _grasp_target(self, args: dict) -> Any:
        """(action, arm args, what, what the next frame must show), or an error string."""
        if self.holding and self.mode == "on":
            return f"already holding the {self.holding}: place it first"
        label = str(args.get("label") or "").strip() or "object"
        p = self._table(args.get("point"), _GRASP_Z_M, "point")
        if isinstance(p, str):
            return p
        request: dict = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "label": label}
        ends = args.get("ends")
        if ends is not None:
            if not isinstance(ends, (list, tuple)) or len(ends) != 2:
                return "ends must be two [y, x] points"
            a, b = (self._table(e, _GRASP_Z_M, "ends") for e in ends)
            for end in (a, b):
                if isinstance(end, str):
                    return end
            request["yaw_deg"] = round(_jaw_yaw(a, b), 1)
        return "grasp", request, f"grasp the {label}", f"the {label} in the gripper, off the table"

    def _place_target(self, args: dict) -> Any:
        """(action, arm args, what, what the next frame must show), or an error string."""
        if self.mode == "on" and not self.holding:
            return "the gripper is empty: grasp something first"
        label = self.holding or "object"
        target = str(args.get("target") or "").strip() or "the table"
        release = args.get("release")
        if release == "on_table":
            p = self._table(args.get("point"), 0.0, "point")
            if isinstance(p, str):
                return p
            request = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "height": _PLACE_HEIGHT_M}
            return "place", request, f"put the {label} down on {target}", f"the {label} on {target}, out of the gripper"
        if release != "into":
            return "release must be on_table or into"
        base = self._table(args.get("base"), 0.0, "base")
        if isinstance(base, str):
            return base
        near, far = self._pixel(args.get("rim_near"), "rim_near"), self._pixel(args.get("rim_far"), "rim_far")
        for pixel in (near, far):
            if isinstance(pixel, str):
                return pixel
        rim = height_above(self.mini, base, near, self.frame.pose)
        if rim is None:
            return "can't tell the container's height from base and rim_near"
        if rim > _RIM_MAX_M:
            return (f"{target} is about {rim * 100:.0f} cm tall: the arm can only drop into containers "
                    f"up to {_RIM_MAX_M * 100:.0f} cm")
        rim = max(rim, _RIM_MIN_M)
        near_T = plane_point(self.mini, near, self.frame.pose, rim)
        far_T = plane_point(self.mini, far, self.frame.pose, rim)
        if near_T is None or far_T is None:
            return "can't find the container's opening from rim_near and rim_far"
        centre = (near_T + far_T) / 2.0
        request = {
            "x": round(float(centre[0]), 4), "y": round(float(centre[1]), 4), "height": round(rim + _ABOVE_RIM_M, 4),
        }
        return "place", request, f"let the {label} drop into {target}", f"the {label} in {target}, out of the gripper"

    def _arm(self, action: str, request: dict, what: str, shows: str) -> dict:
        """Send one grasp or place to the arm (a hover in hover mode) and wait for it. The
        result only says what the arm did; *shows* is what the next frame must show."""
        hover = self.mode == "hover"
        sent, sent_args = action, request
        if hover:
            # A place's height too: the hover stays as high as the place would start down,
            # clear of a container's rim.
            sent, sent_args = "hover", {k: request[k] for k in ("x", "y", "height") if k in request}
        logger.info("arm task: %s %s", sent, sent_args)
        future = self.arm.submit(sent, sent_args)
        try:
            reply = future.result(timeout=_ARM_TIMEOUT_S)
        except FutureTimeout:
            self.arm.submit("stop")
            self.did.append(f"{what}: no answer from the arm, so I stopped it")
            self._end_now = ("failed", "My arm didn't finish in time, so I stopped it.")
            return {"ok": False, "executed": True, "error": "no answer from the arm; it was stopped"}
        logger.info("arm task: %s → %s", sent, reply)
        if reply.get("faults"):
            faults = _fault_words(reply["faults"])
            self.did.append(f"{what}: motor fault ({faults})")
            self._end_now = ("stopped", f"My arm stopped: motor protection tripped ({faults}).")
            return {"ok": False, "executed": True, "error": str(reply.get("error"))}
        if reply.get("cancelled"):
            self.did.append(f"{what}: stopped")
            self._end_now = ("stopped", None)
            return {"ok": False, "executed": True, "error": "stopped"}
        if not reply.get("ok"):
            self.did.append(f"{what}: {reply.get('error')}")
            return {"ok": False, "executed": bool(reply.get("executed", True)), "error": str(reply.get("error"))}
        if hover:
            self.did.append(f"hovered over the spot to {what} (test mode)")
            return {
                "ok": True, "executed": True, "holding": False, "note": "test mode: hovered only",
                "check": "Nothing was picked up or put down: the arm only hovered over the spot and parked.",
            }
        if action == "grasp":
            label = request["label"]
            if reply.get("holding"):
                self.holding = label
                self.did.append(f"picked up the {label}")
                check = f"Claim success only if the new frame shows {shows}."
            else:
                self.holding = None
                self.did.append(f"tried to pick up the {label}: the gripper closed on nothing")
                check = "The gripper closed on nothing and the arm parked: check the new frame."
            return {"ok": True, "executed": True, "holding": bool(reply.get("holding")), "check": check}
        self.holding = None
        self.did.append(what)
        check = f"Claim success only if the new frame shows {shows}."
        return {"ok": True, "executed": True, "holding": False, "check": check}

    # -- head, camera, arm state ------------------------------------------- #

    def _turn(self, facing: float, tilt: float) -> None:
        self.turn(facing, tilt)
        self.facing, self.tilt = facing, tilt

    def _capture(self) -> bool:
        """Take the next numbered frame (the head is still), with its caption."""
        import cv2

        image, pose = self.grab()
        if image is None:
            return False
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return False
        self.holding = self._status_holding()
        number = self.frame.number + 1 if self.frame is not None else 1
        caption = (
            f"Frame {number}: head facing {body_direction(self.facing)}, {_tilt_words(self.tilt)}. "
            f"Holding: {self.holding or 'nothing'}."
        )
        self.frame = _Frame(number, image, pose, base64.b64encode(jpg.tobytes()).decode("ascii"), caption)
        return True

    def _image(self) -> dict:
        return {"type": "image", "mime_type": "image/jpeg", "data": self.frame.jpeg_b64}

    def _status_holding(self) -> Optional[str]:
        """What the arm server says the gripper holds (its label), else what we last knew."""
        try:
            status = self.arm.call("status", timeout=_STATUS_TIMEOUT_S)
        except Exception as exc:
            logger.warning("arm task: no arm status: %s", exc)
            return self.holding
        if isinstance(status, dict) and "holding" in status:
            return str(status["holding"]) if status["holding"] else None
        return self.holding

    def _end(self, outcome: str, say: Optional[str]) -> dict:
        """The result for Live. A task that stopped or failed says whether the arm still holds something."""
        holding = self._status_holding()
        if say is None:
            say = "I stopped."
        if holding and outcome in ("stopped", "failed", "unclear"):
            say = f"{say} I'm still holding the {holding}."
        result = {"outcome": outcome, "say": say, "holding": holding, "did": list(self.did)}
        logger.info("arm task: %s", result)
        return result
