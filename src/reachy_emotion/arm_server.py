"""Arm server: drives the SO-101 follower arm for the Reachy app, and teaches it moves.

Runs in ``.venv-lerobot``. LeRobot can't share the Reachy app's environment:
reachy-mini pins ``huggingface-hub==1.3.0`` and LeRobot needs ``>=1.6``. So this file is
standalone: stdlib only at import time, LeRobot (and numpy) imported lazily, no
``reachy_emotion`` imports.

    serve                       started by the Reachy app; JSON lines on stdin/stdout
    teach NAME                  save the follower's current pose (move it by hand, motors relaxed)
    record NAME [--seconds S]   record a gesture by moving the leader (the follower mirrors it)
    fk [--registers]            read-only: print where the gripper is while you move the arm by hand
    ik-check                    no arm needed: FK of the taught poses, IK round trips, the reach map
    try ACTION [--x CM --y CM]  try hover, close, grasp or place on the arm, one step at a time

Poses the app uses: ``rest``, ``gripper_open``, ``gripper_closed``, and for arm tasks ``ready``
(the arm raised over its workspace, gripper pointing down). Gestures: e.g. ``wave``.
Config (environment, else ``.env``): ARM_PORT, ARM_ID (default ``follower``),
LEADER_PORT, LEADER_ID (default ``leader``), ARM_MOTIONS (default ``<repo>/arm_motions.json``).
Arm tasks: ARM_BASE_X_CM, ARM_BASE_Y_CM (the arm's shoulder-pan axis, cm forward of and to
the left of the centre of Reachy's foot) and ARM_BASE_YAW_DEG (where the arm points at
shoulder_pan 0, degrees counter-clockwise from Reachy's forward) — tasks are off without them;
ARM_BASE_Z_CM (0), ARM_KEEPOUT_CM (16), ARM_JOINT_OFFSETS_DEG (0,0,0,0,0: added to LeRobot
angles to get the URDF's), ARM_ROLL_LIMIT_DEG (90), ARM_URDF (<repo>/third_party/so101/...).

Protocol (``serve``): the server first prints ``{"ready": true, "arm": ID, "actions": [...]}``
(or ``{"ready": false, "error": ...}``). Each request ``{"id": 1, "action": "wave"}`` is
queued and run in order; it gets ``{"id": 1, "ok": true}`` or ``{"id": 1, "ok": false,
"error": ...}`` when it finishes (``"cancelled": true`` too if a stop or a fault cancelled
it). ``stop`` jumps the queue: it interrupts the running action, cancels the queued ones
and freezes the arm in place. ``shutdown`` (or closing stdin) does the same, then parks
the arm at rest and exits. A motor fault (a servo's overload, over-current or overheat
protection tripping) also interrupts and cancels everything, freezes the arm, and is
reported as ``{"event": "fault", "arm": ID, "faults": {motor: [kinds]}, "interrupted":
bool, "cancelled": n, "detail": ...}`` — even when nothing was running.

Arm tasks take ``"args"`` in the table frame T (metres; origin on the table under the centre
of Reachy's foot, x Reachy's forward, y its left, z up). They are listed in ``actions`` only
when the IK loaded, the mount is set and ``ready`` (with rest and the gripper poses) is taught:
``hover {x, y, height?}`` (a test: the gripper 10 cm above the spot for 2 s, then back to rest;
with a place's ``height``, at least as high as that place would start down),
``grasp {x, y, yaw_deg?, label?}`` → ``{"ok": true, "holding": bool}`` (ends lifted above the
spot, holding it), ``place {x, y, height}`` (``height``: where the jaw tips let go, above the
table; ends at rest). A task that can't be done safely is refused before anything moves:
``{"ok": false, "executed": false, "error": ...}``. While the arm holds something, rest,
gestures and gripper moves are refused, ``stop`` (and a fault) freezes only the arm joints so
the gripper keeps its grip, ``shutdown`` first puts the object back where it was picked up,
``status`` says ``"holding": label``, and a drop is announced as ``{"event": "dropped", "arm":
ID, "label": ...}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[2]

_TICK_S = 0.02     # 50 Hz command rate while moving
_SPEED = 45.0      # joint speed for moves: degrees/s (gripper: percent/s)
_MIN_MOVE_S = 0.4
_MAX_STEP = 10.0   # LeRobot max_relative_target: never command a joint >10° from where it is
_RETRY = 3        # Feetech buses occasionally return a corrupted status packet
_REST_TOL = 4.0   # a joint within this many degrees counts as having reached rest
_SETTLE_S = 1.5
_RECORD_HZ = 30
_QUEUE_MAX = 8        # moves allowed to wait behind the running one
# How often to read the servos' status. The servos protect themselves in hardware (an
# overload takes ~2 s of stalling to trip); these only set how soon the queue is cleared,
# the arm frozen and the app told.
_FAULT_POLL_S = 0.25  # while moving
_IDLE_POLL_S = 1.0    # while holding a pose (torque on, nothing running); never when parked

# Feetech status bits (the Status register, and the error byte of every reply).
_FAULT_BITS = {0x20: "overload", 0x08: "overcurrent", 0x04: "overheat"}
# The same faults as LeRobot reports them when a write's reply carries the bit.
_FAULT_ERRORS = {"Overload error": "overload", "OverEle error": "overcurrent", "Overheat error": "overheat"}

# Arm tasks. Joint angles are LeRobot degrees; heights are of the jaw tips above the table.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_PAN_AXIS = (0.0388, 0.0, -0.0024)  # URDF base_link: the shoulder-pan axis (x, y), the base's underside (z)
# Points checked against the keep-out and the table: name -> URDF link (its origin).
_LINKS = {"elbow": "lower_arm_link", "wrist": "wrist_link", "hand": "gripper_link", "gripper": "gripper_frame_link"}
_JAW_BELOW_M = 0.0075  # the jaws reach this far past the gripper frame's origin (the fixed jaw's tip)
_JAW_CENTRE_M = 0.02   # the grasp centre sits about this far toward the moving jaw [unverified: tune at hover]
_IK_STEPS = 60         # LeRobot 0.6.1 runs one placo step per inverse_kinematics() call
_IK_POS_TOL = 0.002
_IK_AXIS_TOL = 3.0     # degrees between the approach reached and the one asked for
_IK_ORI_WEIGHT = 0.1   # vs position weight 1 (metres vs radians)
_IK_JUMP = 5.0         # a joint moving more than this between two IK waypoints = a branch flip
_YAW_SLACK = 10.0      # the jaws may close this far off the asked direction to keep the roll in its limit
_PITCHES = (90.0, 75.0, 60.0)  # approach, degrees below horizontal: top-down first
_MAX_RADIUS = {90.0: 0.26, 75.0: 0.30, 60.0: 0.31}  # target distance from the pan axis, per approach
_MIN_RADIUS = 0.10
_HOVER_Z = 0.06        # grasp: hover with the jaw tips this high, then go down...
_GRASP_Z = 0.004       # ...to this
_STEP_M = 0.005        # IK waypoints on a straight line up or down
_TRANSIT_Z = 0.03      # every arm point stays this high on its way somewhere...
_CONTACT_Z = 0.003     # ...but the jaw tips and the gripper may go down to this near the table
_TEST_HOVER_Z = 0.10   # hover (a test): the gripper origin this high
_PLACE_ABOVE = 0.03    # place: come down from this far above the release height
_GUARDED_BELOW = 0.03  # place: a release this low is near the table, so come down guarded
_MAX_RELEASE = 0.15
_PUT_BACK_Z = 0.01
_TASK_SPEED = 30.0     # joint moves in arm tasks, degrees/s on average (peaks at 0.9° per tick)
_DESCENT_MPS = 0.02
_LIFT_MPS = 0.04
_MAX_DEG_TICK = 1.0
_SETTLE_TASK_S = 0.3
_HOVER_HOLD_S = 10.0  # long enough to check by eye where the jaws are
# Guarded descent: a contact is a joint's load or lag moving this far from its value at the
# hover, for a few ticks in a row.
_GUARD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
_GUARD_LOAD = 150      # Present_Load: 0.1 % of PWM duty
_GUARD_LAG = 3.0
_GUARD_TICKS = 3
_BACK_OFF_M = 0.003
# Compliant close (gripper units, 0-100). No register writes: the squeeze is set by how far
# the goal sits past where the jaws stopped, which keeps it far below the overload threshold.
_CLOSE_STEP = 0.5
_GRIP_LAG = 2.5
_GRIP_LOAD = 120
_SQUEEZE = 1.5
_HOLD_LOAD = (40, 120)
_HOLD_TRIES = 4
_CLOSE_MAX_S = 3.0
_EMPTY_MARGIN = 1.5    # the gripper this close to its closed pose has nothing between the jaws
_HELD_POS, _HELD_LOAD = 1.0, 30  # after the lift: still holding something?
_EASE_LOAD = 200       # holding: a squeeze harder than this is eased off
_DROP_SLACK = 0.5      # holding: jaws this close to their hold goal with no load have nothing between them...
_DROP_CHECKS = 2       # ...on this many idle checks in a row (not one odd reading)
_TASK_POSES = ("ready", "rest", "gripper_open", "gripper_closed")  # arm tasks need all of these taught
_GRIPPER_REGISTERS = ("Torque_Limit", "Max_Torque_Limit", "Overload_Torque", "Protection_Current",
                      "Protection_Time", "P_Coefficient")


def _now() -> float:
    """The clock arm-task moves are paced by (a function, so tests can fake it)."""
    return time.monotonic()


class Stopped(Exception):
    """The current action was interrupted by ``stop``."""


class Refused(Exception):
    """An arm task that can't be done (safely); refused before anything moved."""


class MotorFault(Exception):
    """A servo's protection tripped; *faults* maps motor name -> fault kinds."""

    def __init__(self, faults: dict[str, list[str]]) -> None:
        self.faults = faults
        super().__init__("motor fault: " + ", ".join(f"{m} {'/'.join(k)}" for m, k in faults.items()))


def _freeze(arm: "Arm", faults: Optional[dict] = None) -> str:
    """Stop the arm pushing — only the motors that are powered: re-targeting a let-go
    motor would power it up again. While holding something, the gripper keeps its hold goal
    (unless it is the motor that faulted), so the object isn't dropped. Say how that went."""
    if not arm.powered:
        return "motors already off"
    motors = sorted(arm.powered)
    if arm.holding is not None and "gripper" not in (faults or {}):
        motors = [m for m in motors if m != "gripper"]
    try:
        if motors:
            arm.freeze(motors)
        return "arm frozen in place"
    except Exception as exc:
        return f"could not freeze the arm: {exc}"


def _merge_faults(*sets: Optional[dict]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for faults in sets:
        for motor, kinds in (faults or {}).items():
            merged[motor] = list(dict.fromkeys([*merged.get(motor, []), *kinds]))
    return merged


def _fault_from_error(exc: Exception, bus: Any) -> Optional[MotorFault]:
    """The MotorFault behind a LeRobot write error ("... on id_=5 ... Overload error!"), if any."""
    text = str(exc)
    kinds = [kind for marker, kind in _FAULT_ERRORS.items() if marker in text]
    if not kinds:
        return None
    match = re.search(r"id_=(\d+)", text)
    motor_id = int(match.group(1)) if match else None
    name = next((n for n, m in bus.motors.items() if m.id == motor_id), f"motor {motor_id}")
    return MotorFault({name: kinds})


def _arg(args: dict, key: str, required: bool = True) -> Optional[float]:
    """A number from a request's args (Refused if missing or not a number)."""
    value = args.get(key)
    if value is None:
        if required:
            raise Refused(f"missing argument: {key}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Refused(f"bad argument: {key} should be a number, not {value!r}")
    return float(value)


def _check_release(height: float) -> None:
    """Refused unless *height* (where the jaw tips let go, above the table) is in range."""
    if not _CONTACT_Z <= height <= _MAX_RELEASE:
        raise Refused(f"release height {height * 100:.1f} cm is out of range "
                      f"({_CONTACT_Z * 100:.1f} to {_MAX_RELEASE * 100:.0f} cm)")


# --------------------------------------------------------------------------- #
# Config and taught motions                                                    #
# --------------------------------------------------------------------------- #

def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader (python-dotenv isn't in .venv-lerobot)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def reach_settings(env: dict[str, str]) -> dict[str, Any]:
    """The arm-task settings (see the module docstring); ValueError if one is malformed.
    ``mount`` (metres, degrees) is None unless the base's x, y and yaw are all set."""

    def number(key: str, default: Optional[float] = None) -> Optional[float]:
        text = env.get(key, "").strip()
        if not text:
            return default
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise ValueError(f"{key} should be a number, not {text!r}")
        return value

    x, y, yaw = number("ARM_BASE_X_CM"), number("ARM_BASE_Y_CM"), number("ARM_BASE_YAW_DEG")
    z = number("ARM_BASE_Z_CM", 0.0)
    mount = None if None in (x, y, yaw) else {"x": x / 100, "y": y / 100, "z": z / 100, "yaw": yaw}
    text = env.get("ARM_JOINT_OFFSETS_DEG", "").strip() or "0,0,0,0,0"
    try:
        offsets = tuple(float(v) for v in text.split(","))
    except ValueError:
        offsets = ()
    if len(offsets) != len(ARM_JOINTS) or not all(math.isfinite(v) for v in offsets):
        raise ValueError("ARM_JOINT_OFFSETS_DEG should be 5 comma-separated degrees "
                         f"(pan,lift,elbow,wrist_flex,wrist_roll), not {text!r}")
    return {
        "mount": mount,
        "keepout_m": number("ARM_KEEPOUT_CM", 16.0) / 100,
        "offsets": offsets,
        "roll_limit_deg": number("ARM_ROLL_LIMIT_DEG", 90.0),
        "urdf": Path(env.get("ARM_URDF", "").strip() or REPO / "third_party" / "so101" / "so101_new_calib.urdf"),
    }


def load_config() -> dict[str, Any]:
    env = {**_read_dotenv(REPO / ".env"), **os.environ}
    try:
        reach, reach_error = reach_settings(env), None
    except ValueError as exc:
        reach, reach_error = None, str(exc)
    return {
        "port": env.get("ARM_PORT", "").strip(),
        "id": env.get("ARM_ID", "").strip() or "follower",
        "leader_port": env.get("LEADER_PORT", "").strip(),
        "leader_id": env.get("LEADER_ID", "").strip() or "leader",
        "motions": Path(env.get("ARM_MOTIONS", "").strip() or REPO / "arm_motions.json"),
        "reach": reach,
        "reach_error": reach_error,
    }


def load_motions(path: Path) -> dict[str, dict]:
    if path.is_file():
        data = json.loads(path.read_text())
        return {"poses": data.get("poses", {}), "gestures": data.get("gestures", {})}
    return {"poses": {}, "gestures": {}}


def save_motions(path: Path, motions: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(motions, indent=1))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Kinematics for arm tasks                                                      #
# --------------------------------------------------------------------------- #

def to_base(mount: dict, p: Any) -> tuple[float, float, float]:
    """A table-frame point (m) in the arm's URDF base_link: Rz(-yaw)·(p - mount) + the pan
    axis (base_link's +x is where the arm points at pan 0; its origin is 2.4 mm above the
    base's underside)."""
    th = math.radians(mount["yaw"])
    c, s = math.cos(th), math.sin(th)
    dx, dy, dz = p[0] - mount["x"], p[1] - mount["y"], p[2] - mount["z"]
    return (c * dx + s * dy + _PAN_AXIS[0], -s * dx + c * dy + _PAN_AXIS[1], dz + _PAN_AXIS[2])


def to_table(mount: dict, p: Any) -> tuple[float, float, float]:
    """The inverse of to_base: a base_link point in the table frame."""
    th = math.radians(mount["yaw"])
    c, s = math.cos(th), math.sin(th)
    dx, dy, dz = p[0] - _PAN_AXIS[0], p[1] - _PAN_AXIS[1], p[2] - _PAN_AXIS[2]
    return (c * dx - s * dy + mount["x"], s * dx + c * dy + mount["y"], dz + mount["z"])


def _densify(path: list[dict], max_deg: float = 1.0) -> list[dict]:
    """*path* with joint-space samples added so no joint moves more than *max_deg* between two."""
    out = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(max(abs(b[j] - a[j]) for j in ARM_JOINTS) / max_deg))
        out += [{j: a[j] + (b[j] - a[j]) * k / n for j in ARM_JOINTS} for k in range(1, n + 1)]
    return out


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _origin(x: float, y: float, tips_z: float, pitch: float) -> tuple[float, float, float]:
    """Where the gripper origin goes for the jaw tips to be *tips_z* above the table at *pitch*."""
    return (x, y, tips_z + _JAW_BELOW_M * math.sin(math.radians(pitch)))


class Reach:
    """FK, IK and safety checks for arm tasks, on the SO-101 URDF (LeRobot's RobotKinematics,
    which runs on placo) and the arm's mount on the table. Joints are LeRobot degrees (the
    URDF's minus ARM_JOINT_OFFSETS_DEG); points are in the table frame T unless named ``_b``
    (URDF base_link). The target frame is ``gripper_frame_link``: its origin is the fixed
    jaw's tip, its +z the approach direction, its x the jaws' closing direction."""

    def __init__(self, urdf: Path | str, mount: Optional[dict] = None, offsets: Any = (0.0,) * 5,
                 keepout_m: float = 0.16, roll_limit_deg: float = 90.0) -> None:
        if not Path(urdf).is_file():
            raise FileNotFoundError(f"no URDF at {urdf} (see the README: third_party/so101)")
        from lerobot.model.kinematics import RobotKinematics

        self.kin = RobotKinematics(str(urdf), "gripper_frame_link", list(ARM_JOINTS))
        self.mount = mount
        self.offsets = dict(zip(ARM_JOINTS, (float(v) for v in offsets)))
        self.keepout = keepout_m
        self.roll_limit = roll_limit_deg
        self._lim_u = [tuple(math.degrees(v) for v in self.kin.robot.get_joint_limits(j)) for j in ARM_JOINTS]
        self.limits = {j: (lo - self.offsets[j], hi - self.offsets[j]) for j, (lo, hi) in zip(ARM_JOINTS, self._lim_u)}
        self.model_limits = dict(self.limits)

    def allow_taught(self, poses: dict) -> None:
        """Widen each joint's allowed range to take in the taught rest and ready poses. The
        CAD model's limits (±100° on the shoulder) are narrower than the real arm's — it
        rests at -107° — but a pose the arm holds is reachable, and so is every angle
        between it and a target inside the model's range. IK targets stay inside it."""
        for name in ("rest", "ready"):
            pose = poses.get(name) or {}
            for j in ARM_JOINTS:
                if j in pose:
                    lo, hi = self.limits[j]
                    self.limits[j] = (min(lo, pose[j]), max(hi, pose[j]))

    # FK ------------------------------------------------------------------- #

    def _urdf(self, q: dict) -> Any:
        import numpy as np

        return np.array([q[j] + self.offsets[j] for j in ARM_JOINTS], dtype=float)

    def points_b(self, q: dict) -> dict[str, tuple]:
        """base_link positions at *q* of the links checked for keep-out ("gripper" is the
        gripper origin) and of the jaw tips."""
        T = self.kin.forward_kinematics(self._urdf(q))
        pts = {name: tuple(self.kin.robot.get_T_world_frame(link)[:3, 3]) for name, link in _LINKS.items()}
        pts["jaw tips"] = tuple(T[:3, 3] + _JAW_BELOW_M * T[:3, 2])
        return pts

    def points(self, q: dict) -> dict[str, tuple]:
        """Like points_b, in the table frame."""
        return {name: to_table(self.mount, p) for name, p in self.points_b(q).items()}

    def tip_z(self, q: dict) -> float:
        """Height of the jaw tips above the table at *q*."""
        return self.points(q)["jaw tips"][2]

    def pitch(self, q: dict) -> float:
        """The approach at *q*, degrees below horizontal."""
        a = self.kin.forward_kinematics(self._urdf(q))[:3, 2]
        return math.degrees(math.asin(max(-1.0, min(1.0, -float(a[2])))))

    # IK ------------------------------------------------------------------- #

    def seed(self, p: Any, current: dict) -> dict:
        """The first IK guess for a target at *p*: turned toward it, top-down, the wrist roll
        as it is now. Never rest: its shoulder_lift is outside the model's ±100°."""
        x_b, y_b, _ = to_base(self.mount, p)
        pan = -math.degrees(math.atan2(y_b, x_b - _PAN_AXIS[0])) - self.offsets["shoulder_pan"]
        return {"shoulder_pan": pan, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 70.0,
                "wrist_roll": current["wrist_roll"]}

    def solve(self, p: Any, pitch_deg: float, jaw_yaw_deg: Optional[float], seed: dict,
              centre_m: float = 0.0) -> Optional[dict]:
        """Joints putting the gripper origin at *p* with the approach tilted *pitch_deg* below
        horizontal in the arm's vertical plane; None if the IK can't get within 2 mm and 3°.
        *jaw_yaw_deg* is the jaws' closing direction in T; the jaws are symmetric, so it tries
        both ways round and keeps the one whose wrist roll is within ±ARM_ROLL_LIMIT_DEG. None
        keeps the seed's roll. *centre_m* shifts the origin along the closing direction so
        the point between the jaws, not the fixed jaw's tip, lands on *p*."""
        import numpy as np

        p_b = np.array(to_base(self.mount, p))
        start = self._urdf(seed)
        x_seed = self.kin.forward_kinematics(start)[:3, 0]
        if jaw_yaw_deg is None:
            q = self._ik(p_b, pitch_deg, x_seed, start, centre_m)
            return q if q is not None and abs(q["wrist_roll"]) <= self.roll_limit else None
        # Rolling the wrist turns the jaws by the same angle about the approach axis.
        seed_yaw = math.degrees(math.atan2(x_seed[1], x_seed[0]))
        best = None
        for psi in (jaw_yaw_deg - self.mount["yaw"], jaw_yaw_deg - self.mount["yaw"] + 180.0):
            q = self._ik_yaw(p_b, pitch_deg, psi, start, start[4] + _wrap(psi - seed_yaw), centre_m)
            excess = abs(q["wrist_roll"]) - self.roll_limit if q is not None else 0.0
            if 0.0 < excess <= _YAW_SLACK:
                # Jaws square to the reach need a roll near ±90°: close a little off the asked
                # direction rather than past the roll limit.
                shift = math.copysign(excess + 0.5, q["wrist_roll"])
                q = self._ik_yaw(p_b, pitch_deg, psi - shift, start,
                                 q["wrist_roll"] + self.offsets["wrist_roll"] - shift, centre_m)
            if q is None or abs(q["wrist_roll"]) > self.roll_limit:
                continue
            if best is None or abs(q["wrist_roll"]) < abs(best["wrist_roll"]):
                best = q
        return best

    def _ik_yaw(self, p_b: Any, pitch_deg: float, psi: float, start: Any, roll: float, centre_m: float) -> Optional[dict]:
        """_ik with the jaws closing along base_link azimuth *psi*, from *start* with its roll set to *roll*."""
        import numpy as np

        h = np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi)), 0.0])
        guess = start.copy()
        guess[4] = roll
        return self._ik(p_b, pitch_deg, h, guess, centre_m)

    @staticmethod
    def _target(p_b: Any, pitch_deg: float, phi: float, h: Any, centre_m: float) -> Any:
        """The 4x4 gripper pose: approach *pitch_deg* down along azimuth *phi*, jaws closing
        along *h* (made square to the approach)."""
        import numpy as np

        pt = math.radians(pitch_deg)
        a = np.array([math.cos(pt) * math.cos(phi), math.cos(pt) * math.sin(phi), -math.sin(pt)])
        x = h - (h @ a) * a
        if np.linalg.norm(x) < 1e-6:  # the jaws can't close along the approach: any square direction
            x = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(a, x), a])
        T[:3, 3] = p_b
        if centre_m:
            flat = np.array([x[0], x[1], 0.0])
            if np.linalg.norm(flat) > 1e-6:
                T[:3, 3] = p_b + centre_m * flat / np.linalg.norm(flat)
        return T

    def _ik(self, p_b: Any, pitch_deg: float, h: Any, guess: Any, centre_m: float) -> Optional[dict]:
        """Iterate LeRobot's one-step IK. The approach must lie in the arm's vertical plane,
        which turns with shoulder_pan (positive = clockwise), and the gripper origin sits off
        the roll axis — so the target's azimuth follows the pan each step."""
        import numpy as np

        lo, hi = np.array([v[0] for v in self._lim_u]), np.array([v[1] for v in self._lim_u])
        q = np.clip(guess, lo, hi)
        phi = math.atan2(p_b[1], p_b[0] - _PAN_AXIS[0])
        for _ in range(_IK_STEPS):
            T = self._target(p_b, pitch_deg, phi, h, centre_m)
            new = self.kin.inverse_kinematics(q, T, position_weight=1.0, orientation_weight=_IK_ORI_WEIGHT)
            step, q = float(np.max(np.abs(new - q))), new
            new_phi = -math.radians(float(q[0]))
            if step < 1e-3 and abs(new_phi - phi) < 1e-5:
                break
            phi = new_phi
        T = self._target(p_b, pitch_deg, phi, h, centre_m)
        F = self.kin.forward_kinematics(q)
        axis = math.degrees(math.acos(max(-1.0, min(1.0, float(F[:3, 2] @ T[:3, 2])))))
        if np.linalg.norm(F[:3, 3] - T[:3, 3]) > _IK_POS_TOL or axis > _IK_AXIS_TOL:
            return None
        return {j: float(q[i]) - self.offsets[j] for i, j in enumerate(ARM_JOINTS)}

    def line(self, xy: tuple, z_from: float, z_to: float, pitch: float, start: dict,
             centre_m: float = _JAW_CENTRE_M, partial: bool = False) -> tuple[list, list] | str:
        """IK waypoints (at most 5 mm apart) taking the jaw tips straight up or down at *xy*
        from *z_from* (where *start* already is) to *z_to*, keeping start's jaw direction:
        (joints, jaw-tip heights), or why not. *partial*: stop at the last reachable one."""
        n = max(1, math.ceil(abs(z_to - z_from) / _STEP_M - 1e-9))
        path, zs = [start], [z_from]
        for i in range(1, n + 1):
            z = z_from + (z_to - z_from) * i / n
            q = self.solve(_origin(xy[0], xy[1], z, pitch), pitch, None, path[-1], centre_m)
            if q is None and partial:
                break
            if q is None:
                return f"out of reach: no IK solution with the jaws {z * 100:.1f} cm above the table ({pitch:.0f}° approach)"
            jump = max(abs(q[j] - path[-1][j]) for j in ARM_JOINTS)
            if jump > _IK_JUMP:
                return f"the IK jumps {jump:.0f}° between two waypoints (max {_IK_JUMP:.0f})"
            path.append(q)
            zs.append(z)
        return path, zs

    # Checks ---------------------------------------------------------------- #

    def check(self, path: list[dict], contact: bool = False, floor: float = 0.0) -> Optional[str]:
        """Why moving through *path* (sampled so no joint moves more than 1° between samples)
        is unsafe, or None: every sample inside the model's joint limits and the wrist-roll
        limit, every checked arm point (the elbow, wrist and gripper origins, the jaw tips)
        outside the keep-out around Reachy and at least 3 cm above the table — on a *contact*
        move the gripper and jaw tips may go down to 3 mm. The jaw tips also stay above
        *floor*. Only those points: the arm's shells reach up to ~2.5 cm past them."""
        for q in _densify(path):
            for j in ARM_JOINTS:
                lo, hi = self.limits[j]
                if not lo - 0.01 <= q[j] <= hi + 0.01:
                    return f"{j} would reach {q[j]:.0f}° (the arm model allows {lo:.0f} to {hi:.0f})"
            if abs(q["wrist_roll"]) > self.roll_limit + 0.01:
                return f"the wrist roll would reach {q['wrist_roll']:.0f}° (limit ±{self.roll_limit:.0f})"
            points = self.points(q)
            if points["jaw tips"][2] < floor - 1e-4:
                return (f"too low: the jaw tips would dip to {points['jaw tips'][2] * 100:.1f} cm above the "
                        f"table on the way (min {floor * 100:.1f})")
            for name, (x, y, z) in points.items():
                d = math.hypot(x, y)
                if d < self.keepout:
                    return (f"too close to Reachy: the arm's {name} would pass {d * 100:.0f} cm from "
                            f"its foot (keep-out {self.keepout * 100:.0f})")
                low = _CONTACT_Z if contact and name in ("gripper", "jaw tips") else _TRANSIT_Z
                if z < low - 1e-4:
                    return f"too low: the arm's {name} would be {z * 100:.1f} cm above the table (min {low * 100:.1f})"
        return None

    def pose_warnings(self, poses: dict) -> list[str]:
        """Warnings for taught rest/ready poses that break the keep-out around Reachy."""
        out = []
        for name in ("rest", "ready"):
            if name not in poses or self.mount is None:
                continue
            for part, (x, y, _) in self.points(poses[name]).items():
                d = math.hypot(x, y)
                if d < self.keepout:
                    out.append(f"warning: the taught '{name}' pose puts the arm's {part} {d * 100:.0f} cm "
                               f"from Reachy's foot (keep-out {self.keepout * 100:.0f} cm)")
                    break
        return out

    # Plans ------------------------------------------------------------------ #

    def _per_pitch(self, x: float, y: float, attempt: Callable[[float], Any]) -> dict:
        """The first plan *attempt(pitch)* returns, top-down first; Refused with why it failed
        at the first approach tried."""
        d = math.hypot(x, y)
        if d < self.keepout:
            raise Refused(f"too close to Reachy: {d * 100:.0f} cm from its foot (keep-out {self.keepout * 100:.0f})")
        x_b, y_b, _ = to_base(self.mount, (x, y, 0.0))
        r = math.hypot(x_b - _PAN_AXIS[0], y_b - _PAN_AXIS[1])
        top = max(_MAX_RADIUS.values())
        if r > top:
            raise Refused(f"out of reach: {r * 100:.0f} cm from the arm's base (max {top * 100:.0f})")
        if r < _MIN_RADIUS:
            raise Refused(f"too close to the arm's base: {r * 100:.0f} cm (min {_MIN_RADIUS * 100:.0f})")
        reasons = []
        for pitch in _PITCHES:
            if r <= _MAX_RADIUS[pitch]:
                plan = attempt(pitch)
                if isinstance(plan, dict):
                    return plan
                reasons.append(plan)
        raise Refused(reasons[0])

    def plan_grasp(self, x: float, y: float, jaw_yaw: Optional[float], current: dict,
                   ready: Optional[dict] = None) -> dict:
        """ready → hover with the jaw tips 6 cm above (x, y) → straight down to 4 mm (and back
        up the same way): {"pitch", "hover", "down": [joints], "down_z": [jaw-tip heights]}."""

        def attempt(pitch: float) -> dict | str:
            hover = self.solve(_origin(x, y, _HOVER_Z, pitch), pitch, jaw_yaw, self.seed((x, y, 0.0), current),
                               _JAW_CENTRE_M)
            if hover is None:
                return f"out of reach: no IK solution with the jaws {_HOVER_Z * 100:.0f} cm above it ({pitch:.0f}° approach)"
            line = self.line((x, y), _HOVER_Z, _GRASP_Z, pitch, hover)
            if isinstance(line, str):
                return line
            down, zs = line
            why = (ready is not None and self.check([ready, hover])) or self.check(down, contact=True)
            return why or {"pitch": pitch, "hover": hover, "down": down, "down_z": zs}

        return self._per_pitch(x, y, attempt)

    def plan_place(self, x: float, y: float, height: float, current: dict, ready: Optional[dict] = None) -> dict:
        """From *current* (holding): straight up to a transit height → over (x, y) with the jaw
        tips 3 cm above *height* → down to *height*; afterwards back up 3 cm, then to ready.
        {"pitch", "up", "up_z", "hover", "down", "down_z"}. The lift goes as high as the hover
        over the target where it can (a top-down approach can't lift far), at least 6 cm; on
        the way over, the jaw tips never dip below the lower of the two ends."""
        now = self.points(current)
        z_now = now["jaw tips"][2]
        z_min = max(z_now, _HOVER_Z)
        z_up = max(z_min, height + _PLACE_ABOVE)
        if z_up > z_now + 1e-4:
            line = self.line(now["gripper"][:2], z_now, z_up, self.pitch(current), current, 0.0, partial=True)
            if isinstance(line, str) or line[1][-1] < z_min - 1e-4:
                why = line if isinstance(line, str) else f"no IK solution above {line[1][-1] * 100:.1f} cm"
                raise Refused(f"can't lift what it holds to {z_min * 100:.0f} cm: {why}")
            up, up_z = line
        else:
            up, up_z = [current], [z_now]
        why = self.check(up, contact=True)
        if why:
            raise Refused(why)

        def attempt(pitch: float) -> dict | str:
            hover = self.solve(_origin(x, y, height + _PLACE_ABOVE, pitch), pitch, None,
                               self.seed((x, y, 0.0), current), _JAW_CENTRE_M)
            if hover is None:
                return (f"out of reach: no IK solution with the jaws {(height + _PLACE_ABOVE) * 100:.0f} cm "
                        f"above it ({pitch:.0f}° approach)")
            line = self.line((x, y), height + _PLACE_ABOVE, height, pitch, hover)
            if isinstance(line, str):
                return line
            down, zs = line
            floor = min(up_z[-1], height + _PLACE_ABOVE) - 0.005
            why = (self.check([up[-1], hover], floor=floor) or self.check(down, contact=True)
                   or (ready is not None and self.check([hover, ready])))
            return why or {"pitch": pitch, "up": up, "up_z": up_z, "hover": hover, "down": down, "down_z": zs}

        return self._per_pitch(x, y, attempt)

    def plan_hover(self, x: float, y: float, current: dict, ready: Optional[dict] = None,
                   height: Optional[float] = None, tips: Optional[float] = None) -> dict:
        """ready → the gripper origin 10 cm above (x, y) — or the jaw tips *tips* above the
        table — → back: {"pitch", "hover"}. With *height* (a place's release height), at
        least as high as that place's hover: the jaw tips 3 cm above it (over a container,
        6 cm above its rim), and on the way there they never dip below the lower of the two
        ends."""

        def attempt(pitch: float) -> dict | str:
            z = _TEST_HOVER_Z if tips is None else _origin(x, y, tips, pitch)[2]
            if height is not None:
                z = max(z, _origin(x, y, height + _PLACE_ABOVE, pitch)[2])
            hover = self.solve((x, y, z), pitch, None, self.seed((x, y, 0.0), current), _JAW_CENTRE_M)
            if hover is None:
                return f"out of reach: no IK solution {z * 100:.0f} cm above it ({pitch:.0f}° approach)"
            if ready is None:
                return {"pitch": pitch, "hover": hover}
            floor = 0.0 if height is None else min(self.tip_z(ready), height + _PLACE_ABOVE) - 0.005
            return self.check([ready, hover], floor=floor) or {"pitch": pitch, "hover": hover}

        return self._per_pitch(x, y, attempt)


def _make_reach(settings: dict, poses: Optional[dict] = None) -> Reach:
    reach = Reach(settings["urdf"], settings["mount"], settings["offsets"], settings["keepout_m"],
                  settings["roll_limit_deg"])
    reach.allow_taught(poses or {})
    return reach


def _load_reach(cfg: dict, motions: dict) -> Optional[Reach]:
    """The IK for arm tasks, or None with one stderr line saying why (tasks need the mount
    in .env, LeRobot's kinematics with placo, and the URDF). Warns about taught rest/ready
    poses inside the keep-out."""
    settings = cfg.get("reach")
    if settings is None:
        print(f"arm tasks off: {cfg.get('reach_error') or 'no arm-task settings'}", file=sys.stderr)
        return None
    if settings["mount"] is None:
        print("arm tasks off: set ARM_BASE_X_CM, ARM_BASE_Y_CM and ARM_BASE_YAW_DEG to turn them on", file=sys.stderr)
        return None
    try:
        reach = _make_reach(settings, motions["poses"])
    except Exception as exc:
        print(f"arm tasks off: couldn't load the IK ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    for warning in reach.pose_warnings(motions["poses"]):
        print(warning, file=sys.stderr)
    return reach


# --------------------------------------------------------------------------- #
# The arm                                                                       #
# --------------------------------------------------------------------------- #

class Arm:
    """Moves a connected LeRobot SO-101 follower through taught poses and gestures, and (with
    a Reach) through arm tasks: hover, grasp and place."""

    def __init__(self, robot: Any, motions: dict, reach: Optional[Reach] = None) -> None:
        self.robot = robot
        self.bus = robot.bus
        self.motions = motions
        self.reach = reach
        self.abort = threading.Event()
        self.powered: set[str] = set()  # motors that may be driven (so a fault is possible)
        self.holding: Optional[dict] = None  # {x, y, yaw, pitch, label} while it holds something
        self._grip_goal: Optional[float] = None  # the gripper's goal while holding
        self._gripping = False  # the gripper may be closed on something
        self._out = False  # an arm task took the arm away from rest
        self._slack = 0  # idle checks in a row that found the held object's jaws slack
        self._next_check = 0.0

    @property
    def torque_on(self) -> bool:
        return bool(self.powered)

    def tasks_on(self) -> bool:
        """Arm tasks can run: the IK, the mount and every pose they use (the app turns them
        on only when ``actions`` lists grasp, so this must agree with _task())."""
        poses = self.motions["poses"]
        return (self.reach is not None and self.reach.mount is not None
                and all(name in poses for name in _TASK_POSES))

    def actions(self) -> list[str]:
        poses = self.motions["poses"]
        names = [n for n in ("rest",) if n in poses]
        names += [a for a, p in (("gripper_open", "gripper_open"), ("gripper_close", "gripper_closed")) if p in poses]
        tasks = ["hover", "grasp", "place"] if self.tasks_on() else []
        return names + sorted(self.motions["gestures"]) + tasks + ["stop"]

    def pose(self, name: str) -> dict[str, float]:
        if name not in self.motions["poses"]:
            raise RuntimeError(f"pose '{name}' isn't taught yet; run: scripts/arm.sh teach {name}")
        return self.motions["poses"][name]

    def freeze(self, motors: Optional[str | list[str]] = None) -> None:
        """Set the target to where the joints are now: nothing pushes any more (this also
        clears a tripped Feetech protection). A Feetech servo turns its torque on when it
        gets a target, so this alone holds the pose — and needs no reply from a servo
        that is in a fault state. (So it powers up any motor it's given.)"""
        names = list(self.bus.motors) if motors is None else [motors] if isinstance(motors, str) else motors
        self.bus.sync_write("Goal_Position", self.bus.sync_read("Present_Position", names, num_retry=_RETRY))
        self.powered.update(names)

    def release(self) -> None:
        """Turn torque off on every motor, one by one, so a bad reply on one doesn't stop
        the others. A reply that only reports the servo's status (a fault bit) means the
        write landed. A reply still lost after retries leaves that motor counted as
        powered; the first such error is raised once every motor has been tried."""
        first_error: Optional[Exception] = None
        for motor in self.bus.motors:
            try:
                self.bus.disable_torque(motor, num_retry=_RETRY)
            except Exception as exc:
                if not (isinstance(exc, RuntimeError) and "[RxPacketError]" in str(exc)):
                    first_error = first_error or exc
                    continue
            self.powered.discard(motor)
        if first_error is not None:
            raise first_error

    def hold(self, motors: Optional[str | list[str]] = None) -> None:
        """Turn torque on where the joints are now. The target is set to the present
        position first, so a joint moved by hand while relaxed can't jump back."""
        self.freeze(motors)
        self.bus.enable_torque(motors)

    def stop_here(self) -> None:
        """``stop``: freeze in place, torque on. While holding something only the arm joints
        freeze; the gripper keeps its squeeze, so the object isn't dropped."""
        self.hold(list(ARM_JOINTS) if self.holding is not None else None)

    def check_faults(self) -> None:
        """Raise MotorFault if any servo's protection has tripped."""
        status = self.bus.sync_read("Status", normalize=False, num_retry=_RETRY)
        faults = {m: [kind for bit, kind in _FAULT_BITS.items() if s & bit] for m, s in status.items()}
        faults = {m: kinds for m, kinds in faults.items() if kinds}
        if faults:
            raise MotorFault(faults)

    def _watch(self) -> None:
        """Between commands while moving: honour ``stop`` and catch faults (status read at
        most every _FAULT_POLL_S — a Feetech overload takes ~2 s of stalling to trip)."""
        if self.abort.is_set():
            raise Stopped()
        now = time.monotonic()
        if now >= self._next_check:
            self._next_check = now + _FAULT_POLL_S
            self.check_faults()

    def _wait(self, seconds: float) -> None:
        """Keep still for *seconds*, still honouring stop and watching for faults."""
        for _ in range(max(1, round(seconds / _TICK_S))):
            self._watch()
            time.sleep(_TICK_S)

    def move_to(self, target: dict[str, float], speed: float = _SPEED) -> None:
        """Move smoothly from the present position to *target* (only the joints given)."""
        present = self.bus.sync_read("Present_Position", list(target), num_retry=_RETRY)
        dist = max(abs(target[m] - present[m]) for m in target)
        steps = max(1, round(max(_MIN_MOVE_S, dist / speed) / _TICK_S))
        for i in range(1, steps + 1):
            self._watch()
            a = i / steps
            a = a * a * (3 - 2 * a)  # ease in and out
            self.robot.send_action({f"{m}.pos": present[m] + (target[m] - present[m]) * a for m in target})
            time.sleep(_TICK_S)

    def rest(self, speed: float = _SPEED, gripper: bool = True) -> None:
        """Park at the taught rest pose, then turn the motors off. With *gripper* False the
        gripper stays where it is: the rest pose has it closed, and after letting go of
        something it may still sit between the jaws."""
        target = self.pose("rest")
        if not gripper:
            target = {j: target[j] for j in ARM_JOINTS}
        self.hold()
        self.move_to(target, speed)
        self.settle(target)
        self.release()
        self._out = False

    def park(self) -> None:
        """Rest — first letting go of anything the gripper may be closed on, and then leaving
        the jaws open (rest would close them on it again), and by way of ready if an arm task
        left the arm out (rest's taught path starts there)."""
        let_go = self._gripping
        if let_go:
            self.gripper("gripper_open")
            self.holding = self._grip_goal = None
            self._gripping = False
        if self._out and "ready" in self.motions["poses"]:
            self.hold()
            self.move_to(self._ready(), _TASK_SPEED)
            self.rest(_TASK_SPEED, gripper=not let_go)
        else:
            self.rest(gripper=not let_go)

    def settle(self, target: dict[str, float]) -> None:
        """Wait until every joint is within _REST_TOL of *target*. Otherwise raise and keep
        the motors on, so a lagging or blocked arm is never let go short of rest."""
        deadline = time.monotonic() + _SETTLE_S
        while True:
            present = self.bus.sync_read("Present_Position", list(target), num_retry=_RETRY)
            if all(abs(present[m] - target[m]) <= _REST_TOL for m in target):
                return
            if time.monotonic() > deadline:
                # Its targets are still past whatever blocks it: stop pushing, then report.
                try:
                    self.freeze(sorted(self.powered) or None)
                    note = "holding it in place"
                except Exception as exc:
                    note = f"could not freeze it: {exc}"
                raise RuntimeError(f"the arm didn't reach rest (blocked?); {note}")
            self._watch()
            self.robot.send_action({f"{m}.pos": v for m, v in target.items()})
            time.sleep(_TICK_S)

    def gripper(self, pose_name: str) -> None:
        target = {"gripper": self.pose(pose_name)["gripper"]}
        self.hold("gripper")
        self.move_to(target)

    def gesture(self, name: str) -> None:
        if name not in self.motions["gestures"]:
            raise RuntimeError(f"gesture '{name}' isn't recorded yet; run: scripts/arm.sh record {name}")
        self.pose("rest")  # it ends at rest, so fail before moving if rest isn't taught
        if self._gripping or self._out:
            # A stopped arm task left the arm out, maybe closed on something its frames would
            # squeeze at full force: let go and park first, as rest does.
            self.park()
        frames = self.motions["gestures"][name]["frames"]
        self.hold()
        self.move_to(frames[0][1])
        t0, start = frames[0][0], time.monotonic()
        for t, pos in frames[1:]:
            self._watch()
            delay = (t - t0) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
            self.robot.send_action({f"{m}.pos": v for m, v in pos.items()})
        self.rest()

    def run(self, action: str, args: Optional[dict] = None) -> Optional[dict]:
        """Run one requested action; returns extra fields for its reply, if any."""
        args = args or {}
        if self.holding is not None and action != "place":
            raise Refused(f"holding the {self.holding['label']}: put it down first")
        if action == "rest":
            self.park()
        elif action == "gripper_open":
            self.gripper("gripper_open")
        elif action == "gripper_close":
            self.gripper("gripper_closed")
        elif action == "hover":
            self.hover(args)
        elif action == "grasp":
            return self.grasp(args)
        elif action == "place":
            self.place(args)
        else:
            self.gesture(action)
        return None

    # Arm tasks -------------------------------------------------------------- #

    def _ready(self) -> dict[str, float]:
        ready = self.pose("ready")
        return {j: ready[j] for j in ARM_JOINTS}

    def _task(self) -> tuple[Reach, dict[str, float]]:
        """The IK and the ready pose, or Refused if arm tasks can't run."""
        if self.reach is None or self.reach.mount is None:
            raise Refused("arm tasks are off: the IK didn't load or the arm's mount (ARM_BASE_*) isn't set")
        for name in _TASK_POSES:
            if name not in self.motions["poses"]:
                raise Refused(f"pose '{name}' isn't taught yet; run: scripts/arm.sh teach {name}")
        return self.reach, self._ready()

    def _joints(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position", list(ARM_JOINTS), num_retry=_RETRY)

    def _goals(self) -> dict[str, float]:
        """Where the arm joints were last sent (their Goal_Position)."""
        return self.bus.sync_read("Goal_Position", list(ARM_JOINTS), num_retry=_RETRY)

    def _gripper_now(self) -> tuple[float, float]:
        """The gripper's position and |Present_Load|."""
        pos = self.bus.sync_read("Present_Position", ["gripper"], num_retry=_RETRY)["gripper"]
        load = self.bus.sync_read("Present_Load", ["gripper"], num_retry=_RETRY)["gripper"]
        return pos, abs(load)

    def hover(self, args: dict) -> None:
        """A test without contact: ready → the gripper 10 cm above (x, y) for 10 s → ready → rest,
        with the jaws open as for a grasp, so the point between them shows where a grasp would
        close. With *height* (a place's release height) it hovers where that place would
        start down, if higher: 3 cm above it, clear of a container's rim."""
        x, y = _arg(args, "x"), _arg(args, "y")
        height = _arg(args, "height", required=False)
        tips = _arg(args, "tips", required=False)  # `try hover --above`: the jaw tips this high
        if height is not None:
            _check_release(height)
        reach, ready = self._task()
        plan = reach.plan_hover(x, y, self._joints(), ready, height, tips)
        if self._gripping:
            self.park()  # a stopped grasp may have left the jaws on something: let go first
        self.hold()
        self.gripper("gripper_open")  # the plan aims the point between the open jaws
        self._out = True
        self.move_to(ready, _TASK_SPEED)
        self.move_to(plan["hover"], _TASK_SPEED)
        self._wait(_HOVER_HOLD_S)
        self.move_to(ready, _TASK_SPEED)
        self.rest(_TASK_SPEED)

    def grasp(self, args: dict) -> dict:
        """Pick up the object at (x, y): everything is planned before anything moves. Opens the
        gripper, goes ready → hover → down (guarded), closes compliantly, lifts back to the
        hover. Ends there holding it, torque on; if nothing was gripped, opens and rests."""
        x, y = _arg(args, "x"), _arg(args, "y")
        yaw = _arg(args, "yaw_deg", required=False)
        label = str(args.get("label") or "object")
        reach, ready = self._task()
        plan = reach.plan_grasp(x, y, yaw, self._joints(), ready)
        open_pos = self.pose("gripper_open")["gripper"]
        self.hold()
        self._out = True
        self.move_to({"gripper": open_pos})
        self.move_to(ready, _TASK_SPEED)
        self.move_to(plan["hover"], _TASK_SPEED)
        self._wait(_SETTLE_TASK_S)
        path, zs = plan["down"], plan["down_z"]
        if self._follow(path, zs, _DESCENT_MPS, self._guard(plan["hover"])) is None:
            q, z, i = path[-1], zs[-1], len(path) - 1
        else:
            q, z, i = self._back_off(path, zs)
        self._gripping = True
        gripped = self._close_on_object({"x": x, "y": y, "yaw": yaw, "pitch": plan["pitch"], "label": label})
        self._follow([q] + path[i::-1], [z] + zs[i::-1], _LIFT_MPS)
        if gripped and self._still_holding():
            return {"holding": True}
        self.holding = self._grip_goal = None
        self.move_to({"gripper": open_pos})
        self._gripping = False
        self.move_to(ready, _TASK_SPEED)
        self.rest(_TASK_SPEED)
        return {"holding": False}

    def place(self, args: dict) -> None:
        """Put what it holds down at (x, y), letting go with the jaw tips *height* above the
        table: up to a transit height, over the spot, down (guarded when near the table), open,
        up 3 cm, ready, rest."""
        x, y, height = _arg(args, "x"), _arg(args, "y"), _arg(args, "height")
        if self.holding is None:
            raise Refused("not holding anything")
        _check_release(height)
        reach, ready = self._task()
        # From where the joints were sent, not where they sag to under the object's weight: the
        # sent pose is on the planned path (a sagged one isn't, and its IK jumps).
        plan = reach.plan_place(x, y, height, self._goals(), ready)
        open_pos = self.pose("gripper_open")["gripper"]
        self.hold(list(ARM_JOINTS))  # the gripper keeps its squeeze
        self._out = True
        self._follow(plan["up"], plan["up_z"], _LIFT_MPS)
        self.move_to(plan["hover"], _TASK_SPEED)
        path, zs = plan["down"], plan["down_z"]
        guard = None
        if height <= _GUARDED_BELOW:
            self._wait(_SETTLE_TASK_S)
            guard = self._guard(plan["hover"])
        s = self._follow(path, zs, _DESCENT_MPS, guard)
        if s is None:
            up, up_z = path[::-1], zs[::-1]
        else:  # it touched down early: let go there
            self.freeze(list(ARM_JOINTS))
            q, i = self._joints(), max(0, math.ceil(s) - 1)
            up, up_z = [q] + path[i::-1], [reach.tip_z(q)] + zs[i::-1]
        self.move_to({"gripper": open_pos})
        self.holding = self._grip_goal = None
        self._gripping = False
        self._follow(up, up_z, _LIFT_MPS)
        self.move_to(ready, _TASK_SPEED)
        self.rest(_TASK_SPEED)

    def _put_back(self) -> None:
        """Put what it holds back down where it was picked up."""
        self.place({"x": self.holding["x"], "y": self.holding["y"], "height": _PUT_BACK_Z})

    def shutdown(self) -> None:
        """Park for good: put down what it holds where it was picked up (if that can't be
        done, let go where it is, as for a gripper stopped mid-close), then rest."""
        if self.holding is not None:
            try:
                self._put_back()
                return
            except Refused as exc:
                print(f"couldn't put the {self.holding['label']} back ({exc}); letting go where it is",
                      file=sys.stderr)
        self.park()

    def _follow(self, path: list[dict], zs: list[float], mps: float,
                guard: Optional[Callable[[dict], bool]] = None) -> Optional[float]:
        """Drive the arm joints through *path* (IK waypoints; *zs* their jaw-tip heights),
        straight between waypoints, at most *mps* of jaw-tip travel and _MAX_DEG_TICK per tick
        on any joint. Paced by the clock, not by turns of the loop: each command is where the
        arm should be by now, so slow bus reads (the guard's) don't make it stop and jump.
        With *guard*, stop at the first contact it reports and return how far along the path
        that was (in waypoints); None once the path is done."""
        times = [0.0]  # when each waypoint is due
        for i in range(1, len(path)):
            a, b = path[i - 1], path[i]
            dq = max(abs(b[j] - a[j]) for j in ARM_JOINTS)
            times.append(times[-1] + max(abs(zs[i] - zs[i - 1]) / mps, dq / _MAX_DEG_TICK * _TICK_S, _TICK_S))
        start, tick = _now(), 0
        while True:
            self._watch()
            t = min(_now() - start, times[-1])
            i = next(k for k in range(1, len(times)) if t <= times[k]) if len(times) > 1 else 0
            if i:
                f = (t - times[i - 1]) / (times[i] - times[i - 1])
                a, b = path[i - 1], path[i]
                q = {j: a[j] + (b[j] - a[j]) * f for j in ARM_JOINTS}
            else:
                q, f = path[-1], 0.0
            self.robot.send_action({f"{j}.pos": v for j, v in q.items()})
            if guard is not None and guard(q):
                return max(0, i - 1) + f
            if t >= times[-1]:
                return None
            tick += 1
            time.sleep(max(0.0, start + tick * _TICK_S - _now()))

    def _guard(self, at: dict) -> Callable[[dict], bool]:
        """A contact detector for a descent from *at*, where the arm has settled: a joint's
        load (shoulder_lift, elbow, wrist_flex) or lag behind its goal moving well away from
        its value there, for a few ticks in a row."""
        base_load = self.bus.sync_read("Present_Load", list(_GUARD_JOINTS), num_retry=_RETRY)
        present = self._joints()
        base_lag = {j: present[j] - at[j] for j in ARM_JOINTS}
        ticks = 0

        def touching(goal: dict) -> bool:
            nonlocal ticks
            load = self.bus.sync_read("Present_Load", list(_GUARD_JOINTS), num_retry=_RETRY)
            now = self._joints()
            hit = (any(abs(load[j] - base_load[j]) > _GUARD_LOAD for j in _GUARD_JOINTS)
                   or any(abs(now[j] - goal[j] - base_lag[j]) > _GUARD_LAG for j in ARM_JOINTS))
            ticks = ticks + 1 if hit else 0
            return ticks >= _GUARD_TICKS

        return touching

    def _back_off(self, path: list[dict], zs: list[float]) -> tuple[dict, float, int]:
        """After a contact on the way down: stop pushing, then go back up the path to 3 mm
        above where the jaws are. Returns (joints, jaw-tip height, index of the waypoint above)."""
        self.freeze(list(ARM_JOINTS))
        z = self.reach.tip_z(self._joints()) + _BACK_OFF_M
        i = max([k for k, zk in enumerate(zs) if zk >= z], default=0)
        if i + 1 < len(path) and zs[i] >= z > zs[i + 1]:
            t = (zs[i] - z) / (zs[i] - zs[i + 1])
            q = {j: path[i][j] + (path[i + 1][j] - path[i][j]) * t for j in ARM_JOINTS}
        else:
            q, z = path[i], zs[i]
        self.move_to(q, _TASK_SPEED)
        return q, z, i

    def _close_on_object(self, holding: dict) -> bool:
        """Compliant close: step the gripper toward its closed pose until it stalls on
        something (lagging and loaded for a few ticks), then hold it just past that point
        with a moderate squeeze. True if it closed on something; ``holding`` is set the
        moment it does (so a shutdown mid-grasp still puts it back)."""
        closed = self.pose("gripper_closed")["gripper"]
        goal, _ = self._gripper_now()
        contact, ticks = None, 0
        for _ in range(round(_CLOSE_MAX_S / _TICK_S)):
            self._watch()
            goal = max(goal - _CLOSE_STEP, closed)
            self.robot.send_action({"gripper.pos": goal})
            time.sleep(_TICK_S)
            pos, load = self._gripper_now()
            ticks = ticks + 1 if pos - goal >= _GRIP_LAG and load >= _GRIP_LOAD else 0
            if ticks >= _GUARD_TICKS:
                contact = pos
                break
            if goal <= closed and pos <= closed + _EMPTY_MARGIN:
                return False  # closed on nothing
        if contact is None:  # out of time short of closed: call it a contact where it is
            contact, _ = self._gripper_now()
            if contact <= closed + _EMPTY_MARGIN:
                return False
        self.holding, self._slack = holding, 0
        grip = max(contact - _SQUEEZE, closed)
        for tries in range(_HOLD_TRIES + 1):
            self.robot.send_action({"gripper.pos": grip})
            self._grip_goal = grip
            self._wait(_SETTLE_TASK_S)
            _, load = self._gripper_now()
            if tries == _HOLD_TRIES or _HOLD_LOAD[0] <= load <= _HOLD_LOAD[1]:
                break
            grip = grip + _CLOSE_STEP if load > _HOLD_LOAD[1] else max(grip - _CLOSE_STEP, closed)
        return True

    def _still_holding(self) -> bool:
        pos, load = self._gripper_now()
        return pos > self.pose("gripper_closed")["gripper"] + _HELD_POS and load > _HELD_LOAD

    def check_grip(self) -> Optional[str]:
        """While holding (idle, once a second): ease off a squeeze that got too hard, and
        notice a drop — returns the dropped object's label (and forgets it). Dropped: the
        jaws shut, or (twice in a row) closed to their hold goal with no load, as they stop
        there, not at the closed pose, once the object is out."""
        pos, load = self._gripper_now()
        slack = self._grip_goal is not None and pos <= self._grip_goal + _DROP_SLACK and load < _HELD_LOAD
        self._slack = self._slack + 1 if slack else 0
        if pos <= self.pose("gripper_closed")["gripper"] + _EMPTY_MARGIN or self._slack >= _DROP_CHECKS:
            label, self.holding, self._grip_goal = self.holding["label"], None, None
            return label
        if load > _EASE_LOAD and self._grip_goal is not None:
            self._grip_goal += _CLOSE_STEP
            self.robot.send_action({"gripper.pos": self._grip_goal})
        return None


class Controller:
    """Runs one arm's requests in order, one at a time, on a worker thread.

    Moves queue up (at most _QUEUE_MAX waiting). ``stop`` and ``shutdown`` jump the queue:
    they interrupt the running move and cancel the waiting ones. A motor fault does the
    same, freezes the arm, and is announced with a ``fault`` event — also when it trips
    while the arm is idle but holding a pose. Only the worker thread touches the bus.
    """

    def __init__(self, arm: Arm, reply: Callable[[dict], None], name: str = "arm") -> None:
        self.arm = arm
        self.reply = reply
        self.name = name
        self.cond = threading.Condition()
        self.waiting: deque[tuple[Any, str, dict]] = deque()
        self.control: Optional[tuple[Any, str]] = None  # a pending stop / shutdown
        self.running: Optional[Any] = None  # id of the move in progress
        self._reported: Optional[dict] = None  # fault already announced (no repeats)
        self._unreachable = False  # "not responding" already announced

    def handle(self, msg: dict) -> bool:
        """Accept one request (called from the stdin thread). Returns False once shutdown
        has been requested."""
        rid, action = msg.get("id"), str(msg.get("action", ""))
        if action == "status":
            with self.cond:
                busy, queued = self.running is not None, len(self.waiting)
            holding = self.arm.holding
            self.reply({"id": rid, "ok": True, "busy": busy, "queued": queued, "actions": self.arm.actions(),
                        "holding": holding["label"] if holding else None})
            return True
        if action in ("stop", "shutdown"):
            done = []
            with self.cond:
                cancelled = self._take_waiting()
                if self.control is None or action == "shutdown":
                    if self.control is not None:
                        done.append(self.control[0])  # a pending stop, superseded by shutdown
                    self.control = (rid, action)
                else:
                    done.append(rid)  # already stopping / shutting down
                self.arm.abort.set()  # interrupt the running move
                self.cond.notify()
            self._cancel(cancelled, f"cancelled: {action}")
            for done_id in done:
                self.reply({"id": done_id, "ok": True})
            return action != "shutdown"
        args = msg.get("args") or {}
        if not isinstance(args, dict):
            self.reply({"id": rid, "ok": False, "executed": False, "error": "args must be a JSON object"})
            return True
        with self.cond:
            full = len(self.waiting) >= _QUEUE_MAX
            if not full:
                self.waiting.append((rid, action, args))
                self.cond.notify()
        if full:
            self.reply({"id": rid, "ok": False, "executed": False,
                        "error": f"too many arm moves queued (max {_QUEUE_MAX})"})
        return True

    def work(self) -> None:
        while True:
            with self.cond:
                if self.control is None and not self.waiting:
                    self.cond.wait(_IDLE_POLL_S)
                if self.control is not None:
                    job, self.control = (*self.control, {}), None
                elif self.waiting:
                    job = self.waiting.popleft()
                    self.running = job[0]
                else:
                    job = None
                if job is not None:
                    # Under the lock: a stop that arrives from here on still interrupts it.
                    self.arm.abort.clear()
            if job is None:
                self._idle_check()
                continue
            rid, action, args = job
            try:
                result = None
                if action == "stop":
                    self.arm.stop_here()  # freeze in place
                elif action == "shutdown":
                    self.arm.shutdown()  # park (motors off); the port closes below, after any fault is handled
                else:
                    result = self.arm.run(action, args)
                self.reply({"id": rid, "ok": True, **(result or {})})
            except Stopped:
                self.reply({"id": rid, "ok": False, "error": "stopped", "cancelled": True})
            except Refused as exc:
                self.reply({"id": rid, "ok": False, "executed": False, "error": str(exc)})
            except Exception as exc:
                fault = exc if isinstance(exc, MotorFault) else _fault_from_error(exc, self.arm.bus)
                if fault is not None:
                    self._on_fault(fault, rid)
                elif action == "shutdown":
                    # Couldn't park (blocked, or no rest pose): stop pushing before letting go.
                    self.reply({"id": rid, "ok": False, "error": f"{exc}; {_freeze(self.arm)}"})
                else:
                    self.reply({"id": rid, "ok": False, "error": str(exc)})
                    if isinstance(exc, OSError):  # lost the arm (ConnectionError & co): this reply
                        self._unreachable = True  # told the app; the idle check needn't again
            finally:
                with self.cond:
                    self.running = None
            if action == "shutdown":
                self.arm.bus.disconnect(disable_torque=False)  # parked: motors already off
                return

    def _idle_check(self) -> None:
        """Nothing running: if the arm is holding a pose, make sure no servo has tripped
        (and, if it holds an object, that it hasn't dropped it). A fault is announced once,
        until a status read comes back clean; losing the arm (power or cable) likewise."""
        if not self.arm.torque_on:
            self._reported = None
            return
        dropped = None
        try:
            self.arm.check_faults()
            if self.arm.holding is not None:
                dropped = self.arm.check_grip()
        except MotorFault as fault:
            self._unreachable = False
            if fault.faults != self._reported:
                self._on_fault(fault)
            return
        except Exception as exc:
            if not self._unreachable:
                self._unreachable = True
                print(f"arm not responding: {exc}", file=sys.stderr)
                self.reply({"event": "unreachable", "arm": self.name, "detail": str(exc)})
            return
        self._reported = None
        self._unreachable = False
        if dropped is not None:
            self.reply({"event": "dropped", "arm": self.name, "label": dropped})

    def _on_fault(self, fault: MotorFault, rid: Any = None) -> None:
        """Cancel everything queued, stop pushing (freeze), answer the interrupted move
        (if any) and announce the fault — once: the idle check won't repeat it."""
        with self.cond:
            cancelled = self._take_waiting()
        # A fault caught from a write error names only the first motor: report every servo
        # that has tripped (read before freezing, which may clear the bits).
        fault = MotorFault(_merge_faults(fault.faults, self._status_faults(default=None)))
        detail = _freeze(self.arm, fault.faults)
        # What the idle check compares against, so it doesn't announce this again.
        self._reported = self._status_faults(default=fault.faults)
        if rid is not None:
            self.reply({"id": rid, "ok": False, "error": f"{fault}; {detail}", "faults": fault.faults})
        self._cancel(cancelled, f"cancelled: {fault}")
        self.reply({
            "event": "fault", "arm": self.name, "faults": fault.faults,
            "interrupted": rid is not None, "cancelled": len(cancelled), "detail": detail,
        })

    def _status_faults(self, default: Optional[dict]) -> Optional[dict]:
        """The servos' faults right now (None if clean); *default* if they can't be read."""
        try:
            self.arm.check_faults()
            return None
        except MotorFault as now:
            return now.faults
        except Exception:
            return default

    def _take_waiting(self) -> list[tuple[Any, str, dict]]:
        """Empty the queue (call with the lock held)."""
        jobs = list(self.waiting)
        self.waiting.clear()
        return jobs

    def _cancel(self, jobs: list[tuple[Any, str, dict]], why: str) -> None:
        for rid, *_ in jobs:
            self.reply({"id": rid, "ok": False, "error": why, "cancelled": True})


# --------------------------------------------------------------------------- #
# LeRobot hardware                                                              #
# --------------------------------------------------------------------------- #

def connect_follower(port: str, robot_id: str) -> Any:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    if not port:
        raise RuntimeError("ARM_PORT is not set (find it with lerobot-find-port)")
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=_MAX_STEP))
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=False)
        raise RuntimeError(
            f"follower '{robot_id}' isn't calibrated; run: lerobot-calibrate "
            f"--robot.type=so101_follower --robot.port={port} --robot.id={robot_id}"
        )
    _bring_up(robot)
    return robot


def _bring_up(robot: Any) -> None:
    """Apply LeRobot's motor setup and end with torque on, holding where the arm is."""
    bus = robot.bus
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position", num_retry=_RETRY))
    if any(bus.sync_read("Torque_Enable", num_retry=_RETRY).values()):
        # Still holding from an earlier run (setup already applied). LeRobot's configure()
        # would turn torque off while it writes settings, so the arm would sag and then snap
        # back. Keep it powered instead.
        bus.enable_torque()
    else:
        # Limp, so nothing can sag. configure() ends by turning torque on, and the target
        # set above (where the arm is) means that can't make it jump.
        robot.configure()


def _json_channel() -> Any:
    """The protocol stream: a copy of fd 1. fd 1 itself (and sys.stdout) then point at
    stderr, so nothing else reaches the app — not even C++ code (placo) printing to fd 1."""
    sys.stdout.flush()
    json_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(json_fd, "w", buffering=1)


def serve(cfg: dict, out: Any = None) -> int:
    if out is None:
        out = _json_channel()  # before anything can print to fd 1
    lock = threading.Lock()

    def reply(msg: dict) -> None:
        with lock:
            try:
                out.write(json.dumps(msg) + "\n")
                out.flush()
            except (OSError, ValueError):
                pass  # the app is gone; keep going so the shutdown still parks the arm

    motions = load_motions(cfg["motions"])
    reach = _load_reach(cfg, motions)  # first: it needs no arm, and ~2 s here can't stall a half-moved arm
    try:
        robot = connect_follower(cfg["port"], cfg["id"])
    except Exception as exc:
        reply({"ready": False, "error": str(exc)})
        return 1
    arm = Arm(robot, motions, reach)
    arm.powered = set(robot.bus.motors)  # bring-up leaves every motor holding; rest() below lets go
    try:
        if "rest" in arm.motions["poses"]:
            arm.rest()  # start from a known, relaxed pose
    except Exception as exc:
        detail = _freeze(arm)  # blocked or faulted on the way: stop pushing before letting go
        robot.bus.disconnect(disable_torque=False)
        reply({"ready": False, "error": f"could not move to rest: {exc}; {detail}"})
        return 1
    reply({"ready": True, "arm": cfg["id"], "actions": arm.actions()})

    controller = Controller(arm, reply, name=cfg["id"])
    worker = threading.Thread(target=controller.work, daemon=True)
    worker.start()
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        if not controller.handle(msg):
            break
    else:  # stdin closed (the app exited): park anyway
        controller.handle({"id": None, "action": "shutdown"})
    worker.join()
    return 0


def teach(cfg: dict, name: str) -> int:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=cfg["port"], id=cfg["id"]))
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        print(f"Follower '{cfg['id']}' isn't calibrated yet; run lerobot-calibrate first.")
        return 1
    try:
        input("Support the arm with your hand, then press Enter to turn its motors off... ")
        robot.bus.disable_torque()
        input(f"Move it into the '{name}' pose, then press Enter... ")
        pose = {m: round(v, 2) for m, v in robot.bus.sync_read("Present_Position").items()}
        motions = load_motions(cfg["motions"])
        motions["poses"][name] = pose
        save_motions(cfg["motions"], motions)
        print(f"Saved pose '{name}' to {cfg['motions']}.")
        if name != "rest":
            input("Lower the arm back to rest by hand, then press Enter... ")
    finally:
        robot.bus.disconnect(disable_torque=False)  # already off, unless you quit before that
    return 0


def record(cfg: dict, name: str, seconds: float) -> int:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    motions = load_motions(cfg["motions"])
    if "rest" not in motions["poses"]:
        print("Teach the rest pose first: scripts/arm.sh teach rest")
        return 1
    if not cfg["leader_port"]:
        print("LEADER_PORT is not set (find it with lerobot-find-port).")
        return 1
    follower = connect_follower(cfg["port"], cfg["id"])
    leader = SO101Leader(SO101LeaderConfig(port=cfg["leader_port"], id=cfg["leader_id"]))
    arm = Arm(follower, motions)
    parked = False
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            print(f"Leader '{cfg['leader_id']}' isn't calibrated yet; run lerobot-calibrate first.")
            return 1
        input(f"Pose the leader in the start pose of '{name}', then press Enter. "
              "The follower stays still until you press Enter, then moves to match... ")
        arm.move_to({k.removesuffix(".pos"): v for k, v in leader.get_action().items()})
        for n in (3, 2, 1):
            print(f"{n}...")
            time.sleep(1)
        print(f"Recording '{name}' for {seconds:g} s; perform it on the leader now.")
        frames, start = [], time.monotonic()
        while (t := time.monotonic() - start) < seconds:
            sent = follower.send_action(leader.get_action())
            frames.append([round(t, 3), {k.removesuffix(".pos"): round(v, 2) for k, v in sent.items()}])
            time.sleep(max(0.0, (len(frames) / _RECORD_HZ) - (time.monotonic() - start)))
        print("Done. Moving the follower back to rest.")
        arm.rest()
        parked = True
        motions["gestures"][name] = {"fps": _RECORD_HZ, "frames": frames}
        save_motions(cfg["motions"], motions)
        print(f"Saved gesture '{name}' ({len(frames)} frames) to {cfg['motions']}.")
    except KeyboardInterrupt:
        print("\nStopped; nothing was saved.")
        return 1
    finally:
        if not parked:  # Ctrl-C or an error: don't leave the follower powered where it stopped
            _recover_to_rest(arm)
        follower.bus.disconnect(disable_torque=False)
        if leader.is_connected:
            leader.disconnect()
    return 0


def _recover_to_rest(arm: Arm) -> None:
    """Take the follower back to rest (motors off) after an interrupted recording. If it
    can't get there, rest() keeps it holding, so it can't drop."""
    port = arm.bus.port_handler
    port.clearPort()  # a Ctrl-C can land mid-packet and leave the port marked busy
    port.is_using = False
    print("Moving the follower back to rest.")
    try:
        arm.rest()
    except Exception as exc:
        print(f"Couldn't get back to rest ({exc}). It's still holding: support it and unplug its power.")


# --------------------------------------------------------------------------- #
# Arm-task tools: fk, ik-check, try                                             #
# --------------------------------------------------------------------------- #

def _cm(p: Any) -> str:
    return "(" + ", ".join(f"{v * 100:5.1f}" for v in p) + ")"


def _describe(reach: Reach, q: dict) -> str:
    """One line: the joints, then the gripper origin and jaw tips in base_link (and in T)."""
    joints = " ".join(f"{q[j]:7.2f}" for j in ARM_JOINTS)
    pts = reach.points_b(q)
    line = f"joints {joints} | base_link gripper {_cm(pts['gripper'])} jaws {_cm(pts['jaw tips'])} cm"
    if reach.mount is not None:
        t = reach.points(q)
        line += f" | T gripper {_cm(t['gripper'])} jaws {_cm(t['jaw tips'])} cm"
    return line


def _cli_reach(cfg: dict, need_mount: bool) -> Optional[Reach]:
    settings = cfg.get("reach")
    if settings is None:
        print(f"Bad arm-task settings: {cfg.get('reach_error')}")
        return None
    if need_mount and settings["mount"] is None:
        print("Set ARM_BASE_X_CM, ARM_BASE_Y_CM and ARM_BASE_YAW_DEG in .env first (see .env.example).")
        return None
    try:
        return _make_reach(settings, load_motions(cfg["motions"])["poses"])
    except Exception as exc:
        print(f"Couldn't load the IK: {type(exc).__name__}: {exc}")
        return None


def fk(cfg: dict, registers: bool = False) -> int:
    """Read-only: print where the gripper is, 5 times a second, while you move the limp arm
    by hand. It never writes to a servo: the bus connects without LeRobot's setup."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    reach = _cli_reach(cfg, need_mount=False)
    if reach is None:
        return 1
    robot = SO101Follower(SO101FollowerConfig(port=cfg["port"], id=cfg["id"]))
    robot.bus.connect()  # pings and reads only; robot.connect() would run configure(), which writes
    try:
        if not robot.bus.is_calibrated:
            print(f"Follower '{cfg['id']}' isn't calibrated yet; run lerobot-calibrate first.")
            return 1
        if registers:
            for name in _GRIPPER_REGISTERS:
                value = robot.bus.sync_read(name, ["gripper"], normalize=False, num_retry=_RETRY)["gripper"]
                print(f"gripper {name}: {value}")
        print("Move the arm by hand; Ctrl-C to stop. Joints: " + ", ".join(ARM_JOINTS) + " (LeRobot degrees)."
              + ("" if reach.mount else " Set ARM_BASE_* in .env to see table coordinates too."))
        while True:
            print(_describe(reach, robot.bus.sync_read("Present_Position", num_retry=_RETRY)))
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        robot.bus.disconnect(disable_torque=False)
    return 0


def _why_code(reason: str) -> str:
    """One map character for a refusal."""
    for marker, code in (("would pass", "k"), ("too close to Reachy", "#"), ("no IK solution", "x"),
                         ("the arm's base", "."), ("too low", "z"), ("jumps", "j"), ("would reach", "l")):
        if marker in reason:
            return code
    return "?"


def ik_check(cfg: dict) -> int:
    """No arm needed: load the IK, FK the taught poses, round-trip the IK over a grid of table
    points around the arm, and map where a grasp can be planned."""
    t0 = time.monotonic()
    reach = _cli_reach(cfg, need_mount=False)
    if reach is None:
        return 1
    settings = cfg["reach"]
    print(f"IK loaded in {time.monotonic() - t0:.1f} s from {settings['urdf']}")
    m = reach.mount
    if m is None:
        print("Mount: not set (ARM_BASE_X_CM, ARM_BASE_Y_CM, ARM_BASE_YAW_DEG): base_link only.")
    else:
        print(f"Mount: pan axis {m['x'] * 100:g} cm ahead of and {m['y'] * 100:g} cm left of Reachy's foot "
              f"centre, facing {m['yaw']:g}°, base {m['z'] * 100:g} cm up")
    print(f"Keep-out {reach.keepout * 100:g} cm, wrist roll ±{reach.roll_limit:g}°, "
          f"joint offsets {', '.join(f'{v:g}' for v in settings['offsets'])}")
    poses = load_motions(cfg["motions"])["poses"]
    print("\nTaught poses (cm):")
    for name, pose in poses.items():
        print(f"  {name:15s} {_describe(reach, pose)}")
        lim = reach.model_limits
        bad = [f"{j} {pose[j]:.1f}" for j in ARM_JOINTS if not lim[j][0] <= pose[j] <= lim[j][1]]
        if bad:
            print(f"  {'':15s} past the CAD model's joint limits (the arm holds it, so moves from it are "
                  f"allowed): {', '.join(bad)}")
    if m is None:
        return 0
    print("\nKeep-out (closest checked arm point to Reachy's foot centre; the shells reach ~2.5 cm closer):")
    for name in ("rest", "ready"):
        if name in poses:
            part, d = min(((p, math.hypot(*xyz[:2])) for p, xyz in reach.points(poses[name]).items()), key=lambda v: v[1])
            print(f"  {name}: the {part} at {d * 100:.1f} cm ({'ok' if d >= reach.keepout else 'INSIDE the keep-out'})")
    for warning in reach.pose_warnings(poses):
        print("  " + warning)
    ready = {j: poses["ready"][j] for j in ARM_JOINTS} if "ready" in poses else None
    if ready is None:
        print("  'ready' isn't taught yet: plans below skip the ready → hover transit check.")
    current = poses.get("rest") or {j: 0.0 for j in ARM_JOINTS}
    _reach_map(reach, current, ready)
    _examples(reach, current, ready)
    return 0


def _reach_map(reach: Reach, current: dict, ready: Optional[dict], step: float = 0.03, span: float = 0.33) -> None:
    """Plan a grasp at every point of a table grid around the arm; print a top-down map."""
    m = reach.mount
    n = round(span / step)
    xs = [m["x"] + k * step for k in range(n, -n - 1, -1)]  # top of the map = Reachy's forward
    ys = [m["y"] + k * step for k in range(n, -n - 1, -1)]  # left of the map = Reachy's left
    times, worst_z, worst_xy, rows, counts = [], 0.0, 0.0, [], {}
    for x in xs:
        row = ""
        for y in ys:
            if math.hypot(x - m["x"], y - m["y"]) < step / 2:
                row += "A"
                continue
            if math.hypot(x, y) < step / 2:
                row += "R"
                continue
            t = time.monotonic()
            try:
                plan = reach.plan_grasp(x, y, None, current, ready)
                char = str(int(plan["pitch"]))[0]
                bottom = reach.points(plan["down"][-1])
                worst_z = max(worst_z, abs(bottom["jaw tips"][2] - _GRASP_Z))
                worst_xy = max(worst_xy, abs(math.dist(bottom["gripper"][:2], (x, y)) - _JAW_CENTRE_M))
            except Refused as exc:
                char = _why_code(str(exc))
            times.append(time.monotonic() - t)
            counts[char] = counts.get(char, 0) + 1
            row += char
        rows.append((x, row))
    print(f"\nGrasp map (top-down, {step * 100:g} cm grid; up = Reachy's forward, left = Reachy's left):")
    print("  9/7/6 plan found at a 90/75/60° approach; A arm base; R Reachy's foot centre; # inside the "
          "keep-out; . out of the arm's radius; k path through the keep-out; z too low; x no IK; "
          "j IK jump; l joint/roll limit")
    print("   x\\y  " + "".join(f"{y * 100:<6.0f}" for y in ys[::3]))
    for x, row in rows:
        print(f"  {x * 100:5.0f}  " + " ".join(row))
    found = sum(v for k, v in counts.items() if k in "976")
    print(f"  {found} of {len(times)} points plannable; {', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}")
    print(f"  planning time per grasp: mean {1000 * sum(times) / len(times):.0f} ms, max {1000 * max(times):.0f} ms")
    print(f"  IK round trips (FK of the lowest waypoint): jaw tips within {worst_z * 1000:.2f} mm of {_GRASP_Z * 1000:g} mm; "
          f"gripper origin within {worst_xy * 1000:.2f} mm of {_JAW_CENTRE_M * 100:g} cm from the grasp point")


def _examples(reach: Reach, current: dict, ready: Optional[dict]) -> None:
    """Plan each arm task at a few points relative to the arm."""
    m = reach.mount
    yaw = math.radians(m["yaw"])

    def ahead(d: float, left: float = 0.0) -> tuple[float, float]:
        return (m["x"] + d * math.cos(yaw) - left * math.sin(yaw), m["y"] + d * math.sin(yaw) + left * math.cos(yaw))

    points = [("12 cm ahead of the arm", ahead(0.12)), ("18 cm ahead", ahead(0.18)), ("24 cm ahead", ahead(0.24)),
              ("29 cm ahead", ahead(0.29)), ("35 cm ahead", ahead(0.35)), ("18 cm ahead, 8 cm to its left", ahead(0.18, 0.08)),
              ("18 cm ahead, 8 cm to its right", ahead(0.18, -0.08)), ("halfway to Reachy", (m["x"] / 2, m["y"] / 2))]
    print("\nExamples (T frame, cm):")

    def run(what: str, fn: Callable[[], dict]) -> Optional[dict]:
        t = time.monotonic()
        try:
            plan = fn()
            note = f"ok, {plan['pitch']:.0f}° approach"
        except Refused as exc:
            plan, note = None, f"refused: {exc}"
        print(f"    {what:22s} {note} ({1000 * (time.monotonic() - t):.0f} ms)")
        return plan

    held = None
    for name, (x, y) in points:
        print(f"  {name} ({x * 100:.1f}, {y * 100:.1f}):")
        grasp = run("grasp", lambda: reach.plan_grasp(x, y, None, current, ready))
        run("grasp, jaws at 90° in T", lambda: reach.plan_grasp(x, y, 90.0, current, ready))
        run("hover", lambda: reach.plan_hover(x, y, current, ready))
        if held is not None:
            run("place on the table", lambda: reach.plan_place(x, y, 0.02, held, ready))
            run("place 10 cm up", lambda: reach.plan_place(x, y, 0.10, held, ready))
        if held is None and grasp is not None:
            held = grasp["hover"]  # later places start from this grasp's hover


def _recover_task(arm: Arm) -> None:
    """After `try` (done, refused or Ctrl-C): let go of anything in the gripper, then park
    (via ready if the arm is out). If it can't get back, it keeps holding, so it can't drop."""
    port = arm.bus.port_handler
    port.clearPort()  # a Ctrl-C can land mid-packet and leave the port marked busy
    port.is_using = False
    arm.abort.clear()
    if not arm.powered:
        return  # already parked, motors off
    try:
        print("Moving the arm back to rest.")
        arm.park()  # opens the gripper first if it may be closed on something
    except Exception as exc:
        print(f"Couldn't get back to rest ({exc}). It's still holding: support it and unplug its power.")


def _print_plan(reach: Reach, what: str, plan: dict) -> None:
    print(f"Plan: {what} at a {plan['pitch']:.0f}° approach.")
    for key in ("up", "hover", "down"):
        if key not in plan:
            continue
        qs = plan[key] if isinstance(plan[key], list) else [plan[key]]
        print(f"  {key}: {len(qs)} waypoint(s), jaw tips at {_cm(reach.points(qs[0])['jaw tips'])}"
              + (f" → {_cm(reach.points(qs[-1])['jaw tips'])}" if len(qs) > 1 else "") + " cm (T)")
        print("    joints at the end: " + ", ".join(f"{j} {qs[-1][j]:.1f}" for j in ARM_JOINTS))


def try_task(cfg: dict, what: str, x_cm: Optional[float], y_cm: Optional[float], yaw: Optional[float],
             height_cm: Optional[float], above_cm: float = 4.6) -> int:
    """Try one arm task on the arm: prints the plan and waits for Enter before anything moves.
    ``close`` closes compliantly where the arm is, on an object you put between the jaws;
    ``place`` hands it the object at ready first. Ctrl-C parks it safely."""
    if what != "close" and (x_cm is None or y_cm is None):
        print("Give the spot with --x CM --y CM (table frame: forward of and left of Reachy's foot centre).")
        return 1
    reach = None if what == "close" else _cli_reach(cfg, need_mount=True)
    if reach is None and what != "close":
        return 1
    motions = load_motions(cfg["motions"])
    missing = [p for p in ("rest", "gripper_open", "gripper_closed", "ready") if p not in motions["poses"]
               and not (p == "ready" and what == "close")]
    if missing:
        print("Teach these poses first: " + ", ".join(f"scripts/arm.sh teach {p}" for p in missing))
        return 1
    # place: the release height (default 2 cm); hover: only if given, to hover where that place
    # would start down — else the jaw tips *above_cm* over the table.
    height = (2.0 if height_cm is None and what == "place" else height_cm)
    args = {"x": (x_cm or 0.0) / 100, "y": (y_cm or 0.0) / 100, "yaw_deg": yaw,
            "height": None if height is None else height / 100, "label": "object"}
    if what == "hover":
        args["tips"] = above_cm / 100
    robot = connect_follower(cfg["port"], cfg["id"])
    arm = Arm(robot, motions, reach)
    arm.powered = set(robot.bus.motors)  # bring-up leaves every motor holding where it is
    holding = {"x": None, "y": None, "yaw": None, "pitch": None, "label": "object"}
    try:
        if what == "close":
            print("Plan: open the gripper where the arm is; you put an object between the jaws; a compliant "
                  "close; it holds 10 s (printing the load), then opens and rests.")
        else:
            ready = arm._task()[1]
            current = arm._joints()
            if what == "hover":
                plan = reach.plan_hover(args["x"], args["y"], current, ready, args["height"], args["tips"])
            elif what == "grasp":
                plan = reach.plan_grasp(args["x"], args["y"], yaw, current, ready)
            else:  # you hand it the object at ready, then it places it
                plan = reach.plan_place(args["x"], args["y"], args["height"], ready, ready)
            _print_plan(reach, what, plan)
            if what == "hover":
                print(f"  The jaws open first. Check the point midway between them over the spot; the fixed "
                      f"jaw's tip is aimed {_JAW_CENTRE_M * 100:g} cm from it, toward the moving jaw's side.")
        input("Press Enter to start (Ctrl-C to cancel)... ")
        if what == "hover":
            arm.hover(args)
        elif what == "grasp":
            print("Holding it." if arm.grasp(args)["holding"] else "Nothing gripped; back at rest.")
            if arm.holding is not None:
                input("Press Enter to put it back... ")
                arm._put_back()
        elif what == "place":
            arm.hold()
            arm._out = True
            arm.move_to(arm._ready(), _TASK_SPEED)
            arm.gripper("gripper_open")
            input("Put the object between the jaws, then press Enter... ")
            arm._gripping = True
            if arm._close_on_object(holding):
                arm.place(args)
            else:
                print("Nothing between the jaws.")
        else:
            arm.gripper("gripper_open")
            input("Put an object between the open jaws, then press Enter... ")
            arm._gripping = True
            if arm._close_on_object(holding):
                for _ in range(10):
                    pos, load = arm._gripper_now()
                    print(f"gripper at {pos:.1f} % (goal {arm._grip_goal:.1f}), load {load}")
                    arm._wait(1.0)
            else:
                print("Nothing between the jaws.")
    except Refused as exc:
        print(f"Refused: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _recover_task(arm)
        robot.bus.disconnect(disable_torque=False)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    p_teach = sub.add_parser("teach", help="save a pose: rest, ready, gripper_open, gripper_closed")
    p_teach.add_argument("name")
    p_rec = sub.add_parser("record", help="record a gesture with the leader, e.g. wave")
    p_rec.add_argument("name")
    p_rec.add_argument("--seconds", type=float, default=5.0)
    p_fk = sub.add_parser("fk", help="read-only: print where the gripper is while you move the arm by hand")
    p_fk.add_argument("--registers", action="store_true", help="also read the gripper's protection settings once")
    sub.add_parser("ik-check", help="no arm needed: check the IK, the taught poses and the reach")
    p_try = sub.add_parser("try", help="try an arm task on the arm, step by step")
    p_try.add_argument("what", choices=["hover", "close", "grasp", "place"])
    p_try.add_argument("--x", type=float, help="cm forward of Reachy's foot centre (table frame)")
    p_try.add_argument("--y", type=float, help="cm to Reachy's left of its foot centre (table frame)")
    p_try.add_argument("--yaw", type=float, help="grasp: the jaws' closing direction, degrees (table frame)")
    p_try.add_argument("--height", type=float,
                       help="place: release height of the jaw tips, cm (default 2; hover: as high as that place starts down)")
    p_try.add_argument("--above", type=float, default=4.6,
                       help="hover: the jaw tips this many cm above the table (default 4.6; at least 3)")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.cmd == "serve":
        return serve(cfg)
    if args.cmd == "ik-check":
        return ik_check(cfg)
    if not cfg["port"]:
        print("ARM_PORT is not set (find it with lerobot-find-port).")
        return 1
    if args.cmd == "fk":
        return fk(cfg, args.registers)
    if args.cmd == "try":
        return try_task(cfg, args.what, args.x, args.y, args.yaw, args.height, args.above)
    if args.cmd == "teach":
        return teach(cfg, args.name)
    return record(cfg, args.name, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
