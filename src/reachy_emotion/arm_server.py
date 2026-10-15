"""Arm server: drives the SO-101 follower arm for the Reachy app, and teaches it moves.

Runs in ``.venv-lerobot``. LeRobot can't share the Reachy app's environment:
reachy-mini pins ``huggingface-hub==1.3.0`` and LeRobot needs ``>=1.6``. So this file is
standalone: stdlib only at import time, LeRobot imported lazily, no ``reachy_emotion`` imports.

    serve                       started by the Reachy app; JSON lines on stdin/stdout
    teach NAME                  save the follower's current pose (move it by hand, motors relaxed)
    record NAME [--seconds S]   record a gesture by moving the leader (the follower mirrors it)

Poses the app uses: ``rest``, ``gripper_open``, ``gripper_closed``. Gestures: e.g. ``wave``.
Config (environment, else ``.env``): ARM_PORT, ARM_ID (default ``follower``),
LEADER_PORT, LEADER_ID (default ``leader``), ARM_MOTIONS (default ``<repo>/arm_motions.json``).

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
"""

from __future__ import annotations

import argparse
import json
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
_FAULT_POLL_S = 0.1   # while moving: read the servos' status this often
_IDLE_POLL_S = 0.5    # while holding a pose (torque on, nothing running)

# Feetech status bits (the Status register, and the error byte of every reply).
_FAULT_BITS = {0x20: "overload", 0x08: "overcurrent", 0x04: "overheat"}
# The same faults as LeRobot reports them when a write's reply carries the bit.
_FAULT_ERRORS = {"Overload error": "overload", "OverEle error": "overcurrent", "Overheat error": "overheat"}


class Stopped(Exception):
    """The current action was interrupted by ``stop``."""


class MotorFault(Exception):
    """A servo's protection tripped; *faults* maps motor name -> fault kinds."""

    def __init__(self, faults: dict[str, list[str]]) -> None:
        self.faults = faults
        super().__init__("motor fault: " + ", ".join(f"{m} {'/'.join(k)}" for m, k in faults.items()))


def _freeze(arm: "Arm") -> str:
    """Stop the arm pushing — only the motors that are powered: re-targeting a let-go
    motor would power it up again. Say how that went."""
    if not arm.powered:
        return "motors already off"
    try:
        arm.freeze(sorted(arm.powered))
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


def load_config() -> dict[str, Any]:
    env = {**_read_dotenv(REPO / ".env"), **os.environ}
    return {
        "port": env.get("ARM_PORT", "").strip(),
        "id": env.get("ARM_ID", "").strip() or "follower",
        "leader_port": env.get("LEADER_PORT", "").strip(),
        "leader_id": env.get("LEADER_ID", "").strip() or "leader",
        "motions": Path(env.get("ARM_MOTIONS", "").strip() or REPO / "arm_motions.json"),
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
# The arm                                                                       #
# --------------------------------------------------------------------------- #

class Arm:
    """Moves a connected LeRobot SO-101 follower through taught poses and gestures."""

    def __init__(self, robot: Any, motions: dict) -> None:
        self.robot = robot
        self.bus = robot.bus
        self.motions = motions
        self.abort = threading.Event()
        self.powered: set[str] = set()  # motors that may be driven (so a fault is possible)
        self._next_check = 0.0

    @property
    def torque_on(self) -> bool:
        return bool(self.powered)

    def actions(self) -> list[str]:
        poses = self.motions["poses"]
        names = [n for n in ("rest",) if n in poses]
        names += [a for a, p in (("gripper_open", "gripper_open"), ("gripper_close", "gripper_closed")) if p in poses]
        return names + sorted(self.motions["gestures"]) + ["stop"]

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

    def hold(self, motors: Optional[str] = None) -> None:
        """Turn torque on where the joints are now. The target is set to the present
        position first, so a joint moved by hand while relaxed can't jump back."""
        self.freeze(motors)
        self.bus.enable_torque(motors)

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

    def move_to(self, target: dict[str, float]) -> None:
        """Move smoothly from the present position to *target* (only the joints given)."""
        present = self.bus.sync_read("Present_Position", list(target), num_retry=_RETRY)
        dist = max(abs(target[m] - present[m]) for m in target)
        steps = max(1, round(max(_MIN_MOVE_S, dist / _SPEED) / _TICK_S))
        for i in range(1, steps + 1):
            self._watch()
            a = i / steps
            a = a * a * (3 - 2 * a)  # ease in and out
            self.robot.send_action({f"{m}.pos": present[m] + (target[m] - present[m]) * a for m in target})
            time.sleep(_TICK_S)

    def rest(self) -> None:
        target = self.pose("rest")
        self.hold()
        self.move_to(target)
        self.settle(target)
        self.release()

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

    def run(self, action: str) -> None:
        if action == "rest":
            self.rest()
        elif action == "gripper_open":
            self.gripper("gripper_open")
        elif action == "gripper_close":
            self.gripper("gripper_closed")
        else:
            self.gesture(action)


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
        self.waiting: deque[tuple[Any, str]] = deque()
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
            self.reply({"id": rid, "ok": True, "busy": busy, "queued": queued, "actions": self.arm.actions()})
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
        with self.cond:
            full = len(self.waiting) >= _QUEUE_MAX
            if not full:
                self.waiting.append((rid, action))
                self.cond.notify()
        if full:
            self.reply({"id": rid, "ok": False, "error": f"too many arm moves queued (max {_QUEUE_MAX})"})
        return True

    def work(self) -> None:
        while True:
            with self.cond:
                if self.control is None and not self.waiting:
                    self.cond.wait(_IDLE_POLL_S)
                if self.control is not None:
                    job, self.control = self.control, None
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
            rid, action = job
            try:
                if action == "stop":
                    self.arm.hold()  # freeze in place
                elif action == "shutdown":
                    self.arm.rest()  # park (motors off); the port closes below, after any fault is handled
                else:
                    self.arm.run(action)
                self.reply({"id": rid, "ok": True})
            except Stopped:
                self.reply({"id": rid, "ok": False, "error": "stopped", "cancelled": True})
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
        """Nothing running: if the arm is holding a pose, make sure no servo has tripped.
        A fault is announced once, until a status read comes back clean; losing the arm
        (power or cable) likewise."""
        if not self.arm.torque_on:
            self._reported = None
            return
        try:
            self.arm.check_faults()
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

    def _on_fault(self, fault: MotorFault, rid: Any = None) -> None:
        """Cancel everything queued, stop pushing (freeze), answer the interrupted move
        (if any) and announce the fault — once: the idle check won't repeat it."""
        with self.cond:
            cancelled = self._take_waiting()
        # A fault caught from a write error names only the first motor: report every servo
        # that has tripped (read before freezing, which may clear the bits).
        fault = MotorFault(_merge_faults(fault.faults, self._status_faults(default=None)))
        detail = _freeze(self.arm)
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

    def _take_waiting(self) -> list[tuple[Any, str]]:
        """Empty the queue (call with the lock held)."""
        jobs = list(self.waiting)
        self.waiting.clear()
        return jobs

    def _cancel(self, jobs: list[tuple[Any, str]], why: str) -> None:
        for rid, _ in jobs:
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


def serve(cfg: dict) -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # library prints must not corrupt the JSON protocol
    lock = threading.Lock()

    def reply(msg: dict) -> None:
        with lock:
            try:
                out.write(json.dumps(msg) + "\n")
                out.flush()
            except (OSError, ValueError):
                pass  # the app is gone; keep going so the shutdown still parks the arm

    try:
        robot = connect_follower(cfg["port"], cfg["id"])
    except Exception as exc:
        reply({"ready": False, "error": str(exc)})
        return 1
    arm = Arm(robot, load_motions(cfg["motions"]))
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    p_teach = sub.add_parser("teach", help="save a pose: rest, gripper_open, gripper_closed")
    p_teach.add_argument("name")
    p_rec = sub.add_parser("record", help="record a gesture with the leader, e.g. wave")
    p_rec.add_argument("name")
    p_rec.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.cmd == "serve":
        return serve(cfg)
    if not cfg["port"]:
        print("ARM_PORT is not set (find it with lerobot-find-port).")
        return 1
    if args.cmd == "teach":
        return teach(cfg, args.name)
    return record(cfg, args.name, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
