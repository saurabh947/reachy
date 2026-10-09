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

Protocol (``serve``): the server first prints ``{"ready": true, "actions": [...]}`` (or
``{"ready": false, "error": ...}``). Then each request ``{"id": 1, "action": "wave"}`` gets
``{"id": 1, "ok": true}`` or ``{"id": 1, "ok": false, "error": ...}`` when it finishes.
``stop`` interrupts the running action and freezes the arm in place; ``shutdown`` (or
closing stdin) parks the arm at rest and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
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


class Stopped(Exception):
    """The current action was interrupted by ``stop``."""


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

    def actions(self) -> list[str]:
        poses = self.motions["poses"]
        names = [n for n in ("rest",) if n in poses]
        names += [a for a, p in (("gripper_open", "gripper_open"), ("gripper_close", "gripper_closed")) if p in poses]
        return names + sorted(self.motions["gestures"]) + ["stop"]

    def pose(self, name: str) -> dict[str, float]:
        if name not in self.motions["poses"]:
            raise RuntimeError(f"pose '{name}' isn't taught yet; run: scripts/arm.sh teach {name}")
        return self.motions["poses"][name]

    def hold(self, motors: Optional[str] = None) -> None:
        """Turn torque on where the joints are now. The target is set to the present
        position first, so a joint moved by hand while relaxed can't jump back."""
        self.bus.sync_write("Goal_Position", self.bus.sync_read("Present_Position", motors, num_retry=_RETRY))
        self.bus.enable_torque(motors)

    def move_to(self, target: dict[str, float]) -> None:
        """Move smoothly from the present position to *target* (only the joints given)."""
        present = self.bus.sync_read("Present_Position", list(target), num_retry=_RETRY)
        dist = max(abs(target[m] - present[m]) for m in target)
        steps = max(1, round(max(_MIN_MOVE_S, dist / _SPEED) / _TICK_S))
        for i in range(1, steps + 1):
            if self.abort.is_set():
                raise Stopped()
            a = i / steps
            a = a * a * (3 - 2 * a)  # ease in and out
            self.robot.send_action({f"{m}.pos": present[m] + (target[m] - present[m]) * a for m in target})
            time.sleep(_TICK_S)

    def rest(self) -> None:
        target = self.pose("rest")
        self.hold()
        self.move_to(target)
        self.settle(target)
        self.bus.disable_torque()

    def settle(self, target: dict[str, float]) -> None:
        """Wait until every joint is within _REST_TOL of *target*. Otherwise raise and keep
        the motors on, so a lagging or blocked arm is never let go short of rest."""
        deadline = time.monotonic() + _SETTLE_S
        while True:
            present = self.bus.sync_read("Present_Position", list(target), num_retry=_RETRY)
            if all(abs(present[m] - target[m]) <= _REST_TOL for m in target):
                return
            if time.monotonic() > deadline:
                raise RuntimeError("the arm didn't reach rest (blocked?); holding it in place")
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
            if self.abort.is_set():
                raise Stopped()
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

    def park_and_close(self) -> None:
        """Rest (motors off) if a rest pose is taught, then release the port. Without a
        rest pose the motors stay on, holding the arm where it is, so it can't drop."""
        try:
            self.rest()
        finally:
            self.bus.disconnect(disable_torque=False)


class Controller:
    """Runs one arm action at a time on a worker thread; ``stop`` interrupts it."""

    def __init__(self, arm: Arm, reply: Callable[[dict], None]) -> None:
        self.arm = arm
        self.reply = reply
        self.jobs: "queue.Queue[tuple[Any, str]]" = queue.Queue()
        self.busy = False
        self.lock = threading.Lock()

    def handle(self, msg: dict) -> bool:
        """Accept one request. Returns False once shutdown has been requested."""
        rid, action = msg.get("id"), str(msg.get("action", ""))
        if action == "status":
            self.reply({"id": rid, "ok": True, "busy": self.busy, "actions": self.arm.actions()})
        elif action in ("stop", "shutdown"):
            self.arm.abort.set()  # interrupt whatever is running
            self.jobs.put((rid, action))
            return action != "shutdown"
        else:
            with self.lock:
                if self.busy:
                    self.reply({"id": rid, "ok": False, "error": "busy with another move; say stop first"})
                    return True
                self.busy = True
                self.arm.abort.clear()  # cleared when accepted, so a stop sent right after still counts
            self.jobs.put((rid, action))
        return True

    def work(self) -> None:
        while True:
            rid, action = self.jobs.get()
            try:
                if action == "stop":
                    self.arm.hold()  # freeze in place
                elif action == "shutdown":
                    self.arm.abort.clear()
                    self.arm.park_and_close()
                else:
                    self.arm.run(action)
                self.reply({"id": rid, "ok": True})
            except Stopped:
                self.reply({"id": rid, "ok": False, "error": "stopped"})
            except Exception as exc:
                self.reply({"id": rid, "ok": False, "error": str(exc)})
            finally:
                if action not in ("stop", "shutdown"):
                    with self.lock:
                        self.busy = False
            if action == "shutdown":
                return


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
    try:
        if "rest" in arm.motions["poses"]:
            arm.rest()  # start from a known, relaxed pose
    except Exception as exc:
        robot.bus.disconnect(disable_torque=False)
        reply({"ready": False, "error": f"could not move to rest: {exc}"})
        return 1
    reply({"ready": True, "actions": arm.actions()})

    controller = Controller(arm, reply)
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
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            print(f"Leader '{cfg['leader_id']}' isn't calibrated yet; run lerobot-calibrate first.")
            return 1
        input(f"Put the leader in the start pose of '{name}' and press Enter (the follower moves to match)... ")
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
        motions["gestures"][name] = {"fps": _RECORD_HZ, "frames": frames}
        save_motions(cfg["motions"], motions)
        print(f"Saved gesture '{name}' ({len(frames)} frames) to {cfg['motions']}.")
    finally:
        if leader.is_connected:
            leader.disconnect()
        follower.bus.disconnect(disable_torque=False)
    return 0


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
