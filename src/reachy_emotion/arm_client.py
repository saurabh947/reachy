"""Client for the SO-101 arm server (``arm_server.py``, which runs in ``.venv-lerobot``).

The Reachy app starts the server as a child process and talks to it in JSON lines over
its stdin/stdout. The child gets its own session, so a Ctrl-C in the terminal reaches only
the app, which then parks the arm through ``close()``. If the app dies, the server sees
stdin close and parks the arm itself.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("arm_server.py")
ARM_PYTHON = REPO / ".venv-lerobot" / "bin" / "python"

# What Gemini can ask the arm to do (the server reports which ones are taught).
ARM_ACTIONS = ("wave", "gripper_open", "gripper_close", "rest", "stop")


class ArmClient:
    def __init__(self, cmd: list[str], ready_timeout: float = 30.0) -> None:
        self._cmd = cmd
        self._ready_timeout = ready_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[threading.Event, dict]] = {}
        self._lock = threading.Lock()
        self._ready: "queue.Queue[dict]" = queue.Queue()
        self.actions: list[str] = []

    def start(self) -> None:
        """Start the server and wait until it has connected to the arm."""
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._read, daemon=True).start()
        try:
            msg = self._ready.get(timeout=self._ready_timeout)
        except queue.Empty:
            self._kill()
            raise RuntimeError("arm server did not start in time")
        if not msg.get("ready"):
            self._kill()
            raise RuntimeError(msg.get("error") or "arm server failed to start")
        self.actions = list(msg.get("actions", []))

    def call(self, action: str, timeout: float = 60.0) -> dict:
        """Run one action and wait for its result (thread-safe; ``stop`` isn't queued
        behind a running action)."""
        if self._proc is None or self._proc.poll() is not None:
            return {"ok": False, "error": "arm server is not running"}
        rid = next(self._ids)
        done, result = threading.Event(), {}
        with self._lock:
            self._pending[rid] = (done, result)
            try:
                self._proc.stdin.write(json.dumps({"id": rid, "action": action}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._pending.pop(rid, None)
                return {"ok": False, "error": "arm server is not running"}
        if not done.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            return {"ok": False, "error": f"no answer from the arm within {timeout:g} s"}
        return result

    def close(self, timeout: float = 20.0) -> None:
        """Park the arm (rest, motors off) and stop the server."""
        if self._proc is None or self._proc.poll() is not None:
            return
        self.call("shutdown", timeout=timeout)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill()

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ready" in msg:
                self._ready.put(msg)
                continue
            with self._lock:
                waiter = self._pending.pop(msg.get("id"), None)
            if waiter:
                waiter[1].update({k: v for k, v in msg.items() if k != "id"})
                waiter[0].set()
        # The server exited: fail whatever is still waiting.
        self._ready.put({"ready": False, "error": "arm server exited"})
        with self._lock:
            waiters, self._pending = list(self._pending.values()), {}
        for done, result in waiters:
            result.update({"ok": False, "error": "arm server exited"})
            done.set()

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


def arm_from_env() -> Optional[ArmClient]:
    """Start the arm server if ARM_PORT is set; None when no arm is configured."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    if not os.environ.get("ARM_PORT", "").strip():
        return None
    if not ARM_PYTHON.exists():
        raise RuntimeError(f"{ARM_PYTHON} not found (create .venv-lerobot with LeRobot)")
    client = ArmClient([str(ARM_PYTHON), str(SERVER), "serve"])
    client.start()
    logger.info("Arm connected (%s)", ", ".join(client.actions))
    return client
