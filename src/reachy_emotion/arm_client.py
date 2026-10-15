"""Client for the SO-101 arm server (``arm_server.py``, which runs in ``.venv-lerobot``).

The Reachy app starts one server per follower arm as a child process and talks to it in
JSON lines over its stdin/stdout. Each server queues its arm's moves and runs them in
order. The child gets its own session, so a Ctrl-C in the terminal reaches only the app,
which then parks the arm through ``close()``. If the app dies, the server sees stdin
close and parks the arm itself.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import threading
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("arm_server.py")
ARM_PYTHON = REPO / ".venv-lerobot" / "bin" / "python"

# What Gemini can ask the arm to do (the server reports which ones are taught).
ARM_ACTIONS = ("wave", "gripper_open", "gripper_close", "rest", "stop")
_HELD_EVENTS = 20  # events kept while nobody listens yet


def _resolve(future: "Future[dict]", result: dict) -> None:
    """Hand *result* to a waiting call — never raising, so the reader thread can't die."""
    if not future.done():
        try:
            future.set_result(result)
        except InvalidStateError:
            pass


class ArmClient:
    """One follower arm's server. ``on_event`` (if set) is called with each event the
    server announces, e.g. a motor fault — on the client's reader thread. Events that
    arrive before it's set are held and handed over when it is."""

    def __init__(
        self, cmd: list[str], ready_timeout: float = 30.0, env: Optional[dict[str, str]] = None
    ) -> None:
        self._cmd = cmd
        self._env = env
        self._ready_timeout = ready_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._ids = itertools.count(1)
        self._pending: dict[int, "Future[dict]"] = {}
        self._lock = threading.Lock()
        self._ready: "queue.Queue[dict]" = queue.Queue()
        self.name = ""
        self.actions: list[str] = []
        self._on_event: Optional[Callable[[dict], None]] = None
        self._held_events: list[dict] = []  # announced before anyone listened
        self._event_lock = threading.Lock()

    @property
    def on_event(self) -> Optional[Callable[[dict], None]]:
        return self._on_event

    @on_event.setter
    def on_event(self, callback: Optional[Callable[[dict], None]]) -> None:
        with self._event_lock:
            self._on_event = callback
            held, self._held_events = (self._held_events, []) if callback else ([], self._held_events)
            for event in held:
                self._dispatch(callback, event)

    def _dispatch(self, callback: Callable[[dict], None], event: dict) -> None:
        try:
            callback(event)
        except Exception:
            logger.exception("arm event handler failed")

    def start(self) -> None:
        """Start the server and wait until it has connected to the arm."""
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, start_new_session=True, env=self._env,
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
        self.name = str(msg.get("arm") or "arm")
        self.actions = list(msg.get("actions", []))

    def submit(self, action: str) -> "Future[dict]":
        """Send one request without waiting; the future resolves with the server's reply.
        Requests reach the server in the order submit() is called (thread-safe). The
        server queues moves and runs them in order; ``stop`` jumps its queue."""
        future: "Future[dict]" = Future()
        # Running from the start: a waiter giving up (asyncio cancels the future it
        # awaits) can't cancel it, so the server's reply still lands cleanly.
        future.set_running_or_notify_cancel()
        if self._proc is None or self._proc.poll() is not None:
            future.set_result({"ok": False, "error": "arm server is not running"})
            return future
        with self._lock:
            rid = next(self._ids)
            self._pending[rid] = future
            try:
                self._proc.stdin.write(json.dumps({"id": rid, "action": action}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._pending.pop(rid, None)
                future.set_result({"ok": False, "error": "arm server is not running"})
        return future

    def call(self, action: str, timeout: Optional[float] = None) -> dict:
        """Run one action and wait for its result. No timeout by default: a queued move
        still runs later, so giving up early would misreport it; if the server dies,
        every waiting call fails with "arm server exited"."""
        try:
            return self.submit(action).result(timeout)
        except FutureTimeout:
            return {"ok": False, "error": f"no answer from the arm within {timeout:g} s"}

    def close(self, timeout: float = 20.0) -> None:
        """Park the arm (rest, motors off) and stop the server (killed if it hangs)."""
        if self._proc is None or self._proc.poll() is not None:
            return
        result = self.call("shutdown", timeout=timeout)
        if not result.get("ok"):
            logger.warning("Arm %s didn't park cleanly: %s", self.name, result.get("error"))
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
            if "event" in msg:
                with self._event_lock:
                    if self._on_event is None:
                        self._held_events = [*self._held_events[-(_HELD_EVENTS - 1):], msg]
                    else:
                        self._dispatch(self._on_event, msg)
                continue
            with self._lock:
                waiter = self._pending.pop(msg.get("id"), None)
            if waiter is not None:
                _resolve(waiter, {k: v for k, v in msg.items() if k != "id"})
        # The server exited: fail whatever is still waiting.
        self._ready.put({"ready": False, "error": "arm server exited"})
        with self._lock:
            waiters, self._pending = list(self._pending.values()), {}
        for waiter in waiters:
            _resolve(waiter, {"ok": False, "error": "arm server exited"})

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


def _arm_configs() -> list[dict[str, str]]:
    """Environment overrides for each follower arm's server (the server reads ARM_PORT,
    ARM_ID and ARM_MOTIONS itself). One arm today, from .env; a second arm is one more
    entry, e.g. {"ARM_PORT": ..., "ARM_ID": "follower_left", "ARM_MOTIONS": ...}."""
    return [{}] if os.environ.get("ARM_PORT", "").strip() else []


def arms_from_env() -> dict[str, ArmClient]:
    """Start a server for each configured arm; {arm name: client}, empty when there's none.
    An arm that fails to start is logged and left out, so the others still work."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    configs = _arm_configs()
    if configs and not ARM_PYTHON.exists():
        raise RuntimeError(f"{ARM_PYTHON} not found (create .venv-lerobot with LeRobot)")
    arms: dict[str, ArmClient] = {}
    for overrides in configs:
        client = ArmClient([str(ARM_PYTHON), str(SERVER), "serve"], env={**os.environ, **overrides})
        try:
            client.start()
        except Exception as exc:
            logger.warning("Arm %s not available: %s", overrides.get("ARM_ID", "(ARM_PORT)"), exc)
            continue
        if client.name in arms:
            logger.warning("Two arms are both named %r; keeping the first", client.name)
            client.close()
            continue
        arms[client.name] = client
        logger.info("Arm '%s' connected (%s)", client.name, ", ".join(client.actions))
    return arms
