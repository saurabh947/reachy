"""Reachy's long-term memory: a small text file, loaded into Gemini Live's prompt when a
conversation starts and kept up to date as it goes.

Gemini saves facts with the ``remember`` tool (on its own, or when asked); look_at_scene
saves where it sees things by itself. One fact per line, filed by section and item:
saving the same item again replaces it, so the file stays short and current. It's plain
Markdown, so it can be read and edited by hand::

    # Reachy's memory
    ...
    ## Where things are
    - fridge: behind me on my left (+114°) · seen 2026-10-16
    - keys: in the kitchen drawer
    ## About the user
    - name: Sam
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# Section key (what Gemini passes) -> heading in the file.
SECTIONS = {
    "places": "Where things are",
    "user": "About the user",
    "reachy": "About me (Reachy)",
    "notes": "Other notes",
}
_MAX_ITEMS = 40       # facts per section, so the prompt stays small
_MAX_ITEM_CHARS = 60
_MAX_DETAIL_CHARS = 300
_TITLE = "# Reachy's memory"
_INTRO = "Kept by Reachy across conversations. One fact per line: `- item: detail`. Edit freely."
_LEADING_WORDS = ("the ", "a ", "an ", "my ", "your ")
# A fact: "- item: detail". The colon needs a space after it, so lines like
# "- docs at https://…" or "- 10:30 standup" are notes, kept as written.
_FACT = re.compile(r"^- ([^:]+):\s+(\S.*)$")
# The end of a place look_at_scene saw (note_place). Only these are dropped when a search
# doesn't find the thing or the section is full; places the user told Reachy are kept.
_SEEN = re.compile(r"· seen (\d{4}-\d{2}-\d{2})$")


def memory_path() -> Path:
    """REACHY_MEMORY from env / .env (relative paths are under the repo), else
    ``<repo>/reachy_memory.md``."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    raw = os.path.expanduser(os.environ.get("REACHY_MEMORY", "").strip())
    if not raw:
        return REPO / "reachy_memory.md"
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def item_key(text: str) -> str:
    """How items are matched: 'The Red Cup' and 'red cup' are the same item."""
    key = " ".join(str(text).replace(":", " -").lower().split())
    for word in _LEADING_WORDS:
        if key.startswith(word):
            key = key[len(word):]
            break
    return key.strip(" .:-")


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _fact(line: str) -> Optional[tuple[str, str]]:
    """(item, detail) if *line* is a fact, else None."""
    m = _FACT.match(line)
    return (m.group(1).strip(), m.group(2)) if m else None


def _find(lines: list[str], item: str) -> Optional[int]:
    """Index of the fact for *item* in a section's lines, or None."""
    key = item_key(item)
    return next((n for n, line in enumerate(lines) if (f := _fact(line)) and item_key(f[0]) == key), None)


def _oldest_seen(lines: list[str]) -> Optional[int]:
    """Index of the place look_at_scene saw longest ago, or None if there's none."""
    seen = [(m.group(1), n) for n, line in enumerate(lines) if _fact(line) and (m := _SEEN.search(line))]
    return min(seen)[1] if seen else None


class Memory:
    """The memory file, in memory. Thread-safe; every change picks up edits made to the
    file meanwhile and is written straight back (atomically). If the file can't be read
    or is read-only, nothing is ever written over it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._writable = True
        self._stamp: Optional[tuple[int, int]] = None  # the file as last read or written
        self._load()

    # ----------------------------------------------------------------- reading
    def _load(self) -> None:
        """Read the file, or create it if it isn't there. Every line is kept as written:
        the ones above the first heading (bar the standard title and intro), then each
        section's."""
        self._preamble: list[str] = []
        self._sections: dict[str, list[str]] = {h: [] for h in SECTIONS.values()}
        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            logger.info("Memory: creating %s", self.path)
            self._write()
            return
        except (OSError, ValueError) as exc:  # unreadable, or not UTF-8
            logger.warning("Memory: can't read %s (%s); starting empty and not saving", self.path, exc)
            self._writable = False
            return
        if not os.access(self.path, os.W_OK):
            logger.warning("Memory: %s is read-only; not saving to it", self.path)
            self._writable = False
        block = self._preamble
        for line in text.splitlines():
            line = line.rstrip()
            if line.startswith("## "):
                block = self._sections.setdefault(line[3:].strip(), [])
            elif block is not self._preamble or line not in (_TITLE, _INTRO):
                block.append(line)
        for lines in (self._preamble, *self._sections.values()):
            while lines and not lines[0]:
                lines.pop(0)
            while lines and not lines[-1]:
                lines.pop()
        self._stamp = self._stat()

    def _stat(self) -> Optional[tuple[int, int]]:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _refresh(self) -> None:
        """Pick up edits made to the file since it was last read or written (call with
        the lock held)."""
        if self._writable and self._stat() != self._stamp:
            logger.info("Memory: %s changed on disk; reloading it", self.path)
            self._load()

    def text(self) -> str:
        """The whole memory as Markdown (what goes into the prompt)."""
        with self._lock:
            return self._render()

    def facts(self) -> int:
        with self._lock:
            return sum(1 for lines in self._sections.values() for line in lines if _fact(line))

    def recall(self, section: str, item: str) -> Optional[str]:
        """The detail saved for *item*, or None."""
        heading = SECTIONS.get(section, section)
        with self._lock:
            self._refresh()
            lines = self._sections.get(heading, [])
            at = _find(lines, item)
            return _fact(lines[at])[1] if at is not None else None

    # ----------------------------------------------------------------- writing
    def remember(self, section: str, item: str, detail: str) -> dict:
        """Save *detail* as *item* in *section*, replacing what was there; an empty
        detail forgets the item. Returns a small result for Gemini."""
        if section not in SECTIONS:
            return {"ok": False, "error": f"unknown section {section!r}; use one of {', '.join(SECTIONS)}"}
        item = _one_line(item).replace(":", " -")
        detail = _one_line(detail)[:_MAX_DETAIL_CHARS].strip()
        if not item_key(item) or len(item) > _MAX_ITEM_CHARS:
            return {"ok": False, "error": f"item needs a short name (at most {_MAX_ITEM_CHARS} characters)"}
        heading = SECTIONS[section]
        with self._lock:
            self._refresh()
            lines = self._sections[heading]
            at = _find(lines, item)
            if not detail:
                if at is None:
                    return {"ok": True, "note": f"nothing saved for {item!r}"}
                del lines[at]
                logger.info("Memory: forgot %s / %s", section, item)
                return self._save({"ok": True, "forgot": item})
            result = {"ok": True, "saved": f"{item}: {detail}"}
            if at is not None:
                lines[at] = f"- {item}: {detail}"
            else:
                if sum(1 for line in lines if _fact(line)) >= _MAX_ITEMS:
                    # Full: make room by dropping the place seen longest ago, never a fact
                    # the user gave or wrote.
                    oldest = _oldest_seen(lines) if section == "places" else None
                    if oldest is None:
                        return {"ok": False, "error": f"{heading!r} is full; ask the user what to forget"}
                    result["dropped"] = _fact(lines.pop(oldest))[0]
                lines.append(f"- {item}: {detail}")
            logger.info("Memory: %s / %s = %s", section, item, detail)
            return self._save(result)

    def note_place(self, name: str, where: str, bearing_deg: float) -> None:
        """Where look_at_scene saw *name* (replaces where it was before)."""
        today = datetime.date.today().isoformat()
        result = self.remember("places", item_key(name), f"{where} ({bearing_deg:+.0f}°) · seen {today}")
        if not result["ok"]:
            logger.info("Memory: not saving where %r is: %s", name, result["error"])

    def forget_place(self, name: str) -> None:
        """A full search found no *name*: it isn't where look_at_scene last saw it. A place
        the user told Reachy is kept (it may be out of sight, like keys in a drawer)."""
        with self._lock:
            self._refresh()
            lines = self._sections[SECTIONS["places"]]
            at = _find(lines, name)
            if at is None or not _SEEN.search(lines[at]):
                return
            del lines[at]
            logger.info("Memory: forgot places / %s (not found)", item_key(name))
            self._save({})

    def _save(self, result: dict) -> dict:
        """Write the file (call with the lock held)."""
        if not self._write():
            result["warning"] = "kept for this conversation only: couldn't write the memory file"
        return result

    def _render(self) -> str:
        out = [_TITLE, "", _INTRO]
        if self._preamble:
            out += ["", *self._preamble]
        for heading, lines in self._sections.items():
            out += ["", f"## {heading}", *lines]
        return "\n".join(out) + "\n"

    def _write(self) -> bool:
        if not self._writable:
            return False
        try:
            target = self.path.resolve()  # through a symlink, to the real file
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(self._render(), encoding="utf-8")
            try:
                shutil.copymode(target, tmp)  # keep the file's permissions
            except FileNotFoundError:
                pass
            tmp.replace(target)
        except OSError as exc:
            logger.warning("Memory: couldn't write %s: %s", self.path, exc)
            return False
        self._stamp = self._stat()
        return True
