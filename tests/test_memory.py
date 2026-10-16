"""Tests for Reachy's long-term memory file (memory.py)."""

from __future__ import annotations

import datetime
import os

import pytest

from reachy_emotion import memory as mem

TODAY = datetime.date.today().isoformat()


def test_a_missing_file_is_created_with_every_section(tmp_path):
    path = tmp_path / "reachy_memory.md"
    m = mem.Memory(path)
    text = path.read_text()
    for heading in mem.SECTIONS.values():
        assert f"## {heading}" in text
    assert m.facts() == 0 and m.text() == text


def test_remember_saves_replaces_and_forgets(tmp_path):
    path = tmp_path / "m.md"
    m = mem.Memory(path)
    assert m.remember("user", "name", "Sam")["ok"]
    assert m.remember("places", "The Fridge", "behind me on my left")["ok"]
    assert m.remember("places", "fridge", "on my right")["ok"]  # same item: replaced, not added
    assert m.recall("places", "the fridge") == "on my right"
    assert "- name: Sam" in path.read_text() and "behind me" not in path.read_text()
    assert m.remember("places", "fridge", "") == {"ok": True, "forgot": "fridge"}
    assert m.recall("places", "fridge") is None
    assert m.remember("places", "sofa", "") == {"ok": True, "note": "nothing saved for 'sofa'"}


def test_a_new_conversation_loads_what_the_last_one_saved(tmp_path):
    path = tmp_path / "m.md"
    mem.Memory(path).remember("reachy", "jokes", "keep them short")
    again = mem.Memory(path)
    assert again.recall("reachy", "jokes") == "keep them short" and again.facts() == 1


def test_hand_edits_survive_a_save(tmp_path):
    path = tmp_path / "m.md"
    path.write_text(
        "\ufeff# Reachy's memory\n\nAlways speak softly to Sam.\n\n"  # a BOM, a note above the sections
        "## About the user\n- name: Sam\nLikes tea in the morning.\n\n"
        "## Other notes\n- docs at https://example.com/x\n- 10:30 standup every day\n\n"
        "## Projects\n- robot: Reachy + SO-101\n"
    )
    m = mem.Memory(path)
    assert "Always speak softly to Sam." in m.text()  # in the prompt too
    m.remember("user", "city", "Austin")
    text = path.read_text()
    assert "\nAlways speak softly to Sam.\n" in text
    assert "- name: Sam\nLikes tea in the morning.\n- city: Austin" in text  # free text kept
    assert "- docs at https://example.com/x\n- 10:30 standup every day" in text  # colons as written
    assert "## Projects\n- robot: Reachy + SO-101" in text  # a section of the user's own kept
    assert mem.Memory(path).text() == text  # and it reads back the same


def test_edits_made_while_running_are_kept(tmp_path):
    path = tmp_path / "m.md"
    m = mem.Memory(path)
    m.remember("user", "name", "Sam")
    path.write_text(path.read_text().replace("- name: Sam", "- name: Sam\n- tea: no sugar"))
    m.remember("user", "city", "Austin")
    assert "- name: Sam\n- tea: no sugar\n- city: Austin" in path.read_text()
    assert m.recall("user", "tea") == "no sugar"


def test_bad_input_is_refused(tmp_path):
    m = mem.Memory(tmp_path / "m.md")
    assert not m.remember("secrets", "x", "y")["ok"]
    assert not m.remember("notes", "   ", "y")["ok"]  # no name
    assert not m.remember("notes", "x" * 61, "y")["ok"]  # not a short name
    m.remember("notes", "a: b", "line one\nline two")
    assert m.recall("notes", "a: b") == "line one line two"  # one line, no stray colon
    assert "- a - b: line one line two" in m.path.read_text()


def test_a_full_section_drops_only_places_reachy_saw(tmp_path, monkeypatch):
    monkeypatch.setattr(mem, "_MAX_ITEMS", 3)
    m = mem.Memory(tmp_path / "m.md")
    for i in range(3):
        assert m.remember("user", f"rule {i}", "x")["ok"]
    assert "full" in m.remember("user", "rule 3", "x")["error"]  # nothing given is dropped
    assert m.remember("user", "rule 0", "y")["ok"]  # but it can still be changed
    m.remember("places", "keys", "in the kitchen drawer")  # told by the user
    m.note_place("cup", "on my left", 90)
    m.note_place("book", "ahead", 0)
    m.path.write_text(m.path.read_text().replace(f"cup: on my left (+90°) · seen {TODAY}", "cup: on my left (+90°) · seen 2020-01-01"))
    assert m.remember("places", "phone", "on the sofa") == {"ok": True, "saved": "phone: on the sofa", "dropped": "cup"}
    m.note_place("pen", "ahead", 0)  # makes room by dropping the book, never the keys
    assert [m.recall("places", x) is not None for x in ("keys", "cup", "book", "phone", "pen")] == [
        True, False, False, True, True,
    ]


def test_places_are_saved_with_bearing_and_date(tmp_path):
    m = mem.Memory(tmp_path / "m.md")
    m.note_place("the fridge", "behind me on my left", 114.4)
    assert m.recall("places", "fridge") == f"behind me on my left (+114°) · seen {TODAY}"
    m.forget_place("Fridge")
    assert m.recall("places", "fridge") is None
    long = "the small red ceramic coffee mug with a chipped handle on the desk"  # 66 characters
    m.note_place(long, "ahead", 0)
    m.note_place("mug: blue", "ahead", 0)
    m.forget_place(long)
    m.forget_place("mug: blue")
    assert m.facts() == 0  # forgets exactly what it saved


def test_a_failed_search_keeps_places_the_user_gave(tmp_path):
    m = mem.Memory(tmp_path / "m.md")
    m.remember("places", "my keys", "in the kitchen drawer")  # out of sight
    m.forget_place("the keys")
    assert m.recall("places", "keys") == "in the kitchen drawer"


def _unchanged_after_a_save(path):
    before = path.read_bytes()
    result = mem.Memory(path).remember("user", "name", "Sam")
    assert result["ok"] and "warning" in result  # kept for this conversation only
    assert path.read_bytes() == before and not path.with_name(path.name + ".tmp").exists()


def test_an_unreadable_file_is_never_overwritten(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can read anything")
    path = tmp_path / "m.md"
    path.write_text("# Reachy's memory\n\n## About the user\n- name: Alex\n")
    path.chmod(0)
    try:
        before = path.stat().st_size
        result = mem.Memory(path).remember("user", "name", "Sam")
        assert result["ok"] and "warning" in result
        assert path.stat().st_size == before and not (tmp_path / "m.md.tmp").exists()
    finally:
        path.chmod(0o600)


def test_a_file_that_isnt_utf8_is_never_overwritten(tmp_path):
    path = tmp_path / "m.md"
    path.write_bytes("## About the user\n- city: Montréal\n".encode("latin-1"))
    _unchanged_after_a_save(path)


def test_a_read_only_file_is_never_overwritten(tmp_path):
    path = tmp_path / "m.md"
    path.write_text("## About the user\n- name: Alex\n")
    path.chmod(0o444)
    try:
        _unchanged_after_a_save(path)
    finally:
        path.chmod(0o600)


def test_a_symlink_and_permissions_are_kept(tmp_path):
    real, link = tmp_path / "real.md", tmp_path / "link.md"
    real.write_text("## About the user\n")
    real.chmod(0o600)
    link.symlink_to(real)
    mem.Memory(link).remember("user", "name", "Sam")
    assert link.is_symlink() and "- name: Sam" in real.read_text()
    assert real.stat().st_mode & 0o777 == 0o600


def test_memory_path_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr("reachy_emotion.conversation_app._load_env", lambda: None)
    monkeypatch.delenv("REACHY_MEMORY", raising=False)
    assert mem.memory_path() == mem.REPO / "reachy_memory.md"
    monkeypatch.setenv("REACHY_MEMORY", str(tmp_path / "x.md"))
    assert mem.memory_path() == tmp_path / "x.md"
    monkeypatch.setenv("REACHY_MEMORY", "data/m.md")
    assert mem.memory_path() == mem.REPO / "data/m.md"
