"""Tests for the face crop that feeds the emotion model (faces.py) — no camera."""

from __future__ import annotations

import numpy as np

from reachy_emotion import faces


def _frame():
    return np.zeros((100, 200, 3), np.uint8)


def test_detect_face_finds_nothing_in_a_blank_frame():
    assert faces.detect_face(np.zeros((1080, 1920, 3), np.uint8)) is None


def test_square_crop_is_centred_on_the_face_at_twice_its_size():
    frame = _frame()
    frame[40:60, 80:100] = 255  # the "face"
    crop = faces.square_crop(frame, (80, 40, 20, 20))
    assert crop.shape == (40, 40, 3)
    assert crop[10:30, 10:30].min() == 255 and crop[:10].max() == 0  # face in the middle


def test_square_crop_pads_at_the_frame_edge():
    crop = faces.square_crop(_frame(), (0, 40, 20, 20))
    assert crop.shape == (40, 40, 3)  # still a full square, face still centred


def _cropper(monkeypatch, boxes):
    now = [100.0]
    monkeypatch.setattr(faces.time, "monotonic", lambda: now[0])
    seq = iter(boxes)
    return faces.FaceCropper(detect=lambda f: next(seq)), now


def test_no_face_holds_briefly_then_passes_the_frame_through(monkeypatch):
    a = (80, 40, 20, 20)
    cropper, now = _cropper(monkeypatch, [a, None, None])
    frame = _frame()

    crop, visible = cropper.crop(frame)
    assert visible and crop.shape == (40, 40, 3)
    now[0] += 0.5
    crop, visible = cropper.crop(frame)  # a blink: keep the last box
    assert visible and crop.shape == (40, 40, 3)
    now[0] += 1.0
    out, visible = cropper.crop(frame)  # gone for over 1 s: the full frame, as before
    assert not visible and out is frame


def test_a_face_elsewhere_must_be_seen_twice_before_the_crop_moves(monkeypatch):
    a, b = (20, 40, 20, 20), (150, 40, 20, 20)  # b: e.g. a face in a painting
    cropper, now = _cropper(monkeypatch, [a, b, b])
    frame = _frame()
    frame[40:60, 20:40] = 1
    frame[40:60, 150:170] = 2

    assert cropper.crop(frame)[0][20, 20, 0] == 1  # following a
    now[0] += 0.3
    assert cropper.crop(frame)[0][20, 20, 0] == 1  # b once: still on a
    now[0] += 0.3
    assert cropper.crop(frame)[0][20, 20, 0] == 2  # b again: switch


def test_with_no_face_followed_a_new_face_is_used_at_once(monkeypatch):
    cropper, _ = _cropper(monkeypatch, [(150, 40, 20, 20)])
    frame = _frame()
    frame[40:60, 150:170] = 2
    crop, visible = cropper.crop(frame)
    assert visible and crop[20, 20, 0] == 2
