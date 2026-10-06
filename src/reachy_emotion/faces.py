"""Face finding on Reachy's camera, with the Haar cascade bundled in opencv-python.

Used to (1) crop the emotion model's input around the user's face — without a
face detector the SDK centre-crops the wide 16:9 frame, which usually cuts off an
off-centre user — and (2) find the user when ``look_at_scene`` searches the room.
Frontal faces only; faces smaller than ~70 px at 1080p (beyond ~2 m) are missed.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h in full-frame pixels

_DETECT_WIDTH = 640  # detect on a downscaled copy (~10-20 ms), map the box back
_CROP_SCALE = 2.0    # crop side = 2 x face width: the face fills ~half the crop, like training
_HOLD_S = 1.0        # keep the last face box through brief dropouts (blink, head turn)

_cascade = None
_cascade_lock = threading.Lock()  # a CascadeClassifier isn't safe to share across threads


def detect_face(frame_bgr: np.ndarray) -> Optional[Box]:
    """The largest frontal face in the frame, or None."""
    import cv2

    global _cascade
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, _DETECT_WIDTH / w)
    small = cv2.resize(frame_bgr, (round(w * scale), round(h * scale))) if scale < 1.0 else frame_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    with _cascade_lock:
        if _cascade is None:
            _cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return round(x / scale), round(y / scale), round(fw / scale), round(fh / scale)


def square_crop(frame_bgr: np.ndarray, box: Box) -> np.ndarray:
    """Square of side _CROP_SCALE x face size centred on the face (edges replicated)."""
    import cv2

    x, y, w, h = box
    side = round(_CROP_SCALE * max(w, h))
    x0, y0 = x + w // 2 - side // 2, y + h // 2 - side // 2
    height, width = frame_bgr.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - width, y0 + side - height)
    if pad:
        frame_bgr = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return frame_bgr[y0:y0 + side, x0:x0 + side]


def _near(a: Box, b: Box) -> bool:
    """Whether two boxes' centres are within a face size of each other."""
    return (
        abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) < max(a[2], b[2])
        and abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < max(a[3], b[3])
    )


class FaceCropper:
    """Crops frames around the user's face for the emotion model.

    No face: the last box is kept for _HOLD_S, then frames pass through unchanged
    (the SDK's own centre crop — the behaviour before face cropping). While a face
    is being followed, a face somewhere else must be seen twice in a row before the
    crop moves there, so one stray detection (a face in a painting) can't grab it.
    """

    def __init__(self, detect: Callable[[np.ndarray], Optional[Box]] = detect_face) -> None:
        self._detect = detect
        self._box: Optional[Box] = None
        self._seen = 0.0
        self._pending: Optional[Box] = None

    def crop(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (frame for the model, whether it is a face crop)."""
        now = time.monotonic()
        box = self._detect(frame_bgr)
        following = self._box is not None and now - self._seen <= _HOLD_S
        if box is None:
            self._pending = None
        elif (
            not following
            or _near(box, self._box)
            or (self._pending is not None and _near(box, self._pending))
        ):
            self._box, self._seen, self._pending, following = box, now, None, True
        else:
            self._pending = box
        if following:
            return square_crop(frame_bgr, self._box), True
        return frame_bgr, False
