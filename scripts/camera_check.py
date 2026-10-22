"""Camera -> table check: where does Reachy think things on the table are?

Run with the reachy_mini daemon up and the app closed (the head moves; the arm doesn't):

    .venv/bin/python scripts/camera_check.py --query "the coins" --truth 22,29 28,25 18,37

It turns the head to the arm's work view (or --facing/--tilt), grabs a frame with the head
pose measured at that moment, asks ER 2 for the query, and prints each find's table point
(cm, Reachy's frame: x ahead of its foot centre, y to its left) as the app computes it, and
with the camera's vertical focal length scaled like its horizontal one (the SDK scales them
differently for the 16:9 crop). With --truth, the error to the nearest true point. The frame,
pose and camera model are saved under captures/ for offline fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _square_pixel_k(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """The Lite camera's K for *width*x*height* with one scale for both axes (a centred crop
    of the 3840x2592 calibration), instead of the SDK's per-axis scaling."""
    from reachy_mini.media.camera_constants import CameraResolution, ReachyMiniLiteCamSpecs

    crop = next((r.value[3] for r in CameraResolution if r.value[:2] == (width, height)), 1.0)
    K0, s = ReachyMiniLiteCamSpecs.K, width / 3840 * crop
    out = K.copy()
    out[0, 0], out[1, 1] = K0[0, 0] * s, K0[1, 1] * s
    out[0, 2], out[1, 2] = width / 2 + (K0[0, 2] - 1920) * s, height / 2 + (K0[1, 2] - 1296) * s
    return out


def main() -> int:
    from google import genai  # noqa: F401  (fail early if the app's venv isn't used)
    from reachy_mini import ReachyMini

    from reachy_emotion.arm_task import arm_mount_from_env, work_view_deg
    from reachy_emotion.conversation_app import _load_api_key
    from reachy_emotion.scene_brain import SceneBrain, camera_size, er_client, go_neutral, points_to_pixels, table_point

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--query", default="the small objects on the table")
    p.add_argument("--facing", type=float, help="body heading, degrees (+ = left); default: the arm's work view")
    p.add_argument("--tilt", type=float, default=30.0, help="head tilt, degrees down (default 30)")
    p.add_argument("--truth", nargs="*", default=[], help="true positions as X,Y in cm (Reachy's frame)")
    args = p.parse_args()
    mount = arm_mount_from_env()
    facing = args.facing if args.facing is not None else (work_view_deg(mount) if mount else 0.0)
    truths = [tuple(float(v) for v in t.split(",")) for t in args.truth]

    with ReachyMini() as mini:
        print(f"Turning to {facing:+.1f}°, tilted {args.tilt:.0f}° down...")
        go_neutral(mini, body_yaw=math.radians(facing), pitch=math.radians(args.tilt), duration=1.5)
        time.sleep(1.5)  # let the head settle and fresh frames arrive
        frame = None
        for _ in range(10):
            frame = mini.media.get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        pose = mini.get_current_head_pose()
        if frame is None:
            print("No camera frame.")
            return 1
        cam = mini.media.camera
        width, height = camera_size(mini, frame)
        K_sq = _square_pixel_k(np.asarray(cam.K, float), width, height)
        yaw = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -pose[2, 0]))))
        print(f"Measured head pose: heading {yaw:+.1f}°, pitch {pitch:+.1f}° (down +), z {pose[2, 3] * 100:+.1f} cm; "
              f"frame {frame.shape[1]}x{frame.shape[0]}, camera {width}x{height}")
        print(f"K (SDK): fx {cam.K[0][0]:.1f} fy {cam.K[1][1]:.1f} cx {cam.K[0][2]:.1f} cy {cam.K[1][2]:.1f}; "
              f"square pixels: fy {K_sq[1, 1]:.1f} cy {K_sq[1, 2]:.1f}")

        brain = SceneBrain(er_client(_load_api_key()))
        objects = brain.locate(frame, args.query)
        print(f"ER: {len(objects)} for {args.query!r}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = REPO / "captures" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        # The same geometry with the square-pixel camera model (the SDK's K is read-only).
        square = SimpleNamespace(media=SimpleNamespace(camera=SimpleNamespace(K=K_sq, D=cam.D)), T_head_cam=mini.T_head_cam)
        for o, px in zip(objects, points_to_pixels([o["point"] for o in objects], width, height)):
            a = table_point(mini, px, pose)
            b = table_point(square, px, pose)
            line = f"  {o['label']:<28s} pixel ({px[0]:4d},{px[1]:4d})"
            for name, t in (("app", a), ("square px", b)):
                if t is None:
                    line += f" | {name}: none"
                    continue
                line += f" | {name}: ({t[0] * 100:5.1f}, {t[1] * 100:5.1f})"
                if truths:
                    gx, gy = min(truths, key=lambda g: math.hypot(t[0] * 100 - g[0], t[1] * 100 - g[1]))
                    line += f" off {t[0] * 100 - gx:+5.1f},{t[1] * 100 - gy:+5.1f}"
            print(line)
            rows.append({"label": o["label"], "point": o["point"], "pixel": list(px),
                         "app_cm": None if a is None else [round(v * 100, 2) for v in a],
                         "square_px_cm": None if b is None else [round(v * 100, 2) for v in b]})
        import cv2

        cv2.imwrite(str(out_dir / "frame.jpg"), frame)
        (out_dir / "capture.json").write_text(json.dumps({
            "query": args.query, "facing_deg": facing, "tilt_deg": args.tilt, "truth_cm": truths,
            "pose": np.asarray(pose).tolist(), "K": np.asarray(cam.K).tolist(), "D": np.asarray(cam.D).tolist(),
            "camera_size": [width, height], "frame_shape": list(frame.shape), "T_head_cam": np.asarray(mini.T_head_cam).tolist(),
            "objects": rows,
        }, indent=1))
        print(f"Saved {out_dir}")
        go_neutral(mini, duration=1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
