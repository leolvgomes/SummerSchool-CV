"""Step 7 (Day 3): track people with ByteTrack and count line crossings.

People walk down the corridor toward the camera, so the reference line is
horizontal (70% of the frame height). A person is counted once when the
center of their box moves from above the line to below it.

Usage:  python 07_track_count.py          (confidence 0.10)
        python 07_track_count.py 0.50     (confidence 0.50)
"""
import sys
from pathlib import Path

import cv2

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
WEIGHTS = PROJECT / "runs" / "yolo12_training" / "weights" / "best.pt"
VIDEO = PROJECT / "corridor_video.mp4"
LINE_RATIO = 0.70
CONF = 0.10
if len(sys.argv) > 1:
    CONF = float(sys.argv[1])
RUN_NAME = f"confidence_{round(CONF * 100):03d}"


def main():
    for path in (WEIGHTS, VIDEO):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    model = YOLO(str(WEIGHTS))
    results = model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        device="cpu",
        conf=CONF,
        stream=True,
        save=True,
        project=str(PROJECT / "runs" / "tracking"),
        name=RUN_NAME,
        exist_ok=True,
    )

    previous_side = {}
    counted_ids = set()
    ids_seen = set()

    for frame_number, result in enumerate(results, start=1):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        line_y = result.orig_shape[0] * LINE_RATIO
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        centers_y = boxes.xywh[:, 1].cpu().tolist()

        for track_id, class_id, center_y in zip(ids, classes, centers_y):
            if result.names[class_id] != "person" or center_y == line_y:
                continue
            ids_seen.add(track_id)
            side = "above" if center_y < line_y else "below"
            if (
                previous_side.get(track_id) == "above"
                and side == "below"
                and track_id not in counted_ids
            ):
                counted_ids.add(track_id)
                print(f"Crossing at frame {frame_number} (~{(frame_number - 1) / fps:.2f} s), person ID {track_id}")
            previous_side[track_id] = side

    print(f"Person track IDs seen: {len(ids_seen)} -> {sorted(ids_seen)}")
    print(f"People crossing the line toward the camera: {len(counted_ids)}")
    print(f"Tracked video folder: {model.predictor.save_dir}")


if __name__ == "__main__":
    main()
