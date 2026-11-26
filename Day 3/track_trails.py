from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
VIDEO = PROJECT / "traffic_video.mp4"
WEIGHTS = PROJECT / "best.pt"
OUTPUT = PROJECT / "runs" / "traffic_with_all_trails.mp4"
TRAIL_LENGTH = 40
CONFIDENCE = 0.10

# OpenCV colors are in BGR order.
TRAIL_COLORS = [
    (80, 255, 80),
    (0, 220, 255),
    (255, 180, 60),
    (210, 100, 255),
    (255, 220, 80),
]


def main():
    for path in (VIDEO, WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    capture = cv2.VideoCapture(str(VIDEO))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Could not read the video frame rate: {VIDEO}")
    finally:
        capture.release()

    model = YOLO(str(WEIGHTS))
    print(f"Checkpoint classes: {model.names}", flush=True)

    results = model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        device="cpu",
        conf=CONFIDENCE,
        stream=True,
        save=False,
        verbose=False,
    )

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames_written = 0

    try:
        for result in results:
            frame = result.plot()
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                centers = boxes.xywh[:, :2].cpu().tolist()

                for track_id, (center_x, center_y) in zip(track_ids, centers):
                    trail = trails[track_id]
                    trail.append((round(center_x), round(center_y)))

                    if len(trail) >= 2:
                        points = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
                        color = TRAIL_COLORS[track_id % len(TRAIL_COLORS)]
                        cv2.polylines(
                            frame, [points], False, color, 3, cv2.LINE_AA
                        )

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(OUTPUT),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {OUTPUT}")

            writer.write(frame)
            frames_written += 1

    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"Frames saved: {frames_written}")
    print(f"Track IDs seen: {len(trails)}")
    print(f"Saved video: {OUTPUT}")


if __name__ == "__main__":
    main()