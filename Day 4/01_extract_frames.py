"""Step 1 (Day 2): extract 2 images per second from the video."""
from pathlib import Path
import cv2

PROJECT = Path(__file__).resolve().parent
VIDEO = PROJECT / "corridor_video.mp4"
OUTPUT = PROJECT / "video_frames"
IMAGES_PER_SECOND = 2

if not VIDEO.is_file():
    raise FileNotFoundError(f"Video file not found: {VIDEO}")

video = cv2.VideoCapture(str(VIDEO))
saved = 0
try:
    if not video.isOpened():
        raise RuntimeError(f"OpenCV could not open: {VIDEO}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read the video's frame rate.")
    print(f"Video: {VIDEO.name} | {fps:.2f} fps")

    OUTPUT.mkdir(exist_ok=True)
    frame_index = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_index >= round(saved * fps / IMAGES_PER_SECOND):
            path = OUTPUT / f"frame_{saved + 1:03d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Could not save: {path}")
            saved += 1
        frame_index += 1
finally:
    video.release()

if saved == 0:
    raise RuntimeError("No frames could be read from the video.")
print(f"Saved {saved} images to {OUTPUT}")
