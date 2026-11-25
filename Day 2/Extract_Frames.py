from pathlib import Path
import cv2

project_folder = Path(__file__).resolve().parent
video_path = project_folder / "traffic_video.mp4"
output_folder = project_folder / "video_frames"
images_per_second = 2

if not video_path.is_file():
    raise FileNotFoundError(f"Video file not found: {video_path}")

video = cv2.VideoCapture(str(video_path))
saved_count = 0

try:
    if not video.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read the video's frame rate.")

    output_folder.mkdir(exist_ok=True)
    frame_index = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        next_sample_frame = round(saved_count * fps / images_per_second)
        if frame_index >= next_sample_frame:
            output_path = output_folder / f"frame_{saved_count + 1:03d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"Could not save: {output_path}")
            saved_count += 1

        frame_index += 1
finally:
    video.release()

if saved_count == 0:
    raise RuntimeError("No frames could be read from the video.")

print(f"Saved {saved_count} images to {output_folder}")
