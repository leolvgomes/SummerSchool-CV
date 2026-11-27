# Day 4 – Final project: corridor video

This project applies the whole course (Days 1–3) to `corridor_video.mp4`, a 17.5 s clip at 1024×576 and about 30 fps. The clip shows five people walking down a corridor toward the camera.

## How to run (in order)

```powershell
python -m pip install -r requirements.txt
python 01_extract_frames.py      # Day 2 – 2 frames per second -> video_frames/ (35 images)
python 02_image_processing.py    # Day 1 – size/pixel, RGB channels, grayscale, resize, crop, blur, Canny -> results/day1/
python 03_detect_one.py          # Day 2 – pretrained YOLO12n on one frame -> results/detect_one.jpg
python 04_label_img.py           # Day 2 – opens LabelImg to review the labels (class: person)
python 05_build_dataset.py       # Day 2 – split into train (28) / val (7, every 5th frame)
python 06_train_and_detect.py    # Day 2 – fine-tune YOLO12n, 10 epochs -> runs/yolo12_training/weights/best.pt
python 07_track_count.py 0.10    # Day 3 – ByteTrack + line counting, confidence 0.10
python 07_track_count.py 0.50    # Day 3 – same with confidence 0.50
python 08_track_trails.py        # Day 3 – trails + counting line + live counter -> runs/corridor_with_trails.mp4
```

Steps 4–6 are already done: the labels and the trained `best.pt` are in the folder. Re-run them only if you change the labels.

## Notes on each step

- **Labels:** the frames were pre-labeled with the pretrained YOLO12n (COCO `person` class) and then checked by eye. Two false boxes were removed: in frames 17 and 29 the model had marked a person's **reflection on the glass door** on the right. In frame 11 the box on the head at the bottom-right is a real person passing close to the camera, so it was kept.
- **Training:** 10 epochs at 640 px on the CPU. Final validation: precision 0.999, recall 1.0, mAP50 0.995, mAP50-95 0.914. All 10 people in the 7 validation frames were found, with no false positives.
- **Counting rule:** people walk toward the camera, so the reference line is **horizontal, at 70% of the frame height**. A track is counted once when its box center goes from above the line to below it.

## Tracking observations (Day 3 table)

Timestamps = (frame − 1) / 29.98.

| Observation | 0.10 | 0.50 |
|---|---|---|
| First missing box timestamp | 5.47 s (frame 165): person ID 2 loses the box for one frame while leaving on the left | 11.97 s (frame 360): person ID 5 loses the box for one frame while leaving on the left |
| First ID change timestamp | none | none |
| People crossing the line | 5 (2.20, 3.33, 6.77, 9.00 and 12.67 s) | 5 (same frames) |
| Track IDs seen | 5 (1, 2, 7, 8, 11) | 5 (1, 2, 4, 5, 6) |

- Every one-frame gap happens when a person is **leaving the frame at the side**, half cut off. The tracker keeps the same ID when the box comes back, so no count is lost.
- At 0.10 the boxes last 2–4 frames longer at the edges, but there are more one-frame gaps (3, versus 2 at 0.50).
- At 0.50 the IDs are numbered 1, 2, 4, 5, 6, while at 0.10 they jump to 7, 8 and 11. That means at 0.10 ByteTrack opened some short-lived candidate tracks from low-confidence boxes that were never confirmed. They never appear in the video.
- **Hand count:** five people cross the line (two together at the start, then one in black, one in green and one in white). This matches the console total for both runs.
