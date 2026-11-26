# Day 3 – Tracking observations

Checkpoint: `best.pt` from Day 2 (class `car` only) · Tracker: ByteTrack · Video: `traffic_video.mp4` (25 fps, 480 frames, 19.2 s)
Timestamps = (frame − 1) / 25.

## Step 5 – Compare confidence settings

| Observation | 0.10 | 0.50 |
|---|---|---|
| First missing box timestamp | 9.84 s (frame 247): the car's front enters on the right but has no box until 9.92 s (frame 249). The box has no track ID until 10.00 s (frame 251). | 9.84 s (frame 247): the car is entering but has no box until 10.12 s (frame 254). |
| First ID change timestamp | none (the car keeps ID 1 from 10.00 s to 16.12 s) | none (the car keeps ID 1 from 10.12 s to 16.12 s) |
| Right-to-left car crossings | 1 (frame 317, 12.64 s, car ID 1) | 1 (frame 317, 12.64 s, car ID 1) |

Extra observation at 0.10: from 11.76 s to 12.12 s (frames 295–304) there is a false positive, ID 3, at the left edge of the frame. It sits on a bicycle wheel, with confidence 0.14–0.34. It is not an ID change of the car. At 0.50 it disappears.

In short, 0.10 finds the car about 0.2 s earlier but also tracks a false positive. 0.50 is cleaner but starts tracking the car later. Both runs give the same count.

## Step 6 – Check the count by hand

Using the same center line and the right-to-left direction, one car (the black sedan) crosses the line, at about 12.6 s. The hand count is **1**, which matches the console total for both runs. There was no gap or ID change near the line, so no crossing was missed.
