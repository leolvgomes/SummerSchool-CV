"""Step 2 (Day 1): OpenCV basics on one frame of the video."""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
IMAGE = PROJECT / "video_frames" / "frame_004.jpg"
OUTPUT = PROJECT / "results" / "day1"
OUTPUT.mkdir(parents=True, exist_ok=True)

image_bgr = cv2.imread(str(IMAGE))
if image_bgr is None:
    raise FileNotFoundError(f"Run 01_extract_frames.py first. Missing: {IMAGE}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Size, channels and one pixel
height, width, channels = image_rgb.shape
print("Height:", height, "| Width:", width, "| Channels:", channels, "| dtype:", image_rgb.dtype)
print("RGB pixel at (x=100, y=50):", image_rgb[50, 100])


def show(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=100, bbox_inches="tight")
    plt.show()


# 1. Original
fig = plt.figure(figsize=(8, 4.5))
plt.imshow(image_rgb)
plt.title("Original frame")
plt.axis("off")
show(fig, "01_original.png")

# 2. RGB channels
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, name in enumerate(["Red", "Green", "Blue"]):
    axes[i].imshow(image_rgb[:, :, i], cmap="gray", vmin=0, vmax=255)
    axes[i].set_title(name)
    axes[i].axis("off")
show(fig, "02_rgb_channels.png")

# 3. Grayscale
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(image_rgb)
axes[0].set_title("RGB image")
axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Grayscale image")
for ax in axes:
    ax.axis("off")
show(fig, "03_grayscale.png")

# 4. Resize (keep aspect ratio)
new_width = 512
new_height = round(height * new_width / width)
resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
print("Original:", image_rgb.shape, "| Resized:", resized.shape)
fig = plt.figure(figsize=(6, 3.5))
plt.imshow(resized)
plt.title(f"Resized to {new_width} x {new_height}")
plt.axis("off")
show(fig, "04_resize.png")

# 5. Crop the region where people walk (center of the corridor)
x1, x2 = int(0.30 * width), int(0.65 * width)
y1, y2 = int(0.20 * height), int(1.00 * height)
crop = image_rgb[y1:y2, x1:x2].copy()
print("Crop shape:", crop.shape)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [16, 7]})
ax_a.imshow(image_rgb)
ax_a.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="yellow", lw=2))
ax_a.set_title("Original with crop region")
ax_b.imshow(crop)
ax_b.set_title("Cropped region")
for ax in (ax_a, ax_b):
    ax.axis("off")
show(fig, "05_crop.png")

# 6. Gaussian blur
blur5 = cv2.GaussianBlur(image_rgb, (5, 5), 0)
blur21 = cv2.GaussianBlur(image_rgb, (21, 21), 0)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, img, title in zip(axes, [image_rgb, blur5, blur21], ["Original", "5 x 5", "21 x 21"]):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
show(fig, "06_gaussian_blur.png")

# 7. Canny edges
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[1].imshow(edges, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Canny edges")
for ax in axes:
    ax.axis("off")
show(fig, "07_canny_edges.png")

print(f"Figures saved to {OUTPUT}")
