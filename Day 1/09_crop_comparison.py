import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]
x1, x2 = int(0.30 * width), int(0.88 * width)
y1, y2 = int(0.35 * height), int(0.83 * height)
crop = image_rgb[y1:y2, x1:x2].copy()

print("Original shape:", image_rgb.shape)
print("Crop shape:", crop.shape)

fig, (ax_original, ax_crop) = plt.subplots(1, 2, figsize=(12, 5))
ax_original.imshow(image_rgb)
ax_original.set_title("Original image")
ax_original.axis("off")
ax_crop.imshow(crop)
ax_crop.set_title("Cropped car region")
ax_crop.axis("off")
fig.tight_layout()
plt.show()
