import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
smooth = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(smooth, 100, 200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image_rgb)
axes[0].set_title("original")
axes[1].imshow(edges, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Canny edges")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
