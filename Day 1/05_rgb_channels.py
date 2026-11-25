import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

channels = ["Red", "Green", "Blue"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, name in enumerate(channels):
    axes[i].imshow(image_rgb[:, :, i], cmap="gray", vmin=0, vmax=255)
    axes[i].set_title(name)
    axes[i].axis("off")
plt.tight_layout()
plt.show()
