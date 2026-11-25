import cv2

filename = "car.png"
image_bgr = cv2.imread(filename)
if image_bgr is None:
    raise FileNotFoundError("Check the image filename and folder.")

print("Shape:", image_bgr.shape)
print("Data type:", image_bgr.dtype)
