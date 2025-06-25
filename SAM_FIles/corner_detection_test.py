import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load and preprocess image ---
image_path = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\My_better_CLB_videos\5-9\s1_tifs\frame_0029.tif"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)  # Histogram equalization for better contrast

# Optional: Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

edges = cv2.Canny(gray_blur, threshold1=10, threshold2=150)
# --- Harris Corner Detection ---
harris_img = image_rgb.copy()
gray_float = np.float32(gray_blur)
dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
harris_img[dst > 0.001 * dst.max()] = [255, 0, 0]  # Red marks

# --- Shi-Tomasi Corner Detection ---
shi_img = image_rgb.copy()
corners = cv2.goodFeaturesToTrack(edges, maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=20)
if corners is not None:
    corners = np.intp(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(shi_img, (x, y), 20, (0, 255, 0), -1)  # Green circles

# --- Display ---
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(edges)
axs[0].set_title("Harris Corners (Red)")
axs[0].axis("off")

axs[1].imshow(shi_img)
axs[1].set_title("Shi-Tomasi Corners (Green)")
axs[1].axis("off")

plt.tight_layout()
plt.show()
