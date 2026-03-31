import imagej
import numpy as np
import cv2
import tifffile
import os
# --- start ImageJ headless ---
# ij = imagej.init('sc.fiji:fiji', mode='headless')
os.environ["JAVA_HOME"] = r"C:\Users\raglo\OneDrive - University of Arizona\Desktop\Fiji\java\win64"
os.environ["_JAVA_OPTIONS"] = "-Xmx12g"
os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx12g"

ij = imagej.init(r"C:\Users\raglo\OneDrive - University of Arizona\Desktop\Fiji", mode='headless')

# --- paths ---
model_path = r"D:\giulia\classifierV1_121625.model"
video_path = r"D:\giulia\Test7_Water_Glass_Glass_100um-700um.avi"
out_dir = "masks"

# --- open video with OpenCV ---
cap = cv2.VideoCapture(video_path)
frame_idx = 0

# --- load Weka class ---
import scyjava as sj
WekaSeg = sj.jimport("trainableSegmentation.WekaSegmentation")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale if needed
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(
            frame,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

    # convert to ImageJ image
    ij_img = ij.py.to_java(frame)
    imp = ij.py.to_imageplus(ij_img)

    # apply Weka
    ws = WekaSeg(imp)
    ws.loadClassifier(model_path)

    # apply classifier (no probability stack)
    result = ws.applyClassifier(imp, 1, False)


    # convert result back to numpy
    result_np = ij.py.from_java(result)
    print("result_np shape:", result_np.shape)

    # assume class 1 = droplet edge
    mask = (result_np).astype(np.uint8) * 255

    print('test')
    full_path = os.path.join(os.path.dirname(video_path), out_dir)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    # save
    tifffile.imwrite(f"{full_path}/mask_{frame_idx:05d}.tif", mask)
    print('test2')
    frame_idx += 1

cap.release()
print("Done.")
