# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# === Load images ===
faceImage = cv2.imread("my.jpg")                      # Face image
glassPNG = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)  # Sunglass PNG

# === Check if loaded ===
if faceImage is None:
    raise FileNotFoundError("Could not find 'my.jpg'.")
if glassPNG is None:
    raise FileNotFoundError("Could not find 'sunglass.png'.")

# === 1️⃣ Display Original Face (with axes) ===
plt.figure(figsize=[8,8])
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Face (with axes)")
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")
plt.show()

# === 2️⃣ Handle Sunglasses PNG ===
if glassPNG.shape[2] == 4:
    glassBGR = glassPNG[:, :, :3]
    glassAlpha = glassPNG[:, :, 3]
else:
    glassBGR = glassPNG
    glassAlpha = np.ones(glassBGR.shape[:2], dtype=np.uint8) * 255

# Create a mask for visualization
_, glassMask = cv2.threshold(glassAlpha, 10, 255, cv2.THRESH_BINARY)
glassMask_inv = cv2.bitwise_not(glassMask)

# === Display Sunglasses and Mask (with axes) ===
plt.figure(figsize=[15,5])
plt.subplot(131)
plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Original Sunglasses")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot(132)
plt.imshow(glassMask, cmap='gray')
plt.title("Sunglasses Mask (white = visible)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot(133)
plt.imshow(glassMask_inv, cmap='gray')
plt.title("Inverse Mask (black = visible area removed)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# === 3️⃣ Initialize MediaPipe FaceMesh ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

rgb_img = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_img)

h, w, _ = faceImage.shape

# === 4️⃣ Apply Sunglasses ===
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Eye landmarks
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
        x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

        # Size and position
        eye_width = x2 - x1
        new_w = int(eye_width * 2.0)
        new_h = int(new_w * glassBGR.shape[0] / glassBGR.shape[1])

        x = x1 - int(new_w * 0.25)
        y = y1 - int(new_h * 0.5)

        # Keep inside image
        x = max(0, x)
        y = max(0, y)
        new_w = min(new_w, w - x)
        new_h = min(new_h, h - y)

        # Resize sunglasses and alpha
        glass_resized = cv2.resize(glassBGR, (new_w, new_h))
        alpha_resized = cv2.resize(glassAlpha, (new_w, new_h))

        # Create masks
        _, mask = cv2.threshold(alpha_resized, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        roi = faceImage[y:y+new_h, x:x+new_w]

        # Fix mismatches
        if roi.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
            mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))
            glass_resized = cv2.resize(glass_resized, (roi.shape[1], roi.shape[0]))

        # Combine
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(glass_resized, glass_resized, mask=mask)
        combined = cv2.add(bg, fg)

        # Overlay result
        faceImage[y:y+new_h, x:x+new_w] = combined

else:
    print("⚠️ No face detected! Try another image or better lighting.")

# === 5️⃣ Display Final Output (with axes) ===
plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Face with Auto-Aligned Sunglasses (with axes)")
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")
plt.show()

```
## Output:

<img width="609" height="691" alt="image" src="https://github.com/user-attachments/assets/b023da3a-57a5-41a1-b5a8-f325a8c9aa99" />

<img width="1277" height="232" alt="image" src="https://github.com/user-attachments/assets/e540b91e-dd86-4103-b5ff-f0c7df9c5444" />

<img width="749" height="871" alt="image" src="https://github.com/user-attachments/assets/201af6c5-1751-4631-a2b9-754ce45516d2" />


