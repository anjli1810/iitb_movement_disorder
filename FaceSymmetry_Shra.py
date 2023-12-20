import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")

presentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            landmarks = faceLms.landmark
            ih, iw, ic = img.shape

            # Extract the x and y coordinates of each landmark
            coords = [(lm.x * iw, lm.y * ih) for lm in landmarks]

            # Calculate the symmetry measure (e.g., using Euclidean distance)
            left_coords = coords[:234]  # Assuming 468 landmarks
            right_coords = coords[234:468]  # Mirror the left landmarks

            distances = np.linalg.norm(np.array(left_coords) - right_coords, axis=1)

            # Calculate the symmetry score (you can use a threshold)
            symmetry_score = np.mean(distances)

            # Draw landmarks for both halves
            for x, y in left_coords:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red color for left half
            for x, y in right_coords:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green color for right half

            # Display the symmetry score
            cv2.putText(img, f'Symmetry: {symmetry_score:.2f}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # fps control
    currentTime = time.time()
    fps_rate = 1 / (currentTime - presentTime)
    presentTime = currentTime

    cv2.putText(img, f'fps: {int(fps_rate)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('Facial Landmarks and Symmetry', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()