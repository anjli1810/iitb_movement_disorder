#Smiling or Not

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")

presentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Define landmarks for left and right corners of the mouth
left_corner_id = 61
right_corner_id = 91

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            left_corner = faceLms.landmark[left_corner_id]
            right_corner = faceLms.landmark[right_corner_id]

            # Calculate the Euclidean distance between left and right corner landmarks
            distance = ((left_corner.x - right_corner.x) ** 2 + (left_corner.y - right_corner.y) ** 2) ** 0.5

            # Set a threshold for smile detection
            smile_threshold = 0.05  # You can adjust this threshold as needed

            # Check if the distance between the corners is above the threshold
            if distance > smile_threshold:
                cv2.putText(img, 'Smiling', (20, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            else:
                cv2.putText(img, 'Not Smiling', (20, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # fps control
    currentTime = time.time()
    fps_rate = 1 / (currentTime - presentTime)
    presentTime = currentTime

    cv2.putText(img, f'fps:{int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow('Face Mesh Detection', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
