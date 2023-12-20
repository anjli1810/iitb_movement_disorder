import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")

presentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Define face landmark connections (list of pairs of landmark indices)
FACE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Jawline
    (5, 6), (6, 7), (7, 8),           # Right eyebrow
    (9, 10), (10, 11), (11, 12),      # Left eyebrow
    (13, 14), (14, 15), (15, 16),     # Nose bridge
    (17, 18), (18, 19), (19, 20),     # Lower nose
    (21, 22), (22, 23), (23, 24),     # Right eye
    (25, 26), (26, 27), (27, 28),     # Left eye
    (29, 30), (30, 31), (31, 32),     # Upper lip
    (33, 34), (34, 35), (35, 36),     # Lower lip
]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Draw landmarks with manually defined connections
            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection
                mpDraw.draw_landmarks(img, faceLms, [(start_idx, 0), (end_idx, 0)], drawSpec, drawSpec)
            
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
   
    # fps control
    currentTime = time.time()
    fps_rate = 1 / (currentTime - presentTime)
    presentTime = currentTime
    cv2.putText(img, f'fps: {int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow('Face Mesh Detection', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()