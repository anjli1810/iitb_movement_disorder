import cv2
import numpy as np  # Import NumPy

# Load the pre-trained face detection cascade classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate and draw facial landmarks manually (example: eyes)
        eye_region = gray[y:y + h, x:x + w]
        eyes = cv2.HoughCircles(eye_region, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

        if eyes is not None:
            eyes = np.uint16(np.around(eyes))
            for eye in eyes[0, :]:
                eye_x, eye_y, eye_radius = eye
                cv2.circle(frame, (x + eye_x, y + eye_y), eye_radius, (0, 0, 255), 2)

    # Display the frame with annotations
    cv2.imshow("Facial Movement Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
