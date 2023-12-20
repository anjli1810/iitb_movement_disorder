import cv2
import numpy as np

# Load the hand-drawn image you want to analyze
image = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Perform edge detection (Canny)
edges = cv2.Canny(blurred, 30, 100)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a flag to check if a loop or circle is detected
loop_detected = False

# Loop through the detected contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Check if the area is within a certain threshold
    if area > 500 and area < 5000:
        # Calculate the aspect ratio of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Check if the aspect ratio is close to 1 (indicating a circular shape)
        if 0.8 < aspect_ratio < 1.2:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            loop_detected = True

# Display the result and check if a loop or circle was detected
if loop_detected:
    print("A loop or circle is detected in the hand-drawn image.")
else:
    print("No loop or circle is detected in the hand-drawn image.")

# Save the image with detected contours
cv2.imwrite('output_image.png', image)
