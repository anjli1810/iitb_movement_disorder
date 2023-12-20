import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('/content/line3.jpg')
image2 = cv2.imread('/content/line.png')

# Calculate the slope of the lines in each image
def calculate_line_slopes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use the Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    line_slopes = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # Calculate the slope of the line
            slope = -1 / np.tan(theta)  # Slope = -1/tan(θ)
            line_slopes.append(slope)

    return line_slopes

# Calculate slopes for the lines in each image
slopes_image1 = calculate_line_slopes(image1)
slopes_image2 = calculate_line_slopes(image2)

# Compare the orientation of the first line in both images
if len(slopes_image1) > 0 and len(slopes_image2) > 0:
    # Use the first line from each image for comparison
    slope1 = slopes_image1[0]
    slope2 = slopes_image2[0]

    # Calculate the angular difference
    angular_difference = abs(np.arctan(slope1) - np.arctan(slope2))

    # Define a threshold for line orientation comparison
    angle_threshold = np.radians(10)  # You can adjust this threshold

    # Compare the orientation of lines
    if angular_difference < angle_threshold:
        print("The lines in the images are in similar orientation.")
        print(f"Angular Difference: {np.degrees(angular_difference)} degrees")
    else:
        print("The lines in the images are not in similar orientation.")
        print(f"Angular Difference: {np.degrees(angular_difference)} degrees")
else:
    print("No lines were detected in one or both of the images.")
