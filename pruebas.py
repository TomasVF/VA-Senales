import cv2
import numpy as np

def display_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_shapes(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    display_image(image, "Step 1: Original Image")

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray, "Step 2: Grayscale Image")

    # Step 3: Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    display_image(edges, "Step 3: Canny Edge Detection")

    # Step 4: Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Process each contour
    for contour in contours:
        # Step 6: Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Step 7: Get the number of vertices
        vertices = len(approx)

        # Step 8: Detect shapes based on the number of vertices
        if vertices == 3:
            shape_type = "Triangle"
        elif vertices == 4:
            shape_type = "Square"
        elif vertices >= 10:
            shape_type = "Circle"
        else:
            continue  # Ignore shapes with other vertex counts

        # Step 9: Draw the shape and label on the image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        cv2.putText(image, shape_type, tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Step 10: Display the final result
    display_image(image, "Step 10: Shape Detection")

# Replace 'your_image.ppm' with the path to your .ppm image
image_path = '00001.ppm'
detect_shapes(image_path)
