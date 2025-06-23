import cv2
import numpy as np
import math

def find_long_lines(image_path, min_length=800):
    """
    Finds straight lines in an image longer than a specified minimum length.

    Args:
        image_path (str): The path to the image file.
        min_length (int): The minimum length of lines to detect (in pixels).

    Returns:
        list: A list of tuples, where each tuple (x1, y1, x2, y2) represents
              the coordinates of a detected line segment.
    """
    # Step 3: Implement image loading
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return []

    # Step 4: Implement image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Hough Attempt 3: Canny(30,90) with lenient Hough
    edges = cv2.Canny(gray, 30, 90, apertureSize=3)
    # cv2.imwrite("data/canny_edges_30_90_Hough3.png", edges) # Optional: save intermediate Canny output

    # Step 5: Implement line detection
    # Using HoughLinesP which directly gives line segments
    # Parameters for HoughLinesP:
    # rho: Distance resolution of the accumulator in pixels.
    # theta: Angle resolution of the accumulator in radians.
    # threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes.
    # minLineLength: Minimum line length. Line segments shorter than this are rejected.
    # maxLineGap: Maximum allowed gap between points on the same line to link them.
    # Reverting to parameters from Hough Attempt 4 / Canny(30,90) which gave best Y-match
    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=10, maxLineGap=20)

    long_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            # Step 6: Implement line filtering
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > min_length:
                long_lines.append(((x1, y1, x2, y2)))

    return long_lines

# Step 7: Implement main function
if __name__ == "__main__":
    test_image_path = "data/gpk-37a.png" # Make sure this path is correct

    # First, check if the image file exists
    # For now, we'll assume it exists or cv2.imread will handle the error.
    # A more robust check could be added here using os.path.exists

    print(f"Attempting to load image from: {test_image_path}")

    # List files in data directory to help debug path issues
    # This is a placeholder for a potential debugging step if image loading fails.
    # For now, we'll proceed directly to calling the function.

    detected_lines = find_long_lines(test_image_path)

    if not detected_lines:
        print("No long lines found or error in processing.")
    else:
        print(f"Found {len(detected_lines)} lines longer than 800 pixels:")
        for i, line in enumerate(detected_lines):
            print(f"Line {i+1}: Start({line[0]}, {line[1]}), End({line[2]}, {line[3]})")

    # Optional: To visually verify, you could draw these lines on the original image
    # and display it or save it.
    img_display = cv2.imread(test_image_path)
    if img_display is not None and detected_lines:
        for line_coords in detected_lines:
            x1, y1, x2, y2 = line_coords
            cv2.line(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw red lines
        cv2.imwrite("data/lines_detected.png", img_display)
        print("Saved image with detected lines to data/lines_detected.png")
    else:
        if img_display is None:
            print("Could not load image for visual verification.")
        elif not detected_lines:
            print("No lines detected, so no image for visual verification created.")
        else:
            print("Could not create output image for visual verification for an unknown reason.")

print("Script line_detector.py created and populated.")
