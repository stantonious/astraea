# This file will contain the card detection logic.
import cv2
import numpy as np

class CardDetector:
    def __init__(self, gaussian_blur_kernel_size=(5, 5), canny_threshold1=50, canny_threshold2=150):
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
    def __init__(self, gaussian_blur_kernel_size=(5, 5), canny_threshold1=30, canny_threshold2=100): # Adjusted Canny
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        # Expected aspect ratio: width/height or height/width - Increased tolerance to 10%
        self.expected_aspect_ratio_range = ( (2.5 / 3.5) * 0.90, (2.5 / 3.5) * 1.10 )
        self.expected_aspect_ratio_range_inv = ( (3.5 / 2.5) * 0.90, (3.5 / 2.5) * 1.10 )
        self.min_contour_area_factor = 0.05 # Reduced min area factor

    def _load_and_preprocess_image(self, image_path):
        """Loads an image, converts to grayscale, and applies Gaussian blur."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, self.gaussian_blur_kernel_size, 0)
        return image, blurred_image

    def find_card_edges(self, image_path):
        original_image, preprocessed_image = self._load_and_preprocess_image(image_path)

        # Perform Canny edge detection
        edges_map = cv2.Canny(preprocessed_image, self.canny_threshold1, self.canny_threshold2)

        # Find contours
        contours, _ = cv2.findContours(edges_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (descending) and filter
        if not contours:
            return original_image, None # No contours found

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = original_image.shape[0] * original_image.shape[1]
        min_area = image_area * self.min_contour_area_factor

        card_contour = None
        for contour in contours:
            # Check minimum area
            if cv2.contourArea(contour) < min_area:
                break # Since contours are sorted, no need to check smaller ones

            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True) # Increased epsilon slightly

            # The card should be a quadrilateral
            if len(approx) == 4:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                is_valid_aspect_ratio = (self.expected_aspect_ratio_range[0] <= aspect_ratio <= self.expected_aspect_ratio_range[1]) or \
                                        (self.expected_aspect_ratio_range_inv[0] <= aspect_ratio <= self.expected_aspect_ratio_range_inv[1])

                if is_valid_aspect_ratio:
                    card_contour = approx
                    break # Found the card

        if card_contour is not None:
            # Reshape contour to be a list of points and order them
            points = card_contour.reshape(4, 2)
            ordered_points = self._order_points(points)
            return original_image, ordered_points
        else:
            return original_image, None

    def _order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def draw_edges(self, image, ordered_points, color=(0, 255, 0), thickness=2):
        """Draws the detected edges (a polygon from ordered points) on the image."""
        if ordered_points is None or len(ordered_points) != 4:
            return image # Return original image if no valid points

        # Ensure points are integers for drawing
        pts = np.array(ordered_points, dtype=np.int32)

        # Draw lines between the points
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, thickness) # TL to TR
        cv2.line(image, tuple(pts[1]), tuple(pts[2]), color, thickness) # TR to BR
        cv2.line(image, tuple(pts[2]), tuple(pts[3]), color, thickness) # BR to BL
        cv2.line(image, tuple(pts[3]), tuple(pts[0]), color, thickness) # BL to TL

        # Optionally, draw circles at corners
        for point in pts:
            cv2.circle(image, tuple(point), thickness * 2, color, -1)

        return image

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # detector = CardDetector()
    # try:
    #     # Replace with an actual image path from the data/ directory for testing
    #     # edges = detector.find_card_edges("data/gpk-16a.webp")
    #     print("CardDetector initialized. Preprocessing step would run if an image path was provided.")
    # except FileNotFoundError as e:
    #     print(e)
    print("Card Detector module")
