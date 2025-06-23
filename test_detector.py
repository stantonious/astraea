import cv2
import os
import unittest
import numpy as np
from card_detector import CardDetector

# Ground truth for gpk-37a.png (x, y)
# Top-left: (50, 28)
# Top-right: (1118, 18)
# Bottom-right: (1138, 1528)
# Bottom-left: (46, 1534)
GROUND_TRUTH_GPK_37A = {
    "top_left": (50, 28),
    "top_right": (1118, 18),
    "bottom_right": (1138, 1528),
    "bottom_left": (46, 1534),
}

class TestCardDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CardDetector()
        self.test_image_path = "data/gpk-37a.png"
        self.gpk_37a_ground_truth = np.array([
            GROUND_TRUTH_GPK_37A["top_left"],
            GROUND_TRUTH_GPK_37A["top_right"],
            GROUND_TRUTH_GPK_37A["bottom_right"],
            GROUND_TRUTH_GPK_37A["bottom_left"]
        ], dtype="float32")

    def assertCornersClose(self, detected_corners, ground_truth_corners, tolerance=4):
        self.assertIsNotNone(detected_corners, "Corners were not detected.")
        self.assertEqual(detected_corners.shape, ground_truth_corners.shape, "Shape of detected corners does not match ground truth.")
        for i in range(len(ground_truth_corners)):
            # Compare point by point
            det_pt = detected_corners[i]
            gt_pt = ground_truth_corners[i]
            self.assertAlmostEqual(det_pt[0], gt_pt[0], delta=tolerance,
                                   msg=f"Corner {i} X-coordinate mismatch: detected {det_pt}, ground truth {gt_pt}")
            self.assertAlmostEqual(det_pt[1], gt_pt[1], delta=tolerance,
                                   msg=f"Corner {i} Y-coordinate mismatch: detected {det_pt}, ground truth {gt_pt}")

    def test_find_card_edges_gpk_37a(self):
        """Tests card edge detection for data/gpk-37a.png against ground truth."""
        print(f"\nRunning test_find_card_edges_gpk_37a with image: {self.test_image_path}")
        original_image, detected_edges = self.detector.find_card_edges(self.test_image_path)

        self.assertIsNotNone(original_image, "Original image could not be loaded.")
        if detected_edges is None:
            # Save the image with no detection for debugging
            cv2.imwrite("output/failed_detection_gpk_37a.png", original_image)
            print("Test failed: No card edges detected. Saved original image to output/failed_detection_gpk_37a.png")
        else:
            print(f"Detected edges for gpk-37a: {detected_edges}")
            # Save the image with detected edges for visual inspection
            debug_image = self.detector.draw_edges(original_image.copy(), detected_edges)
            # Ensure output directory exists
            if not os.path.exists("output"):
                os.makedirs("output")
            cv2.imwrite("output/detected_edges_gpk_37a.png", debug_image)
            print("Saved image with detected edges to output/detected_edges_gpk_37a.png")

        self.assertCornersClose(detected_edges, self.gpk_37a_ground_truth, tolerance=4)

def main():
    # Path to the directory containing test images
    # Path to the directory containing test images
    data_dir = "data/"
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    # image_files = ["data/gpk-20b.webp"] # Test gpk-20b.webp with new settings

    if not image_files:
        # This case should not be reached if we are hardcoding the file.
        print(f"No images found. Please add some test images.")
        return

    # Initialize the card detector
    # We can tune parameters here if needed, e.g.:
    # detector = CardDetector(gaussian_blur_kernel_size=(7,7), canny_threshold1=40, canny_threshold2=120)
    detector = CardDetector()

    for image_path in image_files:
        print(f"Processing image: {image_path}")
        try:
            original_image, card_edges = detector.find_card_edges(image_path)

            if original_image is None:
                print(f"Could not load image {image_path}")
                continue

            if card_edges is not None:
                print(f"Card detected. Corners: {card_edges}")
                output_image = detector.draw_edges(original_image.copy(), card_edges)
            else:
                print("No card detected.")
                output_image = original_image.copy() # Show original if no detection

            # Resize for display if too large (optional)
            max_height = 800
            if output_image.shape[0] > max_height:
                scale_factor = max_height / output_image.shape[0]
                output_image = cv2.resize(output_image, None, fx=scale_factor, fy=scale_factor)

            # cv2.imshow(f"Card Detection Result - {os.path.basename(image_path)}", output_image)

            # Save the output image
            output_filename = os.path.join("output", f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_filename, output_image)
            print(f"Saved result to {output_filename}")

            # key = cv2.waitKey(0) # Wait indefinitely for a key press
            # if key == 27: # ESC key to quit
            #     print("ESC key pressed, exiting...")
            #     break
            # cv2.destroyWindow(f"Card Detection Result - {os.path.basename(image_path)}")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

    cv2.destroyAllWindows()
    print("Test script finished.")


if __name__ == "__main__":
    # main() # Comment out direct execution if running as unittest
    unittest.main()
