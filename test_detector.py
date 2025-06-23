import cv2
import os
from card_detector import CardDetector

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
    main()
