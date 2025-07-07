
import cv2
import numpy as np
import imutils

def preprocess_white_border(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    #image = imutils.resize(image, width=1000)

    # Convert to LAB and apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Threshold lightness to isolate white border
    _, mask = cv2.threshold(l_eq, 190, 255, cv2.THRESH_BINARY)

    # Morph cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)

    # Find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cv2.drawContours(lab, contours, -1, (0, 255, 0), 1)
    # Display the image with drawn contours
    cv2.imshow("contours", lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    return image, mask, contours, orig

def is_valid_card(approx):
    pts = approx.reshape(4, 2)
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[0] - pts[3])
    h2 = np.linalg.norm(pts[1] - pts[2])
    width = (w1 + w2) / 2.0
    height = (h1 + h2) / 2.0
    aspect_ratio = min([width,height]) / max([height,width])
    print ('check w/h',width,height,aspect_ratio)
    return 0.70 < aspect_ratio < 0.74  # OS1 cards ~0.714

def detect_card_from_white_border(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print ('approx',len(approx),peri)
        if len(approx) == 4 and is_valid_card(approx):
            return approx
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left
    return rect

def warp_card(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    rect[0] -= 10
    rect[2] += 10
    print ('rect',rect)


    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    #print ('w/h',pts,maxWidth,maxHeight)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image

def analyze_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def main(image_path):
    print("[INFO] Loading image and isolating white border...")
    image, white_mask, contours, original = preprocess_white_border(image_path)

    print("[INFO] Finding OS1 card shape...")
    card_contour = detect_card_from_white_border(contours)
    if card_contour is None:
        print("❌ Could not find valid OS1 card contour.")
        return

    print("[INFO] Warping perspective...")
    warped = warp_card(original, card_contour)

    print("[INFO] Detecting corners...")
    corners_img = detect_corners(warped.copy())

    sharpness = analyze_sharpness(warped)
    print(f"🧠 Sharpness Score (Laplacian variance): {sharpness:.2f}")
    if sharpness < 100:
        print("⚠️ Potential corner dulling or edge wear detected.")

    # Show outputs
    cv2.imshow("White Border Mask", white_mask)
    cv2.imshow("Warped Card", warped)
    cv2.imshow("Corners Highlighted", corners_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 🔄 Update this path for your local image
main("/home/bstaley/git/astraea/data/gpk-20b.webp")
