
import cv2
import numpy as np
import imutils

from algos.card_utils import is_valid_card

def detect_card_from_white_border(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print ('approx',len(approx),peri)
        if len(approx) == 4 and is_valid_card(approx):
            return approx
    return None

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image
