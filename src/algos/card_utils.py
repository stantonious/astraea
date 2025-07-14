
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left
    return rect

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