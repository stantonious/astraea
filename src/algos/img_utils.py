

import cv2
import numpy as np
import imutils

from algos.card_utils import order_points

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