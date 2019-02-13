import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def apply_mask(image):
    mask = np.zeros_like(image)
    roi_corners = np.array([[(200,700), (500,500), (650, 430), (1200,700)]], dtype=np.int32)
    ignore_mask_color = (1,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    mask = np.ones_like(image)
    roi_corners = np.array([[(500,700), (500,550), (800, 550), (900,650)]], dtype=np.int32)
    ignore_mask_color = (0,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(masked_image, mask)
    return masked_image


def apply_perspective_transform(image):
    
    src = np.float32([[550, 480], [300, 650], [730, 480], [1000, 650]])
    dst = np.float32([[300, 0], [300, 720], [1000, 0], [1000, 720]])

    # src = np.float32([[595, 450], [200, 690], [700, 450], [1000, 610]])
    # dst = np.float32([[300, 0], [300, 920], [1000, 0], [1000, 920]])
    
    
    h, w = image.shape
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped, M


