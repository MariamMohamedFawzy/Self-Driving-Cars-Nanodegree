import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def apply_mask(image):
    mask = np.zeros_like(image)
    # roi_corners = np.array([[(200,700), (500,500), (800, 430), (1200,700)]], dtype=np.int32)
    roi_corners = np.array([[(150,720), (500,500), (900, 500), (1200,720)]], dtype=np.int32)
    ignore_mask_color = (255,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # masked_image = cv2.bitwise_and(image, mask)
    masked_image = cv2.bitwise_and(image, mask) / 255.0

    # mask = np.ones_like(image)
    # roi_corners = np.array([[(500,720), (500,550), (800, 550), (900,720)]], dtype=np.int32)
    # ignore_mask_color = (0,)
    # cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # masked_image = cv2.bitwise_and(masked_image, mask)
    return masked_image


# def get_shrink(image):
#     new_image = np.zeros_like(image)
#     rows, cols, _ = image.shape
#     for x in range(rows):
#         for y in range(cols):
#             X = x - rows//2
#             Y= y - cols//2
#             new_image[X, Y, :] = image[x, y, :]
#     return new_image

def apply_perspective_transform(image):
    
    # src = np.float32([[550, 480], [300, 650], [730, 480], [1000, 650]])
    # dst = np.float32([[300, 0], [300, 720], [1000, 0], [1000, 720]])

    # # src = np.float32([[595, 450], [200, 690], [700, 450], [1000, 610]])
    # # dst = np.float32([[300, 0], [300, 920], [1000, 0], [1000, 920]])
    
    src = np.float32([[500, 500], [200, 700], [800, 500], [1100, 700]])
    dst = np.float32([[300, 0], [300, 720], [1000, 0], [1000, 720]])
    
    h, w = image.shape
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped, M


