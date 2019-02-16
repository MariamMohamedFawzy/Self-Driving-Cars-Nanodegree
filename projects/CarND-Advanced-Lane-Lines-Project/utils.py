import numpy as np
import cv2
from skimage import color

def get_shift(left_fitx, right_fitx):

    left_point = np.average(left_fitx)
    right_point = np.average(right_fitx)

    # left_point = left_fit[0]*(720**2) + left_fit[1]*720 + left_fit[2]
    # right_point = right_fit[0]*(720**2) + right_fit[1]*720 + right_fit[2]
    mid_point = (right_point - left_point) / 2 + left_point 
    mid_frame = 1280//2
    shift = mid_frame - mid_point
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return np.round(shift * xm_per_pix, 2)


# def preprocess(image):
#     # image = color.rgb2gray(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(image)
#     return cl1
