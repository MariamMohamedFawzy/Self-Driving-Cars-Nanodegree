import numpy as np

def get_shift(left_fit, right_fit):
    left_point = left_fit[0]*(720**2) + left_fit[1]*720 + left_fit[2]
    right_point = right_fit[0]*(720**2) + right_fit[1]*720 + right_fit[2]
    mid_point = right_point - left_point
    mid_frame = 1280//2
    shift = mid_frame - mid_point
    return np.round((shift * 2.6458333333333E-6) / 0.01, 2)

