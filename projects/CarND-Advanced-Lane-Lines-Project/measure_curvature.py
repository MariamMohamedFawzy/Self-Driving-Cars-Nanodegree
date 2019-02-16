import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# def generate_data(left_fit, right_fit, ym_per_pix, xm_per_pix):
#     ploty = np.linspace(0, 719, num=720)

#     left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

#     left_fitx = left_fitx[::-1]
#     right_fitx = right_fitx[::-1]

#     left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
#     right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

#     return left_fit_cr, right_fit_cr


# def measure_curvature_real(left_fitx, right_fitx):
#     '''
#     Calculates the curvature of polynomial functions in meters.
#     '''
#     # Define conversions in x and y from pixels space to meters
#     ym_per_pix = 30/720 # meters per pixel in y dimension
#     xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
#     # Start by generating our fake example data
#     # Make sure to feed in your real data instead in your project!
# #     ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
#     left_fit_cr, right_fit_cr = generate_data(left_fit, right_fit, ym_per_pix, xm_per_pix)

#     # left_fit_cr = left_fit_cr[::-1]    
#     # right_fit_cr = right_fit_cr[::-1] 

#     ploty = np.linspace(0, 719, num=720)
    
#     # Define y-value where we want radius of curvature
#     # We'll choose the maximum y-value, corresponding to the bottom of the image
#     y_eval = np.max(ploty)
    
#     # Calculation of R_curve (radius of curvature)
#     left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
#     right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
#     return left_curverad, right_curverad


def measure_curvature_real (leftx, rightx):
    
    leftx = leftx[::-1]
    rightx = rightx[::-1]

    xm_per_pix=3.7/700
    ym_per_pix = 30/720
    
    ploty = np.linspace(0, 719, 720)

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_curverad, right_curverad)
