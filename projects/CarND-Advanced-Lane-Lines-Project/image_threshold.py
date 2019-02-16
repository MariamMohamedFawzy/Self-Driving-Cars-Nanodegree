import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color

 
def pipeline(img, s_thresh=(100, 255), l_thresh=(180, 255)):
    img = np.copy(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # l_channel = lab[:, :, 0]

    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # v_channel = hsv[:, :, 0]

    # # Sobel x
    # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    # abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # # Threshold x gradient
    # sxbinary = np.zeros_like(scaled_sobel)
    # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 255

    # v_binary = np.zeros_like(v_channel)
    # v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    
    # Stack each channel
    # color_binary = np.dstack(( s_binary, s_binary, s_binary)) * 255
    img = np.bitwise_or(l_binary, s_binary)
    color_binary = img
    return color_binary


def convert_thresh(image):
    result = pipeline(image)   
    # result = color.rgb2gray(result)
    return result


if __name__ == '__main__':

    images = glob.glob('test_images/*.jpg')

    for image_path in images:
        image = mpimg.imread(image_path)
        result = pipeline(image)   
        result = color.rgb2gray(result)
        plt.imsave('output_images/threshold_' + image_path, result, cmap=plt.get_cmap('gray'))