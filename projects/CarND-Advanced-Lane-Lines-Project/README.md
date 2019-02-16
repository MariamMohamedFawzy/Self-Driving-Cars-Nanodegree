## Advanced Lane Finding Project

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./output_images/camera_cal/calibration1.jpg "Undistorted"
[image2]: ./test_images/test6.jpg "Road Transformed"
[image3]: ./image_thresh.jpg "Binary Example"
[image4]: ./image_warped.jpg "Warp Example"
[image5]: ./lane_detected.jpg "Fit Visual"
[image7]: ./output_images/result_final.png "Output"
[video1]: ./project_video_final.mp4 "Video"

---

### Camera Calibration

The code for this step is contained in the file called `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: <br>
**Distorted**
![alt text][image0]
**Undistorted**
![alt text][image1]

### Pipeline (single images)

#### 1. distortion Correction

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Color transforms and gradients.

I used a combination of l and s channels from HSL color space afther applying a threshold to generate a binary image (thresholding steps at lines 9 through 50 in `image_threshold.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Perspective transformation.

The code for my perspective transform includes a function called `apply_perspective_transform()`, which appears in lines 34 through 49 in the file `perspective_transform.py`.  The `apply_perspective_transform()` function takes as inputs an image.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[500, 500], 
    [200, 700], 
    [800, 500], 
    [1100, 700]])
dst = np.float32(
    [[300, 0], 
    [300, 720], 
    [1000, 0], 
    [1000, 720]])
    
```

Before applying the perspective transformation, I used a mask to get only the lane lines to help me in the next step. <br>


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Finding lane-line pixels and fit their positions with a polynomial.

Then I fit my lane lines with a 2nd order polynomial in `find_lane_pixels.py` kinda like this:

![alt text][image5]


I used the histogram to draw a window around the pixels where there is a lane then I fitted a 2nd order polynomial to these pixels.

After that, I used the fitted polynomial to get the fitted x and y values to be able to get the lane boundaries.

#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center

I did this in lines 52 through 69 in my code in `measure_curvature.py`.

<br>

The shift of the car from the center is calculated in `utils.py` in a function called `get_shift` that takes the left and right points and 
calculates the average point in the left and right side then calculates their average to be the midpoint. Then, I calculate the midpoint of the image and calculate the difference between the two mid points. 
#### 6. Final output

I implemented this step in my code in `solution.ipynb` in the function `process_frame(frame)`.  Here is an example of my result on a frame from the project video:

![alt text][image7]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_sol.mp4)
or on [Youtube](https://studio.youtube.com/video/PthTPkrIBLg/edit)

---

### Discussion

#### 1. The problems / issues I faced in your implementation of this project.

The main problem was how to pick the points of the source and destination of the calibration function to keep the two lines of the lane parallel. The src and dst points are hardcoded, so it is not the best solution. 
<br>
Therefore, I make a smoothing technique to handle this.
<br>
I used a Line class in `solution.ipynb` to keep track of the left and right lines.
<br>
I keep track of the past best 10 lines. The good fit is where:
* Most of the fitted x values of the right line do not exist in the left side of the image and vice versa.
* The curvature is not too small.
* The 2nd polynomial is not too big.
<br>

I then use the average of the past 10 lines.
<br>

However, This sometimes fails when frames become different and go away from the center or become too curved or some shape appears in the lane. So, There should be another stage before the lane detection. This new stage should categorize the frames and send each one to a specific pipeline that can handle the environment of that frame.
