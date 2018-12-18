
# **Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./examples/ex1.png "Edges"
---

## Pipeline
My pipeline consists of 6 steps summarized as follows :
* Resizing the image if it does not match the size I am working on which is (540, 960). In this step I created a function called `resize_image` that returns a new image of the required size.
* Changing the illumination of the image to be able to work on images captured in light and shadows. I used a function called `adjust_gamma` from [stackoverflow](https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351).
* Converting the image to a grayscale image.
* Bluring the image to remove noise to be able to detect clear edges
* Appling canny edge detector
* Applying hough transform to filter the edges.

---

## Documentation of the `draw_lines` function
* First I define the vertices of the area where I most likey will find the lane.
* Then I calculate the averge x and y values in each side (left, right) and slope. I neglect the lines (x, y) with infinity or zero or near-zero slope to ignore the horizontal (or semi-horizontal) lines.
* After that, I calculate the average slope in each side as follows :
    * Then, I check if the global slope variable is None, then I will assign the average slope to the global variable.
    * If the global slope is not None, the slope in each side will be `learning rate * slope + (1-learning rate) * global slope`. Learning rate is 0.2. It ensures the robustness of the slope not to be affected by outliers.
* After these steps, I get the slope and an averge point. I then calculate the interstion points of this line with the bottom line and another middle line.
* Finally, I draw the lines.

---

# Output

## Mask image
![alt text][image1]

## Video with edges
[![Edges](https://img.youtube.com/vi/hy5Pee8-mZs/0.jpg)](https://youtu.be/hy5Pee8-mZs)

## Video with the edges converted to one solid line on each side
[![Solid lines](https://img.youtube.com/vi/-oosGehhrd8/0.jpg)](https://youtu.be/-oosGehhrd8)

## The challenging video where the images' size is different, the illumination changes during the video and there are horizontal edges with near-zero slope
[![Challenge Video](https://img.youtube.com/vi/70jqpiYk9to/0.jpg)](https://youtu.be/70jqpiYk9to)



----


# Potential shortcomings with the current pipeline

* The Pipeline depend on the hardcoded area that I make to search for images in. If the view changes or the position of the lane changes, this pipeline won't be able to detect the lane.
* The lines that represent the lane are slighlty shaking.
---

# Possible improvements to the pipeline

* One possible improvement is to use another pipelne that does not depend on the hardcoded mask but detect that mask dynamicly based on the view.

