# Path Planning Project - Self-Driving Cars Nanodegree
   
## This project is divided into several parts:
1. Searching for the best lane/path.
2. Selecting the best control parameters.
3. Going through the selected path.

## 1. Searching for the best lane/path
The car can have three states:
* keep lane <br>
    1. if the cars in front of me in the same lane have a distance <b> >= 30 m </b>
* change lane
    1. if the cars in front of me in the same lane have a distance <b> < 30 m </b>
    2. the nearby lane (one-lane offset) has a car at a distance <b> greater </b> than the distance of the car in front of my car.
* change speed
   1. if the cars in front of me in the same lane have a distance <b> < 30 m </b>
    2. the nearby lane (one-lane offset) has a car at a distance <b> less </b> than the distance of the car in front of my car.


## 2. Selecting the best control parameters

* The speed should not exceed the max speed <b> 21.0 m/s </b> which is less than the maximum speed of the road by a sufficient margin.

* changing the speed is done by adding the <b> acceleration / 50 </b> to the speed at each point which is the maximum acceleration in 0.2 seconds, and should not exceed the maximum speed. Then, I keep track of the speed of the last point in the path, not the car speed that comes from the simulator because the car speed belongs to some point in the path not the last one.

* the acceleration of the speed is constant and equals <b> 5 m/s^2 </b> to account for the normal acceleration, so that the total acceleration does not exceed 10 m/s^2.


## 3. Going through the selected path
* I keep track of the points of the new path.

* First, I add the past points that the car did not go through to ensure that the path is smooth.

* Then, I add the points starting from d = the new lane * 4 + 2 and s = last s in the path + step size.

 * After that, I give spline the points to get some new points on the path but are smooth.


### Note 

The points have to be in car coordinate system so each time I give the car some points, I have to convert these points to be in car cooridnate system.