# Controls PID Project

## Hyperparameters:
The P, I, D hyperparameters are tuned manually the same way as twiddle algorithm.
* First I select a value for P, while I and D = 0.
* I changed P such that the vehicle oscillates in an accepted way inside the trach.
* Then, I finetuned the D parameter with the selected P value and D = 0.
* D is selected such that the vehicle does not oscilate too much or does not change the steering angle suddenly. It should be smoother.
* Finally, I trained the I parameter. The i_error is set to zero for the first 100 values, then, it is set to the summation of the past 100 values not all the past values. <br>
I used this approach as it oscillates alot in the first few frames if I use it in the calculation in the first frames.<br>
The I parameter is selected such that the motion is smooth and averaged over the past values and the vehicle does not change its motion in a sudden way based on one unique value like when the vehicle moves from the bridge to the usual track.

## How to run the project:
* run `./run.sh` from the command line
* start the simulator and play the PID game
---

You will find a recording of the PID game on [Youtube](https://www.youtube.com/watch?v=GUS2Kgwj0-g)



