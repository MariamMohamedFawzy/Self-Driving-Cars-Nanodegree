# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/sample_crop.png "Images Cropped"
[image3]: ./images/train_history.png "Training History"




# Data

### Data Collection
The data was collected using term 1 simulator for 6 laps as follows: <br />
* 4 laps where the car is at the center of the road.
* 1 lap to handle the deviations of the road.
* The last one that focuses only on the curves.

### Preprocessing
The processing of the data is embedded in the model architecture. It is mainly two steps:
1. Converting the RGB images to YUV
2. Normalization
3. Cropping the upper and lower parts of the image that are not part of the road and not necessary.

### Data Augmentation
Since the size of the data is small, we augment the data using two techniques:
1. Using the left and righ camera images and aplying a correction factor = 0.4 to the angle
2. Flipping the images from left to right and using the negative of the steering angle

### Data Generator

A data generator is used to be able to load only the needed images into memory not the whole dataset.

### Train Validation Split

The data is splitted into two sets: training set and validation set.<br>
The validation set consists of 0.2 of the whole dataset.

# Model Architecture

The model design is similar to Nvidia paper: End to End Learning for Self-Driving Cars<br>
However, there are some differences:
* Batch normalization is used before every convolution layer and fully connected layer except the last one to reduce overfitting and make the model converge faster.
* Dropout is applied before the fully connected layers.
* Tanh activation function is used as the steering angle is between 1 and -1.
* The first fully connected layer has 100 nodes instead of >1000 to avoid overfitting.

Layer (type)         |        Output Shape     |       #  Param  
=================================================================
cropping2d_2 (Cropping2D)  |  (None, 90, 320, 3)   |     0         

color_space_layer_1 |(ColorSp (None, 90, 320, 3)     |   0         

normalization_layer_1 |(Norma (None, 90, 320, 3)     |   0         

conv2d_1 (Conv2D)       |     (None, 43, 158, 24)   |    1824      

leaky_re_lu_1 (LeakyReLU)  |  (None, 43, 158, 24)   |    0         

batch_normalization_1 | (Batch (None, 43, 158, 24)   |    96        

dropout_1 (Dropout)      |    (None, 43, 158, 24)   |    0         

conv2d_2 (Conv2D)         |   (None, 20, 77, 36)    |    21636     

leaky_re_lu_2 (LeakyReLU)  |  (None, 20, 77, 36)    |    0         

batch_normalization_2 | (Batch (None, 20, 77, 36)    |    144       

dropout_2 (Dropout)   |       (None, 20, 77, 36)    |    0         

conv2d_3 (Conv2D)     |       (None, 18, 75, 48)     |   15600     

leaky_re_lu_3 (LeakyReLU) |   (None, 18, 75, 48)    |    0         

batch_normalization_3 | (Batch (None, 18, 75, 48)    |    192       

dropout_3 (Dropout)     |     (None, 18, 75, 48)    |    0         

conv2d_4 (Conv2D)      |      (None, 16, 73, 64)   |     27712     

leaky_re_lu_4 (LeakyReLU)  |  (None, 16, 73, 64)    |    0         

batch_normalization_4 | (Batch (None, 16, 73, 64)    |    256       

dropout_4 (Dropout)   |       (None, 16, 73, 64)     |   0         

conv2d_5 (Conv2D)     |       (None, 14, 71, 64)     |   36928     

leaky_re_lu_5 (LeakyReLU) |   (None, 14, 71, 64)     |   0         

flatten_1 (Flatten)   |       (None, 63616)      |       0         

batch_normalization_5 | (Batch (None, 63616)      |       254464    

dropout_5 (Dropout)   |       (None, 63616)       |      0         

dense_1 (Dense)       |       (None, 100)         |      6361700   

leaky_re_lu_6 (LeakyReLU) |   (None, 100)        |       0         

batch_normalization_6 | (Batch (None, 100)         |      400       

dropout_6 (Dropout)    |      (None, 100)           |    0         

dense_2 (Dense)      |        (None, 50)             |   5050      

leaky_re_lu_7 (LeakyReLU)  |   (None, 50)          |      0         

batch_normalization_7 | (Batch (None, 50)        |        200       

dropout_7 (Dropout)    |      (None, 50)        |        0         

dense_3 (Dense)      |        (None, 10)         |       510       

leaky_re_lu_8 (LeakyReLU)  |  (None, 10)        |        0         

dense_4 (Dense)      |        (None, 1)        |         11        


![alt text][image1]

## Overfitting

As mentioned above, the overfitting is handled by using dropout, batch normalization, data augmentation, simple architecture.

## Training

The model is trained using Adam optimizer with learning rate = 1e-3 (code line 157), and the learning rate is decreases if the validation loss increased for one epoch (code line 165).<br>
I used also early stopping to stop training the model if the validation loss increases for 2 epochs (code line 164).<br>
The model was trained using batch size = 32 before augmentation for 10 epochs using early stopping.

![alt text][image3]

## How to choose the correction factor

This is not in the [model.py](./model.py) file but in [Colab](https://colab.research.google.com/drive/1cYGKCxJ4nH4ssoa2R48-MFKQhcziHmkv#scrollTo=WFra-c1CRxXo
) <br>

This can be summarized as following:
1. I trained the model only on the images from the cener camera to predict the steering angle
2. After that, I used transfer learning to build another model. The old model is frozen and used in the new model. The input is the image (left or right) and a float number [flag] (either 1 or 0) to tell if the image is a left camera image or right camera image.
3. I then concatenate the output of the old model and the flag input and add another fuly connected layer with 1 node.
4. The new model should be given a left/right images with a flag 1/0 (left / right) and output the steering angle as the center camera image.
5. I then explored the weights of the last layer:
<br>
let y<sub>1</sub> is the output of the old model, y<sub>2</sub> is the output of the new model, (w<sub>1</sub>,w<sub>2</sub>, b) the weights of the last layer.
<br>
w<sub>1</sub> * y<sub>1</sub> + w<sub>1</sub> * flag + b = y<sub>2</sub>
<br>
w<sub>1</sub> ~ 0.634 <br>
w<sub>2</sub> ~ -0.034 <br>
b ~ -0.009

This is not an correct number since the models may be overfitting the data or not trained enough, but it gives me an intuition where to first pick a value for the corection factor.<br>
Therefore, I used o.5, but it was not a good one, 0.4 gives me good results.

# Testing

The model is tested on the simulator to make sure that the car does not go off the road or is flipped.



