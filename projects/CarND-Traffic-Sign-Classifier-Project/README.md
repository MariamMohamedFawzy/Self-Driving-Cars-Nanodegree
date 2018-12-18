# **Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./hist.png "Histogram"
[image2]: ./dataaug.png "Data Augmentation"
[image3]: ./modelarc.png "Model Architecture"
[image4]: ./newimgs.png "New Images"
[image5]: ./sftmx.png "Softmax"


The steps of this project are the following:
* Data Loading
* Data Augmentation
* Model Architecture, Training, Oversampling.
* Testing

-------

## Data Loading

I used the pandas library to load the dataset and calculate a summary of the dataset.

* Shape of training features is (34799, 32, 32, 3)
* Shape of training labels is (34799,)

* Shape of validation features is (4410, 32, 32, 3)
* Shape of validation labels is (4410,)

* Shape of testing features is (12630, 32, 32, 3)
* Shape of testing labels is (12630,)

![Histogram of classes in training and validation set][image1]

---------

## Data Augmentation

* The RGB images are converted to grayscale
* The dataset is augmented with the same images after these two steps:
    * The images are rotated by angle from -20 to 20
    * Noise of black pixels is added to the images with probability = 0.05


This should increase the size of the dataset and make the model more robust to images with variations that do not exist in the original dataset. <br />
This also acts as regularization to the model.

![Data augmentation][image2]

-------

## Model

* The model is a deep convolutional neural networks with inception module with resnet connection as the last convolutional layer.
* The dropout is used with probability=0.3 between the convolution layers and probability=0.5 between the fully connected layers to avoid overfitting.
* The filters of the convolution start with 64 and increase to 128.
* The model is trained with decreasing learning rate if the validation loss does not improve for 2 epochs and the model stops training if the validation loss does not improve for 4 epochs.

<!-- ![Model Architecture][image3] -->

<img src="modelarc.png" alt="Model Architecture" style="width:200px;" />

----

### Oversampling
* The data is unbalanced and there are some similar classes where one of them has more data.
* I did oversampling to increase the size of the classes with less data.
* Oversampling does not improve the model accuracy on the validation data, so it is ignored in the final results.

------

## Testing

### Validation set and Testing set performance

* Model accuracy of validation set = 98.75 %
* Model accuracy of testing set = 97.62 %

### Testing on new images from the web

* The model have been tested on 11 images from the web.

![New Images][image4]

* Model accuracy of the new images = 72.73 %

* we found that the model is not able to differentiate well between the speeds (numvbers). This may be because:
    * The classes is similar 
    * There are classes with fewer samples than other classes
    * The images are blurry


### However, the model erformance may be improved by pretraining on another dataset, such as MNIST dataset.


### Softmax analysis

* The model is not pretty sure about the wrong prediction except in the speed of 30 that was predicted as 80. <br />
This may be because the two classes are very similar
* The pedestrian class and the right-of-way at the next interestion class are similar, so when the model is wrong, it was not sure about the prediction.


![Softmax Analysis][image5]

