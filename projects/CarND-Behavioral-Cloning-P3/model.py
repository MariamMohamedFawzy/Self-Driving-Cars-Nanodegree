# Code on colab: 
# https://colab.research.google.com/drive/1cYGKCxJ4nH4ssoa2R48-MFKQhcziHmkv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras

import tensorflow as tf

import os
import csv

import sklearn
from sklearn.model_selection import train_test_split

from random import shuffle

import cv2

import numpy as np

from skimage.color import rgb2yuv

import random


# Read the images path
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      samples.append(line)

# Generator to generate the data in batches
# Images are read and augmented in the following manner:
# 1- Use left and right camera images with correction = 0.5
# 2- flip the three images and the new angle will be old angle * -1
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-1.0*center_angle)
                
                correction = 0.4
                
                name = './IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = center_angle + correction
                
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-1.0*left_angle)
                
                name = './IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = center_angle - correction
                
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(right_image))
                angles.append(-1.0*right_angle)

            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            
            yield sklearn.utils.shuffle(X_train, y_train)

# split the data to 80 : 20
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# using the generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Custom Layer to normalize the data
class NormalizationLayer(keras.layers.Layer):
  def call(self, x):
    return x / 127.0 - 1.0

# Custom layer to convert the RGB to YUV
class ColorSpaceLayer(keras.layers.Layer):
  def call(self, x):
    return tf.image.rgb_to_yuv(x)

# Model Architecture

input_shape=(160, 320, 3)

model = keras.models.Sequential()
model.add(keras.layers.Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))

model.add(ColorSpaceLayer())

model.add(NormalizationLayer())

model.add(keras.layers.Conv2D(24, (5, 5), strides=2))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(36, (5, 5), strides=2))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(48, (3, 3)))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.Flatten())

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(50))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.LeakyReLU(0.2))

model.add(keras.layers.Dense(1, activation='tanh'))



opt = keras.optimizers.Adam(0.003)
model.compile(opt, loss=keras.losses.mean_squared_error, metrics=['accuracy'])

# Callbacks to store the history, 
# decrease learning rate if validation loss increases for 1 epoch,
# stop if validation loss increases for 2 epochs
history = keras.callbacks.History()
earlystop = keras.callbacks.EarlyStopping(patience=2)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.00001)

ckpt = keras.callbacks.ModelCheckpoint('model_sdc_tanh_.h5', monitor='val_loss')

# Training
model.fit_generator(train_generator, steps_per_epoch=\
                   len(train_samples) // 32, validation_data=validation_generator,\
                   validation_steps= len(validation_samples)//32, epochs=20,\
                   callbacks=[history, earlystop, reduce_lr, ckpt])




