# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:01:59 2020

@author: Pranav, Bhargav Sagiraju (Crediting him for the batch validation & Code structuring)
"""

from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

imagesize = 150

train_batchsize = 20
val_batchsize = 20
train_dir = 'suddu_bro_dataset/train'
validation_dir = 'suddu_bro_dataset/test'

#Image Data Generators for Train and Validation
train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 20,
      zoom_range = 0.15,
      width_shift_range = 0.25,
      height_shift_range = 0.25,
      horizontal_flip = True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

#Load the images
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (imagesize, imagesize),
        batch_size = train_batchsize,
        class_mode = 'categorical',
        shuffle = True)
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size = (imagesize, imagesize),
        batch_size = val_batchsize,
        class_mode = 'categorical',
        shuffle = True)


path = 'keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = Xception(input_shape = (150, 150, 3), weights = path, include_top = False)

x = base_model.output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512,activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True
    
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr=1e-4),
              metrics=['acc'])

checkpoint = ModelCheckpoint('Xception.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

history = model.fit_generator(
      train_generator,
      steps_per_epoch = train_generator.samples/train_generator.batch_size ,
      epochs = 5,
      validation_data = validation_generator,
      validation_steps = validation_generator.samples/validation_generator.batch_size,
      verbose = 1,
      callbacks = callbacks_list)

#Accuracy & Loss graphs
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Xception | From Scratch - Accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Xception | From Scratch - Loss')
plt.show()

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size = (imagesize, imagesize),
        batch_size = val_batchsize,
        class_mode = 'categorical',
        shuffle = False)

import numpy as np
# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
  
'''
Total params: 74,866,730
Trainable params: 74,812,202
Non-trainable params: 54,528
'''