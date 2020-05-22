import pickle

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import ear_classifier_model
import os


img_width, img_height = 100, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/val'
nb_train_samples = 44000
nb_validation_samples =4000
log_dir='logs/'
epochs = 8000
batch_size = 16
# the augmentation configuration  for training
train_datagen = ImageDataGenerator(
   rescale=1. / 255,
  shear_range=0.2,
   zoom_range=0.2
)

# augmentation configuration for testing
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
ear_classifier_model.ear_model(img_width,img_height,log_dir,epochs,train_generator,validation_generator,nb_train_samples,nb_validation_samples,batch_size)

