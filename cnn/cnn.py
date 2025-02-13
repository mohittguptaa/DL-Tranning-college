import tensorflow as tf
import numpy as np
from keras.api.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.api.models import Sequential
from keras.src.legacy.preprocessing.image import ImageDataGenerator

cnn=Sequential()
cnn.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(16,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
e=cnn.add(Flatten())

cnn.add(Dense(64,activation='relu'))
cnn.add(Dense(32,activation='relu'))
cnn.add(Dense(16,activation='relu'))
cnn.add(Dense(8,activation='relu'))
cnn.add(Dense(4,activation='relu'))
cnn.add(Dense(1,activation='sigmoid'))

c=cnn.compile(loss='binary_crossentropy',optimizer='adam')
# print(c)

ImageDataGenerator(rescale=1.0/255)
test_datagen=ImageDataGenerator(rescale=1.0/255,
                                shear_range=0.2,
                                zoom_range=.2,
                                horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1.0/255)
train_generator=test_datagen.flow_from_directory('data/training_set/',
                                                 target_size=(64,64),
                                                  batch_size=32,
                                                  class_mode='binary')
test_generator=test_datagen.flow_from_directory('data/test_set/',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')