# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:17:25 2020

@author: 320001866
"""
 from keras.models import Sequential
 from keras.layers import Convolution2D
 from keras.layers import Conv2D

 from keras.layers import MaxPooling2D
 from keras.layers import Flatten
 from keras.layers import Dense
 
 
# Intilize the CNN
 classfier = Sequential()
 
#Step 1 - Convulation
 classfier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation ='relu' ))
 classfier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

#Step 2 
 classfier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3
 classfier.add(Flatten())
  
#Step 4
 classfier.add(Dense(units = 128, activation = 'relu'))
 classfier.add(Dense(units = 1, activation = 'sigmoid'))
 
 classfier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
 #part 2 - Fitting the CNN images -https://keras.io/preprocessing/image/
 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classfier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

classfier.get_weights


 
