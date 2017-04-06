import csv
import cv2
import numpy as np

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_measurements = []

for line in lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    
    images.append(image)
    steering_measurements.append(float(line[3]))
    
    images.append(cv2.flip(image, 1))
    steering_measurements.append(float(line[3]) * -1.0)
    
X_train = np.array(images)
y_train = np.array(steering_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Convolution2D, Cropping2D

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1))) 
    model.add(Activation('relu'))

    model.add(Flatten())    

    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse") 
    
    return model

model = get_model()

model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save("../../model.h5")  

    

