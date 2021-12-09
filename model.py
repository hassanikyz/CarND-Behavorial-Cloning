import csv
import keras
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

import numpy as np

CROPTOP = 60
CROPBOTTOM = 25

def main():
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    count = 0 
    for line in lines:

        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = "./data/IMG/" + filename
            image = cv2.imread(local_path)
            images.append(image)

        correction = 0.2
  
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement + correction)
        measurements.append(measurement - correction)

    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * (-1.0)
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)
    
    X_train = np.array(images)
    y_train = np.array(measurements)

    model = Sequential()

    # normalization
    model.add(Lambda(lambda x: (x / 255 - 0.5),  input_shape=(160,320,3)))
    # crops at top and bottom, output shape = (75, 320, 3)
    model.add(Cropping2D(cropping=((CROPTOP,CROPBOTTOM), (0,0)), input_shape=(160,320,3)))
    
    # convolutional layers
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
#     model.add(MaxPooling2D())
#     model.add(Dropout(0.1))
    model.add(Convolution2D(64,3,3,activation="relu"))
    
    # flattening
    model.add(Flatten())
    
    # fully connected layers with dropouts
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(1))

#     model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
#     model.add(Cropping2D(cropping=((70,25),(0,0))))
#     model.add(Convolution2D(32,3,3,activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(Dropout(0.1))
#     model.add(Convolution2D(64,3,3,activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(Dropout(0.1))
#     model.add(Convolution2D(128,3,3, activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(Convolution2D(256,3,3, activation='relu'))
#     model.add(MaxPooling2D())

#     model.add(Flatten())
#     model.add(Dense(120))
#     model.add(Dense(20))
#     model.add(Dense(1))


    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    model.save('mymodel.h5')


if __name__ == '__main__':
    main()