import os
import argparse
import json
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D

from sklearn.utils import shuffle
import random
from sklearn.cross_validation import train_test_split

from os import listdir
from os.path import isfile, join

import numpy as np
from scipy import misc

import matplotlib.pyplot as plt

from PIL import Image
import cv2

################
#Create NN model
################
def get_model(time_len=1):
  ch, row, col = 39, 160, 3 # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model


##################
#preprocess images
##################
def preprocess(data, y):
    
    #crop images
    for i in range(0, len(data)):
        data[i] = data[i][62:140, :]

    #downsample images - fast way
    for i in range(0, len(data)):
        image = data[i]
        data[i] = image[::2,::2].copy()

    #flip image
    for i in range(0, len(data)):
        data.append(cv2.flip(data[i], 1))
        y.append(-float(y[i]))

def main():
    print('loading data')
    ##########################
    #load filenames and angles
    ##########################
    X_train_center = []
    X_train_left = []
    X_train_right = []
    Y_train_center = []
    Y_train_left = []
    Y_train_right = []
    with open('../images/driving_log.csv', newline='') as csvfile:
        driving_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(driving_reader, None)  # skip headers
        for row in driving_reader:
            angle = float(row[3].strip())
            speed = float(row[6].strip())
            if abs(speed) < 0.1 : 
                continue
                #balance the dataset. Ignore many of the zero angle cases
            if angle == 0:
                if random.randint(0,9) < 8:
                    continue
            X_train_center.append(row[0])
            X_train_left.append(row[1])
            X_train_right.append(row[2])
            Y_train_center.append(angle)
            Y_train_left.append(max(angle + 0.25, -1))
            Y_train_right.append(min(angle - 0.25, 1))

############################
#Start loading center images
############################
    X_train = []
    for filename in X_train_center :
        X_train.append(misc.imread("../images/"+filename))

############################
#Start loading left images
############################
    for i in range(0, len(X_train_left)) :
        X_train.append(misc.imread("../images/"+X_train_left[i].strip()))
	    
############################
#Start loading right images
############################
    for i in range(0, len(X_train_right)) :
        X_train.append(misc.imread("../images/"+X_train_right[i].strip()))


#################
#sum all y trains
#################
    Y_train = Y_train_center + Y_train_left + Y_train_right

################
#preprocess
################
    print('preprocessing')
    preprocess(X_train, Y_train)


#####################
#load recovery images
#####################
    print('Start loading recovery images!')

    X_train_recovery= []
    Y_train_recovery= []
    with open('../cloning/labels.csv', newline='') as csvfile:
        driving_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in driving_reader:
            if len(row) == 0:
                continue
            filename = '../cloning/'+row[0]
            angle = float(row[1].strip())
            if os.path.isfile(filename) :
                X_train_recovery.append(misc.imread(filename))
                Y_train_recovery.append(angle)

###########################
#preprocess recovery images
###########################
    print('Preprocess recovery images!')
    preprocess(X_train_recovery, Y_train_recovery)

    for img in X_train_recovery:
        X_train.append(img)
    for angle in Y_train_recovery:
        Y_train.append(angle)

#################################
#create test & validation dataset
#################################
    print ('create test and validation')
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train_recovery, Y_train_recovery = shuffle(X_train_recovery, Y_train_recovery)

    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.04)

    X_train = np.array(X_train).astype(np.float)
    Y_train = np.array(Y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

#################################
#train NN and save
#################################
    batch_size = 150
    epochs = 12

    print ('create NN model')
    model = get_model()

    # Configures the learning process and metrics
    model.compile(metrics=['mean_squared_error'], optimizer='Nadam', loss='mean_squared_error')

    print ('train NN model')
    # Train the model
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(X_test, Y_test))

    print ('save NN model')
    # Save model architecture and weights
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    json_file.close()
    model.save('model.h5')

    print ('Done!')

    # Show summary of model
    model.summary()
if __name__ == "__main__":
    main()
