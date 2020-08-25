import argparse
import base64
import json
import cv2

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from os import getcwd

from model import random_distort
from model_functions import *

if __name__ == '__main__':
    '''
    This little guy mostly takes bits from drive.py and model.py to help clean up some data, pulling the data points
    that generate the most erroneous predictions from the model and visualizing them (to make sure they're actually bad)
    so I can then edit the actual steering angle values in the csv file 
    '''
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    using_udacity_data = False
    img_path_prepend = ''
    csv_path = './training_data/driving_log.csv'
    if using_udacity_data:
        img_path_prepend = getcwd() + '/udacity_data/'
        csv_path = './udacity_data/driving_log.csv'

    import csv
    # Import driving data from csv
    with open(csv_path, newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    image_paths = []
    angles = []

    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend + row[0])
        angles.append(float(row[3]))

    image_paths = np.array(image_paths)
    angles = np.array(angles)

    print('shapes:', image_paths.shape, angles.shape)

    # visualize some predictions
    n = 12
    X_test,y_test = generate_training_data_for_visualization(image_paths[:n], angles[:n], batch_size=n,                                                                     validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    #visualize_dataset(X_test, y_test, y_pred)

    # get predictions on a larger batch - basically pull out worst predictions from each batch so they can be 
    # corrected manually in the csv
    n = 1000
    for i in reversed(range(len(image_paths)//n + 1)):
        start_i = i * n
        end_i = (i+1) * n
        batch_size = n
        if end_i > len(image_paths):
            end_i = len(image_paths)
            batch_size = end_i - start_i - 1
        X_test,y_test = generate_training_data_for_visualization(image_paths[start_i:end_i], 
                                                                 angles[start_i:end_i], 
                                                                 batch_size=batch_size,                                                                     
                                                                 validation_flag=True)
        y_pred = model.predict(X_test, n, verbose=2).reshape(-1,)
        # sort the diffs between predicted and actual, then take the bottom m indices
        m = 5
        bottom_m = np.argsort(abs(y_pred-y_test))[batch_size-m:]
        print('indices:', bottom_m+(i*n) + 1)
        print('actuals:', y_test[bottom_m])
        print('predictions:', y_pred[bottom_m])
        print('')
        visualize_dataset(X_test[bottom_m], y_test[bottom_m], y_pred[bottom_m])
                                                            