import math
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from os import getcwd
import csv

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model_functions import *

# Fix error with TF and Keras
import tensorflow as tf
#tf.disable_v2_behavior()


'''
Main program 
'''

# select data source(s) here
using_test_track = True
using_challenge_track = False

data_to_use = [using_test_track, using_challenge_track]
img_path_prepend = ['', getcwd() + '/udacity_data/']
csv_path = ['./training_data/driving_log.csv', './training_data_challenge/driving_log.csv']

image_paths = []
angles = []

for j in range(2):
    if not data_to_use[j]:
        # 0 = my own data, 1 = Udacity supplied data
        print('not using dataset ', j)
        continue
    # Import driving data from csv
    with open(csv_path[j], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    print(len(driving_data))

    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend[j] + row[0])
        angles.append(float(row[3]))
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[1])
        angles.append(float(row[3])+0.25)
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[2])
        angles.append(float(row[3])-0.25)

image_paths = np.array(image_paths)
angles = np.array(angles)

print('Before:', image_paths.shape, angles.shape)

# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.savefig('./report_images/distribution_steering_angle.jpg')
#plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
# plt.clf()
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.savefig('./report_images/distribution_steering_angle_plus.jpg')
#plt.show()

print('After:', image_paths.shape, angles.shape)

# visualize a single batch of the data
X,y = generate_training_data_for_visualization(image_paths, angles)
# visualize_dataset(X,y)

# split into train/test sets
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.05, random_state=42)
print('Train:', image_paths_train.shape, angles_train.shape)
print('Test:', image_paths_test.shape, angles_test.shape)

###### ConvNet Definintion ######

# for debugging purposes - don't want to mess with the model if just checkin' the data
just_checkin_the_data = False

if not just_checkin_the_data:
    model = Sequential()
    kernel_size = (5, 5)
    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66,200,3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(36, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(48, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dropout(0.10))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.10))
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.10))
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.10))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model, 
    #model.compile('adam', 'mean_squared_error')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    ############  just for tweaking model ##############
    # pulling out 128 random samples and training just on them, to make sure the model is capable of overfitting
    # indices_train = np.random.randint(0, len(image_paths_train), 128)
    # indices_test = np.random.randint(0, len(image_paths_test), 12)
    # image_paths_train = image_paths_train[indices_train]
    # angles_train = angles_train[indices_train]
    # image_paths_test = image_paths_test[indices_test]
    # angles_test = angles_test[indices_test]
    #############################################################

    # initialize generators
    train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
    val_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
    test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    #history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2, verbose=2)
    history = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=2560, steps_per_epoch=23040,
                                  epochs=5, verbose=2, callbacks=[checkpoint])

    print('Test Loss:', model.evaluate_generator(test_gen, 128))

    print(model.summary())

    # visualize some predictions
    n = 12
    X_test,y_test = generate_training_data_for_visualization(image_paths_test[:n], angles_test[:n], batch_size=n,                                                                     validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    visualize_dataset(X_test, y_test, y_pred)

    # Save model data
    model.save_weights('./model.h5')
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)