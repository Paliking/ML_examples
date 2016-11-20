
# source: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
# v tutoriali je par chyb
# 18.11.2016



import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer


def target_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data_examples', 'IdentifyDigits')
sub_dir = os.path.join(root_dir, 'sub')
# check for existence
print(root_dir)
print(data_dir)
print(sub_dir)
print(os.path.exists(root_dir))
print(os.path.exists(data_dir))
print(os.path.exists(sub_dir))
# load data
train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))
print(train.head())

# # choose random inage
# img_name = rng.choice(train.filename)
# filepath = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
# show image
# img = imread(filepath, flatten=True)
# # plt.imshow(img, cmap='gray')
# # plt.axis('off')
# # plt.show()

# print('image shape:', img.shape)

# For easier data manipulation, let’s store all our images as numpy arrays
temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32').reshape(-1)
    temp.append(img)
    
train_x = np.stack(temp)
train_x /= 255.0
# train_x = train_x.reshape(1,-1).astype('float32')
train_x = train_x.astype('float32')

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32').reshape(-1)
    temp.append(img)
    
test_x = np.stack(temp)
test_x /= 255.0
# test_x = test_x.reshape(1,-1).astype('float32')
test_x = test_x.astype('float32')


# one hot encoding of target
train_y = keras.utils.np_utils.to_categorical(train.label.values)

# split training set for train and validation 
split_size = int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
# train_y, val_y = train.label[:split_size], train.label[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]
print('train_x shape:', train_x.shape)
print('val_x shape:', val_x.shape)
print('train_y shape:', train_y.shape)
print('val_y shape:', val_y.shape)

# train_y = target_to_one_hot(train_y)
# val_y = target_to_one_hot(val_y)
# ---------------- model building---------------------------------
# define vars
input_num_units = 28*28 # image size is 28*28
hidden_num_units = 500 # #neurons in hidden layer
output_num_units = 10 # we have 10 digits

epochs = 5
batch_size = 128

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential([
  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
  Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model training
# trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))





# # -----test model with our own eyes-----
# # predict all test images
# pred = model.predict_classes(test_x)

# img_name = rng.choice(test.filename)
# filepath = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)

# img = imread(filepath, flatten=True)

# test_index = int(img_name.split('.')[0]) - train.shape[0]

# print ("Prediction is: ", pred[test_index])

# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()





# ------------------------------ deep model instead wide-----------------------------------
# define vars
input_num_units = 784
hidden1_num_units = 50
hidden2_num_units = 50
hidden3_num_units = 50
hidden4_num_units = 50
hidden5_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),

Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
 ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_5d = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))
# Looks worse. This may be because our model may be overfitting. 
# To deal with this, we use a method called dropout.

# --------------------------- deep model with dropout-----------------------------------------------
# define vars
input_num_units = 784
hidden1_num_units = 50
hidden2_num_units = 50
hidden3_num_units = 50
hidden4_num_units = 50
hidden5_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

dropout_ratio = 0.2

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
 Dropout(dropout_ratio),

Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
 ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_5d_with_drop = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))
# It seems that our model is not performing well enough. One reason may be because we are 
# not training our model to its full potential. Increase our training epochs to 50 and check it out!

# ----------------------------- deep model with dropout and more epochs---------------------------
# define vars
input_num_units = 784
hidden1_num_units = 50
hidden2_num_units = 50
hidden3_num_units = 50
hidden4_num_units = 50
hidden5_num_units = 50
output_num_units = 10

epochs = 50
batch_size = 128
dropout_ratio = 0.2

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
 Dropout(dropout_ratio),

Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_5d_with_drop_more_epochs = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))
# We see an increase in accuracy.

# -------------------------------- deep and wide model ------------------------------
# define vars
input_num_units = 784
hidden1_num_units = 500
hidden2_num_units = 500
hidden3_num_units = 500
hidden4_num_units = 500
hidden5_num_units = 500
output_num_units = 10

epochs = 25
batch_size = 128
dropout_ratio = 0.2

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
 Dropout(dropout_ratio),

Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
 ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_deep_n_wide = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))
# # it is best until now!!!! as expected.

# make submission
# pred = model.predict_classes(test_x)
# sample_submission.filename = test.filename
# sample_submission.label = pred
# sample_submission.to_csv(os.path.join(data_dir, 'sub03.csv'), index=False)



# Until now we made multilayer perceptrons (MLP). Let’s now change it to a convolutional neural network.
# -------------------- convolutional neural network CNN ----------------
# reshape data

train_x_temp = train_x.reshape(-1, 28, 28, 1)
val_x_temp = val_x.reshape(-1, 28, 28, 1)
test_x_temp = test_x.reshape(-1, 28, 28, 1)

# define vars
input_shape = (784,)
input_reshape = (28, 28, 1)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 4, 4, activation='relu'),

 Flatten(),

 Dense(output_dim=hidden_num_units, activation='relu'),

 Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))

# make submission
# pred = model.predict_classes(test_x_temp)
# sample_submission.filename = test.filename
# sample_submission.label = pred
# sample_submission.to_csv(os.path.join(data_dir, 'subCNN.csv'), index=False)


# my experiment-------------------------------------
# reshape data
train_x_temp = train_x.reshape(-1, 28, 28, 1)
val_x_temp = val_x.reshape(-1, 28, 28, 1)
test_x_temp = test_x.reshape(-1, 28, 28, 1)

# define vars
input_shape = (784,)
input_reshape = (28, 28, 1)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden1_num_units = 500
hidden2_num_units = 500
output_num_units = 10

epochs = 10
batch_size = 128

model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 4, 4, activation='relu'),

 Flatten(),

 Dense(output_dim=hidden1_num_units, activation='relu'),

 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 
 Dense(output_dim=output_num_units, input_dim=hidden2_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv2 = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))

# make submission
pred = model.predict_classes(test_x_temp)
sample_submission.filename = test.filename
sample_submission.label = pred
sample_submission.to_csv(os.path.join(data_dir, 'subCNN2.csv'), index=False)