# https://www.kaggle.com/danielelton/digit-recognizer/keras-cnn-with-explanation

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.core import Dropout, Dense, Flatten, Activation
from keras.callbacks import EarlyStopping
import os

K.set_image_dim_ordering('th') #input shape: (channels, height, width)

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data_examples', 'DigitRecognizer')


train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
valid_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

x_train = train_df.drop(['label'], axis=1).values.astype('float32')
Y_train = train_df['label'].values.astype('float32')
x_valid = valid_df.values.astype('float32')

img_width, img_height = 28, 28

n_train = x_train.shape[0]
n_valid = x_valid.shape[0]

n_classes = 10

x_train = x_train.reshape(n_train,1,img_width,img_height)
x_valid = x_valid.reshape(n_valid,1,img_width,img_height)

x_train = x_train/255 #normalize from [0,255] to [0,1]
x_valid = x_valid/255

y_train = to_categorical(Y_train)



n_filters = 64
filter_size1 = 3
filter_size2 = 2
pool_size1 = 3
pool_size2 = 1
n_dense = 128

model = Sequential()

model.add(Convolution2D(n_filters, filter_size1, filter_size1, batch_input_shape=(None, 1, img_width, img_height), activation='relu', border_mode='valid'))

model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))

model.add(Convolution2D(n_filters, filter_size2, filter_size2, activation='relu', border_mode='valid'))

model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(n_dense))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation('softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


batch_size = 128
# n_epochs = 100
# 2 epochs 4-5 mins
n_epochs = 50
# 50 epochs better than 100...

early_stopping = EarlyStopping(monitor='val_acc', patience=1)
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          nb_epoch=n_epochs,verbose=1,
          validation_split=.2)

yPred = model.predict_classes(x_valid,batch_size=32,verbose=1)

np.savetxt('mnist_output6.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')