import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras

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

# train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
# test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))