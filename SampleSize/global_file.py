#%%
import pickle
from pickle import load
from pickle import dump
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import Callback

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# K.tensorflow_backend._get_available_gpus()
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import itertools
import random


# Define working direcrtory
path = r'D:\~your work path~\SurrogateNN\SampleSize'

