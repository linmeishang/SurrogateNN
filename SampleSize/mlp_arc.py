# This file defines the Train_MLP function
#%%
import numpy as np
import random
import pandas as pd
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
from scipy import stats


#%%
# This function takes the training data, and fixed hyperparameters, model name and a random see as arguments
def Train_MLP(X_train, Y_train, layer_1, layer_2, layer_3, m, lr, optimizer, n_epoch, model_name, seed): 
        
    # os.environ['PYTHONHASHSEED'] = '0'
    # np.random.seed(0)
    # random.seed(1254)
    tf.random.set_seed(seed)

    # Define the model
    inputs = Input(shape = (X_train.shape[1],))

    dense = Dense(layer_1, activation = 'relu')(inputs)

    if layer_2 != 0:
        dense = Dense(layer_2, activation = 'relu')(dense)
    else: 
        None
    if layer_3 != 0:
        dense = Dense(layer_3, activation = 'relu')(dense)
    else: 
        None

    outputs = Dense(Y_train.shape[1], activation = 'relu')(dense)

    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    

    model.compile(optimizer= optimizer, loss='mse', metrics= 'mse')

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint(model_name+"_"+str(seed)+'.h5', monitor='val_loss', mode='min', verbose=0, save_weights_only = False, save_best_only=True)
    csv_logger = CSVLogger(model_name+"_"+str(seed)+'training.log')


    my_history = model.fit(x = X_train, y = Y_train, 
                epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=0, 
                callbacks=[es, mc, csv_logger]
                )
                

    tf.keras.backend.clear_session

