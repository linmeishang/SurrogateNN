# This file defines the Train_BiLSTM function
#%%
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dropout, Activation, Input, Dense, Conv1D, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from scipy import stats

#%%
# This function takes the training data, and fixed hyperparameters, model name and a random see as arguments
def Train_BiLSTM(X_train, Y_train, layer_1, layer_2, layer_3, m, lr, optimizer, n_epoch, model_name, seed):

    X_train_2 = np.expand_dims(X_train,axis=-1)
    Y_train_2 = np.expand_dims(Y_train,axis=-1)

    tf.random.set_seed(seed)

    inputs=Input(shape=(X_train_2.shape[1], X_train_2.shape[2]))

    n_timesteps = X_train_2.shape[1]

    # print(n_timesteps)

    if layer_2 == 0:
        rnn = Bidirectional(LSTM(layer_1, activation='tanh', return_sequences = False, input_shape=(n_timesteps, 1)))(inputs) 
        # ruu = Dropout(dropout_rate)(rnn) 

    else:
        if layer_3 == 0:
            rnn = Bidirectional(LSTM(layer_1, activation='tanh', return_sequences = True))(inputs)
            # ruu = Dropout(dropout_rate)(rnn) 
            rnn = Bidirectional(LSTM(layer_2, activation='tanh', return_sequences = False))(rnn)
            # ruu = Dropout(dropout_rate)(rnn) 
        else:
            rnn = Bidirectional(LSTM(layer_1, activation='tanh', return_sequences = True))(inputs)
            # ruu = Dropout(dropout_rate)(rnn) 
            rnn = Bidirectional(LSTM(layer_2, activation='tanh', return_sequences = True))(rnn)
            # ruu = Dropout(dropout_rate)(rnn) 
            rnn = Bidirectional(LSTM(layer_3, activation='tanh', return_sequences= False))(rnn)
            # ruu = Dropout(dropout_rate)(rnn) 


    outputs = Dense(Y_train_2.shape[1], activation='relu')(rnn)


    model = Model(inputs=inputs, outputs= outputs)

    model.summary()

    model.compile(optimizer= optimizer, loss='mse', 
                    metrics= 'mse')

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint(model_name+"_"+str(seed)+'.h5', monitor='val_loss', mode='min', verbose=0, save_weights_only = False, 
                            save_best_only=True)
    csv_logger = CSVLogger(model_name+"_"+str(seed)+'training.log')


    my_history = model.fit(x = X_train_2, y = Y_train_2, 
                epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=0, 
                callbacks=[es, mc, csv_logger])

    tf.keras.backend.clear_session