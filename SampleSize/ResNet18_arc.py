# This file defines the Train_ResNet18 function
#%%
import tensorflow as tf
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dropout, Activation, Input, Dense, Conv1D, Flatten, Conv2D, MaxPooling1D
from tensorflow.keras.layers import Add, Activation, ZeroPadding1D, BatchNormalization, AveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# K.tensorflow_backend._get_available_gpus()
from keras.initializers import glorot_uniform
from scipy import stats


#%%
#define identity block  
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block  
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

#%%
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(F1,  kernel_size = 3, strides = s, padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    # Second component of main path 
    X = Conv1D(F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer=glorot_uniform(seed =0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv1D(F2, kernel_size = 1, strides = s, name = conv_name_base + '1', kernel_initializer=glorot_uniform(seed =0))(X_shortcut)
    X_shortcut = BatchNormalization(name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

#%%

def ResNet18(input_shape, classes):
    """
    Implementation of the popular ResNet18 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*1 -> CONVBLOCK -> IDBLOCK*1
    -> CONVBLOCK -> IDBLOCK*1 -> CONVBLOCK -> IDBLOCK*1 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding1D(padding = 1)(X_input)

    filters_stage_2 = [32, 32]
    filters_stage_3 = [2*32, 2*32]
    filters_stage_4 = [4*32, 4*32]
    filters_stage_5 = [8*32, 8*32]
    
    # Stage 1
    X = Conv1D(64, kernel_size = 7, strides = 1, name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size = 3, strides= 2)(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = filters_stage_2, stage = 2, block='a', s = 1)
    X = identity_block(X, 3, filters_stage_2, stage=2, block='b')


    # Stage 3 
    X = convolutional_block(X, f = 3, filters = filters_stage_3, stage = 3, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_3, stage=3, block='b')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = filters_stage_4, stage = 4, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_4, stage=4, block='b')


    # Stage 5 
    X = convolutional_block(X, f = 3, filters = filters_stage_5, stage = 5, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_5, stage=5, block='b')

    # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
    # X = AveragePooling1D()(X) # causing error

    # output layer
    X = Flatten()(X)
    
    outputs = Dense(classes, activation='relu')(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = outputs, name='ResNet18')


    return model



#%%
# This function takes the training data, and fixed hyperparameters, model name and a random see as arguments
def Train_ResNet18(X_train, Y_train, m, lr, optimizer, n_epoch, model_name, seed):

    # X_train must be a 3D tensor
    X_train_2 = np.expand_dims(X_train,axis=-1)
    # print(X_train_2.shape)
    
    tf.random.set_seed(seed)

    model = ResNet18(input_shape = (X_train.shape[1], 1), classes = Y_train.shape[1])

    model.summary()

    model.compile(optimizer= optimizer, loss='mse', 
                metrics= 'mse')

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint(model_name+"_"+str(seed)+'.h5', monitor='val_loss', mode='min', verbose=0, save_weights_only = False, 
                            save_best_only=True)
    csv_logger = CSVLogger(model_name+"_"+str(seed)+'training.log')

    my_history = model.fit(x = X_train_2, y = Y_train, 
                epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=0, 
                callbacks=[es, mc, csv_logger])

    tf.keras.backend.clear_session
