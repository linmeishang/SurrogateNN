# This is the code of training a ResNet34
#%%
import tensorflow as tf
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
from pickle import load
from pickle import dump
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dropout, Activation, Input, Dense, Conv1D, Flatten, Conv2D, MaxPooling1D
from tensorflow.keras.layers import Add, Activation, ZeroPadding1D, BatchNormalization, AveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.utils import plot_model
# K.tensorflow_backend._get_available_gpus()
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from scipy import stats
from sklearn.metrics import r2_score


#%%
# find the all DataPreparation folders
path = r'N:\agpo\work2\MindStep\SurrogateNN\DataPreparation'

all_folders = glob.glob(os.path.join(path + '/*/'))

# find the latest Train Data
DataPreparation = max(all_folders, key=os.path.getctime)

# set DataPreparation as work dir
path = DataPreparation

os.chdir(path)
print("Current Working Directory " , os.getcwd())

#%%
# load data and read parquet
X_train = pd.read_parquet('X_train.parquet.gzip') 
Y_train = pd.read_parquet('Y_train.parquet.gzip') 
X_test = pd.read_parquet('X_test.parquet.gzip') 
Y_test = pd.read_parquet('Y_test.parquet.gzip')

# load X_scaler and Y_scaler
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))

# load true raw data 
Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')

#%%
# X_train must be a 3D tensor
X_train_2 = np.expand_dims(X_train,axis=-1)
print(X_train_2.shape)

X_test_2 = np.expand_dims(X_test,axis=-1)
print(X_test_2.shape)

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
def ResNet34(input_shape, classes):
    """
    Implementation of the popular ResNet34 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

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
    
    # Stage 1
    X = Conv1D(64, kernel_size = 7, strides = 1, name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size = 3, strides= 2)(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = filters_stage_2, stage = 2, block='a', s = 1)
    X = identity_block(X, 3, filters_stage_2, stage=2, block='b')
    X = identity_block(X, 3, filters_stage_2, stage=2, block='c')


    # Stage 3 
    X = convolutional_block(X, f = 3, filters = filters_stage_3, stage = 3, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_3, stage=3, block='b')
    X = identity_block(X, 3, filters_stage_3, stage=3, block='c')
    X = identity_block(X, 3, filters_stage_3, stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = filters_stage_4, stage = 4, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_4, stage=4, block='b')
    X = identity_block(X, 3, filters_stage_4, stage=4, block='c')
    X = identity_block(X, 3, filters_stage_4, stage=4, block='d')
    X = identity_block(X, 3, filters_stage_4, stage=4, block='e')
    X = identity_block(X, 3, filters_stage_4, stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = filters_stage_5, stage = 5, block='a', s = 2)
    X = identity_block(X, 3, filters_stage_5, stage=5, block='b')
    X = identity_block(X, 3, filters_stage_5, stage=5, block='c')

    # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
    # X = AveragePooling1D()(X) # causing error

    # output layer
    X = Flatten()(X)
    
    outputs = Dense(classes, activation='relu')(X)

    # Create model
    model = Model(inputs = X_input, outputs = outputs, name='ResNet34')


    return model


#%%
# Define number of filters in the second stage
f = 4
filters_stage_2 = [f, f]
filters_stage_3 = [2*f, 2*f]
filters_stage_4 = [4*f, 4*f]
filters_stage_5 = [8*f, 8*f]


# set a new working directory and save the model and all related results
model_path = path+"ResNet34"+ datetime.now().strftime("_%Y%m%d%H") 

try:
    os.makedirs(model_path)
except OSError:
    print ("Creation of the directory %s failed" % model_path)
else:
    print ("Successfully created the directory %s" % model_path)

# change working directory in order to save all data later
os.chdir(model_path)

print("Current Working Directory " , os.getcwd())

model_name = str(model_path[-19:])

print(model_name)

# Define hyperparameters
m = 32
n_epoch = 200
lr = 0.001
optimizer = optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
dropout_rate = 0.0   


model = ResNet34(input_shape = (X_train.shape[1], 1), classes = Y_train.shape[1])
model.summary()
number_parameters = model.count_params()

model.compile(optimizer= optimizer, loss='mse', metrics= 'mse')


# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_weights_only = False, save_best_only=True)
csv_logger = CSVLogger('training.log')


# Record the training time
start_time = time.time()

my_history = model.fit(x = X_train_2, y = Y_train, 
            epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=2, 
            callbacks=[es, mc, csv_logger])

end_time = time.time()

time_to_stop =  end_time - start_time
stopped_epoch = es.stopped_epoch


#%%
# save the model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#%%
# Summarize history for loss
print(my_history.history)
history = my_history.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('./fig_loss.png')
plt.show()

#%%
model = load_model('best_model.h5')

#%%
train_score = model.evaluate(x = X_train_2, y = Y_train, batch_size=m, verbose=0)
print("train_score = ", train_score)
# train_score contains 3 losses: mean loss, loss of classification and regression

test_score = model.evaluate(x = X_test_2, y = Y_test, batch_size=m, verbose=0)
print("test_score = ", test_score)


# Precit for train dataset
yhat_train = model.predict(X_train_2)
yhat_train = pd.DataFrame(yhat_train, columns = Y_train.columns)
yhat_train.to_parquet('yhat_train.parquet.gzip', compression='gzip')


# Predict for test dataset, record time for predicting, and save the results
start_time = time.time()
yhat_test = model.predict(X_test_2)
end_time = time.time()
time_cost_test =  end_time - start_time
print("time_cost_test = ", time_cost_test)

yhat_test = pd.DataFrame(yhat_test, columns = Y_test.columns)
yhat_test.to_parquet('yhat_test.parquet.gzip', compression='gzip')


# r2 of train dataset without scaling back to raw values
r2_train = r2_score(Y_train, yhat_train)
print("r2 of training set", r2_train)

# r2 for test dataset
r2_test = r2_score(Y_test, yhat_test)
print("r2 of test set", r2_test)


yhat_train_raw = Y_scaler.inverse_transform(yhat_train)
yhat_test_raw = Y_scaler.inverse_transform(yhat_test)

# Assign columns names to y_hat
yhat_train_raw = pd.DataFrame(yhat_train_raw, columns = Y_train.columns)
yhat_test_raw = pd.DataFrame(yhat_test_raw, columns = Y_test.columns)

# save predicted yhat
yhat_train_raw.to_parquet('yhat_train_raw.parquet.gzip', compression='gzip')
yhat_test_raw.to_parquet('yhat_test_raw.parquet.gzip', compression='gzip')


# R2 for each targets for test dataset

test_r2_dic = {}

for k, j in zip(range(0,len(Y_test.columns)), Y_test.columns): 

    y_true = Y_test.iloc[:,k]

    y_pred = yhat_test.iloc[:,k]

    r2 = r2_score(y_true,y_pred)

    test_r2_dic[j] = r2   


# store hyperparameters into a list
hyperparameters_dic = {"filters_stage_2": str(filters_stage_2), 
                        "filters_stage_3": str(filters_stage_3), 
                        "filters_stage_4": str(filters_stage_4), 
                        "filters_stage_5": str(filters_stage_5), 
                        "train_size": X_train.shape[0],
                        "test_size": X_test.shape[0],
                        "learning_rate": lr,
                        "minibatch_size": m,
                        "dropout_rate": dropout_rate,
                        "epoch": n_epoch, 
                        "stopped_epoch": stopped_epoch,
                        "optimizer": optimizer, 
                        "number_parameters": number_parameters,
                        "time_to_stop": time_to_stop, 
                        "time_cost_test": time_cost_test} 


# store overall indicators into a list
train_dic = {"r2_train": r2_train, "r2_test": r2_test} 


# Combine all dictionaries that we want to store
result_dic = {**hyperparameters_dic, **train_dic,  **test_r2_dic}


# Append this dictionary to an excel in model assessment
df = pd.DataFrame(data=result_dic, index=[model_name])
df = (df.T)
print (df)
df.to_excel(model_name+".xlsx")