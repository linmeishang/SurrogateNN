# training a NN
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
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as ke
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import optimizers
from keras import backend as K
from keras.losses import mean_squared_error, binary_crossentropy
# K.tensorflow_backend._get_available_gpus()
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score, BinaryAccuracy

#%%
# Find the latest TrainData

path = r'N:\agpo\work2\MindStep\SurrogateNN\TrainData'

all_folders = glob.glob(os.path.join(path + '/*/'))

# find the latest Train Data
TrainData = max(all_folders, key=os.path.getctime)

#%%
# set TrainData as work dir
path = TrainData

os.chdir(path)

print("Current Working Directory " , os.getcwd())

#%%
# load data and read parquet
X_train = pd.read_parquet('X_train.parquet.gzip') 
Y_train = pd.read_parquet('Y_train.parquet.gzip') 

X_test = pd.read_parquet('X_test.parquet.gzip') 
Y_test = pd.read_parquet('Y_test.parquet.gzip')

#%%
num_nans = Y_train.size - Y_train.count().sum()
print(num_nans)
print(Y_train.size)
#%%
# fillna with 0
# Issue: some empty cells are actually, some not. 
# Solution: Inputs replace with mean, output with 0 -> domain knowledge
X_train= X_train.fillna(X_train.mean(axis=0))
Y_train= Y_train.fillna(0)

X_test = X_test.fillna(X_test.mean(axis=0))
Y_test = Y_test.fillna(0)

#%%
print('X_train:', X_train)
print('Y_train:', Y_train)
print('shape of X_train:', X_train.shape)
print('shape of Y_train:', Y_train.shape)
print('shape of X_test:', X_test.shape)
print('shape of Y_test:', Y_test.shape)

#%%
# number of binary outputs
# Binary loss: bunkerSilo0, potaStore500t, the first two columns of Y
b = 2  
# c: number of continuous outputs
c = Y_train.shape[1] - b
# Weights of different outputs
weights = tf.ones([1,c]) 


# Define activation functions for different outputs
def myactivation(x):

    #x is the layer's output, shaped as (batch_size, units)
    #each element in the last dimension is a neuron
    n0 = x[:,0:b]
    #each Neuron is shaped as (batch_size, 1)
    n1 = x[:,b:]  
    #apply the activation to each neuron
    x0 = K.sigmoid(n0)

    x1 = K.relu(n1)
    #return to the original shape
    return K.concatenate([x0,x1], axis=-1) 

# Define a cosumized loss function
def my_custom_loss(y_true, y_pred):
    
    crossentropy = binary_crossentropy(y_true[:,0:b], y_pred[:,0:b])

    mse = tf.reduce_mean(tf.square((y_true[:,b:] - y_pred[:,b:])*weights), axis=-1)
    
    return mse + crossentropy

# Define the model
model = Sequential()

model.add(Dense(40, input_shape=(X_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(Y_train.shape[1], activation=myactivation))

print(model.summary())

# compile the model
optimizer = keras.optimizers.Adam(lr=0.001)

model.compile(loss=my_custom_loss, optimizer=optimizer)

my_history = model.fit(X_train, Y_train, 
                       batch_size=32, epochs=50, verbose=2,
                       validation_split=0.1
                       )

train_score = model.evaluate(X_train, Y_train, batch_size=32, verbose=0)
print("train_score = ", train_score)

test_score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
print("test_score = ", test_score)

# save the model
model.save("model") 

#%%
# Summarize history for loss
print(my_history.history.keys())
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
# Precit for train dataset
yhat_train = model.predict(X_train)
print(yhat_train.shape)

# Predict for test dataset
yhat_test = model.predict(X_test)
print(yhat_test.shape)

# r2 or train dataset
r2_train = r2_score(Y_train, yhat_train)
print("r2 for train", r2_train)

# r2 for test dataset
r2_test = r2_score(Y_test, yhat_test)
print("r2 for test", r2_test)


#%%
# Trainsform predicted value into raw data scale
# load X_scaler and Y_scaler
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))

#%%
yhat_train_raw = Y_scaler.inverse_transform(yhat_train)
yhat_test_raw = Y_scaler.inverse_transform(yhat_test)

# Assign columns names to y_hat
yhat_train_raw = pd.DataFrame(yhat_train_raw, columns = Y_train.columns)
yhat_test_raw = pd.DataFrame(yhat_test_raw, columns = Y_test.columns)

# save predicions
yhat_train_raw.to_parquet('yhat_train_raw.parquet.gzip', compression='gzip')
yhat_test_raw.to_parquet('yhat_test_raw.parquet.gzip', compression='gzip')

#%%
# load true raw data 
Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')

# # load predicted raw data
# yhat_train_raw = pd.read_parquet('yhat_train_raw.parquet.gzip') 
# yhat_test_raw = pd.read_parquet('yhat_test_raw.parquet.gzip')

#%%
# fill nan values of Y_train_raw and Y_test_raw
# Y_train_raw = Y_train_raw.fillna(0)
Y_test_raw = Y_test_raw.fillna(0)

#%%
# Plot R2 for each targets for test dataset
accuracy_dic = {}

r2_dic = {}

for k, j in zip(range(0,len(Y_test_raw.columns)), Y_test_raw.columns): 

    y_true = Y_test_raw.iloc[:,k]

    y_pred = yhat_test_raw.iloc[:,k]

    if k < b: 

        accuracy = K.get_value(tf.keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5))
        
        accuracy_dic[j] = accuracy

    else:
    
        r2 = r2_score(y_true,y_pred)

        r2_dic[j] = r2    
    # append to the dictionary of result using the column name


    # plt.figure(figsize=(5,5))

    # plt.plot(y_true,y_pred,'o',color='black', markersize=2)

    # plt.title(str('y'+ str(k)))

    # plt.xlabel('y')

    # plt.ylabel('yhat')

    # plt.savefig(str('k = '+ str(k) + '_BaselineANN'))


#%%
# store indicators into a list
train_dic = {"train score":train_score, "test score": test_score, "r2_train": r2_train, "r2_test": r2_test} 

result_dic = {**train_dic, **accuracy_dic, **r2_dic}

#%%
# save results
dump(result_dic, open('result_dic.pkl', 'wb'))
# %%
dic = load(open('result_dic.pkl', 'rb'))
print(dic)

# %%
# save python interactive as txt
# 2>&1 | tee log_linmei.txt


#%%
# # predict for a single sample
# x_train = np.full((X_train.shape[1], ), 0.2) # fill with 0.2
# x_train = np.array([x_train])
# print(x_train.shape)

# y_hat = model.predict(x_train)
# print(y_hat)
# print(y_hat.shape)