# training a NN
#%%
import tensorflow as tf
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as ke
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import optimizers
from keras import backend as K

# K.tensorflow_backend._get_available_gpus()
from scipy import stats
from sklearn.metrics import r2_score

#%%
# Find the latest TrainData
path = r"N:\agpo\work2\MindStep\SurrogateNN\TrainData\TrainData_20201117"

os.chdir(path)

print("Current Working Directory " , os.getcwd())

# load data and read parquet
X_train = pd.read_parquet('X_train.parquet.gzip') 
Y_train = pd.read_parquet('Y_train.parquet.gzip') 

X_test = pd.read_parquet('X_test.parquet.gzip') 
Y_test = pd.read_parquet('Y_test.parquet.gzip')

#%%
print('X_train:', X_train)
print('Y_train:', Y_train)
print('shape of X_train:', X_train.shape)
print('shape of Y_train:', Y_train.shape)
print('shape of X_test:', X_test.shape)
print('shape of Y_test:', Y_test.shape)


# %%
model = Sequential()
model.add(Dense(8, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(Y_train.shape[1], activation='linear'))
print(model.summary())


# %%
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
my_history = model.fit(X_train, Y_train, 
                       batch_size=32, epochs=50, verbose=2,
                       validation_split=0.1
                       )

train_score = model.evaluate(X_train, Y_train, batch_size=32, verbose=0)
print("train_score = ", train_score)

test_score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
print("test_score = ", test_score)



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
plt.savefig('./fig_BaselineANN.png')
plt.show()

#%%
# Precit for train dataset
yhat_train = model.predict(X_train)
print(yhat_train.shape)

# Predict for test dataset
yhat_test = model.predict(X_test)
print(yhat_test.shape)

r2_train = r2_score(Y_train, yhat_train)
print("r2 for train", r2_train)

r2_test = r2_score(Y_test, yhat_test)
print("r2 for test", r2_test)



# save the model
model.save("my_model") # need a self-defined name

# tomorrow: find the latest folder, gpu, save everything into a folder, inputs and outputs table