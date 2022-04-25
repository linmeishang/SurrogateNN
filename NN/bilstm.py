# This is the code of training a BiLSTM
#%%
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(1254)
import glob
import pickle
from pickle import load
from pickle import dump
from datetime import datetime
import time
import tensorflow as tf
tf.random.set_seed(89)
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dropout, Activation, Input, Dense, Conv1D, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from scipy import stats
from sklearn.metrics import r2_score


#%%
# import data
# find the latest DataPreparation
path = r'D:\~your work path~\SurrogateNN\DataPreparation'

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

print('shape of X_train:', X_train.shape)
print('shape of Y_train:', Y_train.shape)
print('shape of X_test:', X_test.shape)
print('shape of Y_test:', Y_test.shape)

#%%
# load X_scaler and Y_scaler
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))

# load true raw data 
Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')

# set a new working directory and save the model and all related results

model_path = path+"BiLSTM"+ datetime.now().strftime("_%Y%m%d%H") 

try:
    os.makedirs(model_path)
except OSError:
    print ("Creation of the directory %s failed" % model_path)
else:
    print ("Successfully created the directory %s" % model_path)

# change working directory in order to save all data later
os.chdir(model_path)

print("Current Working Directory " , os.getcwd())

#%%
model_name = str(model_path[-17:])
print(model_name)
#%%
# X_train must be a 3D tensor
X_train_2 = np.expand_dims(X_train,axis=-1)
print(X_train_2.shape)

X_test_2 = np.expand_dims(X_test,axis=-1)
print(X_test_2.shape)

Y_train_2 = np.expand_dims(Y_train,axis=-1)
print(Y_train_2.shape)

Y_test_2 = np.expand_dims(Y_test,axis=-1)
print(Y_test_2.shape)



#%%
# Sepecify number of nodes in each layer, mini-batch size, epochs, learning rate, oprimizer, and drop out rate
layer_1, layer_2, layer_3 = 256, 0, 0 # specify your own numbers
m = 32
n_epoch = 200
lr = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
dropout_rate = 0.0


inputs=Input(shape=(X_train_2.shape[1], X_train_2.shape[2]))

n_timesteps = X_train_2.shape[1]
print(n_timesteps)

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
number_parameters = model.count_params()

model.compile(optimizer= optimizer, loss='mse', metrics= 'mse')



# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_weights_only = False, save_best_only=True)
csv_logger = CSVLogger('training.log')


# Record training time
start_time = time.time()
my_history = model.fit(x = X_train_2, y = Y_train_2, 
            epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=2, 
            callbacks=[es, mc, csv_logger])
end_time = time.time()

time_to_stop =  end_time - start_time
stopped_epoch = es.stopped_epoch

# save the model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# save the loss figure 
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
model = load_model('best_model.h5')

#%%
train_score = model.evaluate(x = X_train_2, y = Y_train_2, batch_size=m, verbose=0)
print("train_score = ", train_score)
# train_score contains 3 losses: mean loss, loss of classification and regression

test_score = model.evaluate(x = X_test_2, y = Y_test_2, batch_size=m, verbose=0)
print("test_score = ", test_score)


# Precit for train dataset
yhat_train = model.predict(X_train_2)
yhat_train = pd.DataFrame(yhat_train, columns = Y_train.columns)
yhat_train.to_parquet('yhat_train.parquet.gzip', compression='gzip')


# Predict for test dataset and record predicting time
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
hyperparameters_dic = {"layer_1": layer_1, 
                        "layer_2": layer_2, 
                        "layer_3": layer_3, 
                        "train_size": X_train_2.shape[0],
			            "test_size": X_test_2.shape[0],
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

