# This is the code of training a MLP
#%%
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import random
import glob
import pickle
from pickle import load
from pickle import dump
from datetime import datetime
import time
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0) # specify a random seed for reproducibility
rn.seed(1254) # specify a random seed for reproducibility
import tensorflow as tf
tf.random.set_seed(20) # specify a random seed for reproducibility
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
from scipy import stats
from sklearn.metrics import r2_score


#%%
# find all DataPreparation folders
path = r'N:\agpo\work2\MindStep\SurrogateNN\DataPreparation'

all_folders = glob.glob(os.path.join(path + '/*/'))

# find the latest DataPreparation folder
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
X_train_raw = pd.read_parquet('X_train_raw.parquet.gzip') 
X_test_raw = pd.read_parquet('X_test_raw.parquet.gzip')

Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')


#%%
# set a new working directory and save the model and all related results
model_path = path+"MLP"+ datetime.now().strftime("_%Y%m%d%H%M") 

try:
    os.makedirs(model_path)

except OSError:
    print ("Creation of the directory %s failed" % model_path)

else:
    print ("Successfully created the directory %s" % model_path)

# change working directory in order to save all data later
os.chdir(model_path)

print("Current Working Directory " , os.getcwd())

model_name = str(model_path[-16:-1])


#%%
# Sepecify number of nodes in each layer, mini-batch size, epochs, learning rate, oprimizer, and drop out rate
layer_1, layer_2, layer_3 = 20, 0, 0 # specify your own numbers
m = 32
n_epoch = 200
lr = 0.0003
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
dropout_rate = 0.0


# Define the model
inputs = Input(shape = (X_train.shape[1],))

dense = Dense(layer_1, activation = 'relu')(inputs)
# dense = Dropout(dropout_rate)(dense)
if layer_2 != 0:
    dense = Dense(layer_2, activation = 'relu')(dense)
    # dense = Dropout(dropout_rate)(dense)
else: 
    None

if layer_3 != 0:
    dense = Dense(layer_3, activation = 'relu')(dense)
    # dense = Dropout(dropout_rate)(dense)
else: 
    None


outputs = Dense(Y_train.shape[1], activation = 'relu')(dense)


model = Model(inputs = inputs, outputs = outputs)
model.summary()
number_parameters = model.count_params()


model.compile(optimizer= optimizer, loss='mse', metrics= 'mse')


# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_weights_only = False, save_best_only=True)
csv_logger = CSVLogger('training.log')


# Record the training time
start_time = time.time()

my_history = model.fit(x = X_train, y = Y_train, 
            epochs=n_epoch, batch_size=m, shuffle=True, validation_split=0.1, verbose=2, 
            callbacks=[es, mc, csv_logger]
            )

end_time = time.time()

time_to_stop =  end_time - start_time
stopped_epoch = es.stopped_epoch


# save the final model, which might not be the best model
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
# Evaluate model performance 
model = load_model('best_model.h5')

train_score = model.evaluate(x = X_train, y = Y_train, batch_size=m, verbose=0)
print("train_score = ", train_score)
# train_score contains 3 losses: mean loss, loss of classification and regression

test_score = model.evaluate(x = X_test, y = Y_test, batch_size=m, verbose=0)
print("test_score = ", test_score)


# Precit for train dataset and save the results
yhat_train = model.predict(X_train)
yhat_train = pd.DataFrame(yhat_train, columns = Y_train.columns)
yhat_train.to_parquet('yhat_train.parquet.gzip', compression='gzip')


# Predict for test dataset, record time for predicting, and save the results
start_time = time.time()
yhat_test = model.predict(X_test)
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

# Convert the prediction back the original scale
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

