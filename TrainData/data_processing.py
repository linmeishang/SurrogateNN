# data procing before training
# must type in the command "conda activate base"

#%%
import pickle
import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

#%%
# load pkl 
path = r'N:\agpo\work2\MindStep\SurrogateNN\TrainData'

all_pickles = glob.glob(os.path.join(path+"\\total_df_20*.pkl"))

df = max(all_pickles, key=os.path.getctime)

print(df)
#%%
df = pd.read_pickle(df)

print(df)

#%%
# Delete all columns that have "mean"
df = df[df.columns.drop(list(df.filter(regex='mean')))]

print("shape of df:", df.shape)

print("df:", df) 

#%%
# Define X_all and Y_all

in_col = [ 
     # Group 1: Crops hours
     'Crops_hours__2020', # what is this feature
     'Crops_hours_FEB_2020', #
     'Crops_hours_MAR_2020', #  
     'Crops_hours_APR_2020' # 
     # Group 2:  work earn  
       
]
        
       
out_col = [
     'Crops_hours_JUN_2020', # 
     'Crops_hours_JUL_2020', # 
     'Crops_hours_AUG_2020' #
]


X_all = df.loc[:,in_col]
Y_all = df.loc[:,out_col]

print("X:", X_all)
print("Y:", Y_all)
print("shape of X_all:", X_all.shape)
print("shape of Y_all:", Y_all.shape)

#%%
# Train-test split
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_all, Y_all, test_size=0.2, random_state=0)


print("shape of X_train_raw:", X_train_raw.shape)
print("shape of Y_train_raw:", Y_train_raw.shape)
print("shape of X_test_raw:", X_test_raw.shape)
print("shape of Y_test_raw:", Y_test_raw.shape)


#%%
# Normalize/standardize X and Y
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

X_train = X_scaler.fit_transform(X_train_raw)
X_test  = X_scaler.transform(X_test_raw)
Y_train = Y_scaler.fit_transform(Y_train_raw)
Y_test  = Y_scaler.transform(Y_test_raw)


# Save both raw data and normalized data 
# raw data
with open('X_train_raw.pickle','wb') as output:
    pickle.dump(X_train_raw, output)
with open('X_test_raw.pickle','wb') as output:
    pickle.dump(X_test_raw, output)
with open('Y_train_raw.pickle','wb') as output:
    pickle.dump(Y_train_raw, output)
with open('Y_test_raw.pickle','wb') as output:
    pickle.dump(Y_test_raw, output)


# normalized data
with open('X_train.pickle','wb') as output:
    pickle.dump(X_train, output)
with open('X_test.pickle','wb') as output:
    pickle.dump(X_test, output)
with open('Y_train.pickle','wb') as output:
    pickle.dump(Y_train, output)
with open('Y_test.pickle','wb') as output:
    pickle.dump(Y_test, output)

