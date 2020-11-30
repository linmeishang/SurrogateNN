# data procing before training and save data into a seperate folder
# must type in the command "conda activate base"

#%%
import pickle
import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from pickle import dump
#%%
# load parquet
path = r'N:\agpo\work2\MindStep\SurrogateNN\Data'

all_parquets = glob.glob(os.path.join(path+"\\total_df_20*.parquet.gzip"))

# find the latest df
df = max(all_parquets, key=os.path.getctime)
print(df)

# load df from parquet
df = pd.read_parquet(df)

print("shape of df:", df.shape)
print("df:", df) 

#%%

path = r'N:\agpo\work2\MindStep\SurrogateNN\TrainData'

os.chdir(path)

print("Current Working Directory " , os.getcwd())

#%%
#Load Input and outout table
InputOutputArable = pd.read_excel('InputOutputArable.xlsx', index_col=0)  

# Assign these whose Input = 1 to in_col
Input = InputOutputArable.query('Input==1')
Output = InputOutputArable.query('Output==1')

# Get name of indexs
in_col = Input.index.values.tolist() 
out_col = Output.index.values.tolist()

print("Input features: ", in_col)
print("Output targets: ", out_col)


#%%
X_all = df.reindex(columns = in_col)
Y_all = df.reindex(columns = out_col)

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
# Creat a new folder with DATE under TrainData and save all parquets there

# define the name of dir to be created 
path = r"N:\agpo\work2\MindStep\SurrogateNN\TrainData\TrainData"+ datetime.now().strftime("_%Y%m%d")

try:
    os.makedirs(path)

except OSError:
    print ("Creation of the directory %s failed" % path)

else:
    print ("Successfully created the directory %s" % path)

# change working directory in order to save all data later
os.chdir(path)

print("Current Working Directory " , os.getcwd())

#%%
# Normalize/standardize X and Y
# # test set must be normalized with X_scaler of training set
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

X_train = X_scaler.fit_transform(X_train_raw)
X_test  = X_scaler.transform(X_test_raw)
Y_train = Y_scaler.fit_transform(Y_train_raw)
Y_test  = Y_scaler.transform(Y_test_raw)

# After transformation, names of columns are lost. Now we put them back
X_train = pd.DataFrame(X_train, columns = X_train_raw.columns)
Y_train = pd.DataFrame(Y_train, columns = Y_train_raw.columns)
X_test = pd.DataFrame(X_test, columns = X_test_raw.columns)
Y_test = pd.DataFrame(Y_test, columns = Y_test_raw.columns)

#%%
# Pickle scaler X_scaler Y_scaler
dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(Y_scaler, open('Y_scaler.pkl', 'wb'))
# For loading back
# X_scaler = load(open('X_scaler.pkl', 'rb'))
# Y_scaler = load(open('Y_scaler.pkl', 'rb'))


#%%
my_list = [X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, X_train, Y_train, X_test, Y_test]

filenames = ['X_train_raw', 'Y_train_raw', 'X_test_raw', 'Y_test_raw', 'X_train', 'Y_train', 'X_test', 'Y_test']

# parallel iteration of two list
#
for df, j in zip(my_list, range(8)):

    filename = filenames[j]

    #  save all data as parquet files
    df.to_parquet(filename+'.parquet.gzip', compression='gzip')


#%%
# # read parquet
# Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip') 
# print(Y_test_raw)

#%%
