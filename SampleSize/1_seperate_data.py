# This file seperate the training data in to different sizes
#%%
from global_file import * 
os.chdir(path)
print("Current Working Directory " , os.getcwd())

# Load raw data 163k with train and test
X_train_raw = pd.read_parquet('X_train_raw.parquet.gzip') 
Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
X_test_raw = pd.read_parquet('X_test_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')

print('shape of X_train_raw:', X_train_raw.shape)
print('shape of Y_train_raw:', Y_train_raw.shape)
print('shape of X_test_raw:', X_test_raw.shape)
print('shape of Y_test_raw:', Y_test_raw.shape)

#%%
# We first create folders of different sample size
# And prepare and store the data
for i in [1000, 5000, 10000, 50000, 100000, 150000]: 

    # create a folder 
    sample_path = r"N:\agpo\work2\MindStep\SurrogateNN\SampleSize\Sample"+ str(i)
    os.makedirs(sample_path)
    os.chdir(sample_path)
    print("Current Working Directory " , os.getcwd())

    # select training data 
    X_train_raw_s = X_train_raw[0:i]
    Y_train_raw_s = Y_train_raw[0:i]
    print('shape of X_train_raw:', X_train_raw_s.shape)
    print('shape of Y_train_raw:', Y_train_raw_s.shape)

    # We do not select test set because we will always use the same test set 
    # But we will normalize the test set according to the selected training set

    # Normalize train and test data
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(X_train_raw_s)
    Y_train = Y_scaler.fit_transform(Y_train_raw_s)
    X_test  = X_scaler.transform(X_test_raw)
    Y_test  = Y_scaler.transform(Y_test_raw)

    # After transformation, names of columns are lost. Now we put them back
    X_train = pd.DataFrame(X_train, columns = X_train_raw.columns)
    Y_train = pd.DataFrame(Y_train, columns = Y_train_raw.columns)
    X_test = pd.DataFrame(X_test, columns = X_test_raw.columns)
    Y_test = pd.DataFrame(Y_test, columns = Y_test_raw.columns)


    # Save the training data
    X_train_raw_s.to_parquet('X_train_raw.parquet.gzip', compression='gzip')
    Y_train_raw_s.to_parquet('Y_train_raw.parquet.gzip', compression='gzip')

    X_train.to_parquet('X_train.parquet.gzip', compression='gzip')
    Y_train.to_parquet('Y_train.parquet.gzip', compression='gzip')

    # save the normalized test data
    X_test.to_parquet('X_test.parquet.gzip', compression='gzip')
    Y_test.to_parquet('Y_test.parquet.gzip', compression='gzip')

    # Save the scaler
    # Pickle scaler X_scaler Y_scaler
    dump(X_scaler, open('X_scaler.pkl', 'wb'))
    dump(Y_scaler, open('Y_scaler.pkl', 'wb'))

