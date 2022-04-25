#%%
# This file go through model folders and make predictions using the trained models and store the results

from global_file import * 

os.chdir(path)
print("Current Working Directory " , os.getcwd())

# Find all sample folders
all_folders = glob.glob(os.path.join(path + '/Sample*/'))
print(all_folders)

#%%
# We have trained all models, we will make predictions with all models
for sample_path in all_folders[0:]: 
    
    # set this sample folder as work dir
    path = sample_path
    os.chdir(path)
    print("Current Working Directory " , os.getcwd())


    # load the normalized train and test data

    # add a if not MLP then transform X_test (and) Y_test
    X_test = pd.read_parquet('X_test.parquet.gzip') 
    Y_test = pd.read_parquet('Y_test.parquet.gzip')
    print('shape of X_test:', X_test.shape)
    print('shape of Y_test:', Y_test.shape)


    # # this is only for ResNet 
    # X_test_2 = np.expand_dims(X_test,axis=-1)
    # print(X_test_2.shape)


    # # this is for LSTM and BiLSTM
    # X_train must be a 3D tensor
    # X_test_2 = np.expand_dims(X_test,axis=-1)
    # print('shape of X_test_2:', X_test_2.shape)
    # Y_test_2 = np.expand_dims(Y_test,axis=-1)
    # print('shape of Y_test_2:',Y_test_2.shape)


    # Load the scaler
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    Y_scaler = load(open('Y_scaler.pkl', 'rb'))
    # Now all models are trained for all sample sizes
    # We need to create folders for each model to store their predictions for X_test

    
    all_models = glob.glob(os.path.join(path+"\\*.h5"))
    print(all_models)


    all_names = []
    for i in all_models: 
        all_names.append(str(i[-11:-3])) # for MLP
        # all_names.append(str(i[-15:-3])) # for ResNet
        # all_names.append(str(i[-11:-3])) # for LSTM
    print(all_names)


    for model, name in zip(all_models[0:], all_names[0:]):

        # load the trained model
        model = load_model(model)

        # Precit for test dataset
        yhat_test = model.predict(X_test) # for ResNet and LSTM and BiLSTM it should be X_test_2
        print('shape of yhat_test:', yhat_test.shape)
        yhat_test = pd.DataFrame(yhat_test, columns = Y_test.columns)
        num_nan = yhat_test.isna().sum().sum()
        print('number of nan in yhat_test:', num_nan)

        if num_nan == 0:
            # transform the test data to original scale
            yhat_test_raw = Y_scaler.inverse_transform(yhat_test)
            yhat_test_raw = pd.DataFrame(yhat_test_raw, columns = Y_test.columns)

            # Caculate R2 or # Import Evaluation Metrics
            r2_test = r2_score(Y_test, yhat_test) # correct. Should not be Y_test_2
            print("r2 of test set", r2_test)

            r2_dict = {name: r2_test}
            print(r2_dict)
            df = pd.DataFrame.from_dict(r2_dict, orient='index').T

            # create a folder and save the predictions
            model_path = os.path.join(path, 'Model_' + name)
            os.makedirs(model_path)
            os.chdir(model_path)
            print("Current Working Directory " , os.getcwd())

            yhat_test.to_parquet('yhat_test.parquet.gzip', compression='gzip')
            yhat_test_raw.to_parquet('yhat_test_raw.parquet.gzip', compression='gzip')
            df.to_excel('r2.xlsx')

        else:
            print('yhat contains nan')
