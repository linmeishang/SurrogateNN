# This file trains the NN with different sample sizes
#%%
from global_file import * 

os.chdir(path)
print("Current Working Directory " , os.getcwd())

# Find all folders of different sample sizes
all_folders = glob.glob(os.path.join(path + '/Sample*/'))
print(all_folders)


#%%
# Select 5 random seeds
seeds = random.sample(range(0, 1000), 5)
print(seeds)

# For all samples size folder, train all types of models with 5 seeds
for sample_path in all_folders[0:]:

    tf.keras.backend.clear_session

    # set this sample folder as work dir
    path = sample_path
    os.chdir(path)
    print("Current Working Directory " , os.getcwd())

    # load the normalized train and test data
    X_train = pd.read_parquet('X_train.parquet.gzip') 
    Y_train = pd.read_parquet('Y_train.parquet.gzip')
    print('shape of X_train:', X_train.shape)
    print('shape of Y_train:', Y_train.shape)
        
    # Train all models with 5 seeds
    for seed in seeds:
        
        # Train the 3 MLPs using the fixed architetures
        # All Train_... function trains the model and stores the trained best model in the model folder
        from mlp_arc import *
        Train_MLP(X_train = X_train, Y_train = Y_train, layer_1 = 128, layer_2 = 0, layer_3 = 0, 
                m = 32, lr = 0.001, 
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001) , n_epoch = 200, 
                model_name ="MLP1", seed = seed)

        Train_MLP(X_train = X_train, Y_train = Y_train, layer_1 = 64, layer_2 = 512, layer_3 = 0, 
                    m = 32, lr = 0.0003, 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003) , n_epoch = 200, 
                    model_name ="MLP2", seed = seed)

        Train_MLP(X_train = X_train, Y_train = Y_train, layer_1 = 128, layer_2 = 32, layer_3 = 256, 
                    m = 32, lr = 0.0003, 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003) , n_epoch = 200, 
                    model_name ="MLP3", seed = seed)

        
        # Train the 3 ResNets using the fixed architetures
        
        from ResNet18_arc import *
        Train_ResNet18(X_train = X_train, Y_train = Y_train, 
                        m = 128, lr = 0.001, 
                        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), n_epoch = 200, 
                        model_name ="ResNet18")
                        

        from ResNet34_arc import *
        Train_ResNet34(X_train = X_train, Y_train = Y_train, 
                        m = 32, lr = 0.0003, 
                        optimizer= tf.keras.optimizers.Adam(learning_rate=0.0003), n_epoch = 200, 
                        model_name ="ResNet34")


        from ResNet50_arc import *
        Train_ResNet50(X_train = X_train, Y_train = Y_train, 
                        m = 64, lr = 0.001, 
                        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), n_epoch = 200, 
                        model_name ="ResNet50", seed = seed)


        from lstm_arc import *
        Train_LSTM(X_train = X_train, Y_train = Y_train, layer_1 = 256, layer_2 = 0, layer_3 = 0, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="LSTM1")

        Train_LSTM(X_train = X_train, Y_train = Y_train, layer_1 = 128, layer_2 = 64, layer_3 = 0, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="LSTM2")

        Train_LSTM(X_train = X_train, Y_train = Y_train, layer_1 = 32, layer_2 = 128, layer_3 = 1024, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="LSTM3")


        from BiLSTM_arc import *
        Train_BiLSTM(X_train = X_train, Y_train = Y_train, layer_1 = 2048, layer_2 = 0, layer_3 = 0, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="BiLSTM1")

        Train_BiLSTM(X_train = X_train, Y_train = Y_train, layer_1 = 32, layer_2 = 256, layer_3 = 0, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="BiLSTM2")

        Train_BiLSTM(X_train = X_train, Y_train = Y_train, layer_1 = 32, layer_2 = 128, layer_3 = 512, 
                    m = 32, lr = 0.001, 
                    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001) , n_epoch = 200, 
                    model_name ="BiLSTM3")
