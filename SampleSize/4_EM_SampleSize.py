# Thif file collect the results of evaluation metrics
#%%
from global_file import * 

os.chdir(path)
print("Current Working Directory " , os.getcwd())

# load evaluation metrics
from EvaluationMetrics import *

# load the raw test data
X_test_raw = pd.read_parquet('X_test_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')
print('shape of X_test_raw:', X_test_raw.shape)
print('shape of Y_test_raw:', Y_test_raw.shape)

#%%
all_folders = glob.glob(os.path.join(path + '/Sample*/'))
print(all_folders)

SizeList = ['Sample1000', 'Sample10000', 'Sample100000', 
            'Sample150000', 'Sample5000', 'Sample50000']

#%%
for sample_path, i in zip(all_folders[0:], SizeList): 
    
    # set this sample folder as work dir
    SamplePath = sample_path
    os.chdir(SamplePath)
    print("Current Working Directory " , os.getcwd())

    # load the normalozed X_test and Y_test
    X_test = pd.read_parquet('X_test.parquet.gzip') 
    Y_test = pd.read_parquet('Y_test.parquet.gzip')
    # print('shape of X_test:', X_test.shape)
    # print('shape of Y_test:', Y_test.shape)


    all_models = glob.glob(os.path.join(SamplePath + '/Model*/'))
    # print(all_models)


    Evaluation_total_df = pd.DataFrame()

    for model_path in all_models[0:]:

        os.chdir(model_path)
        print("Current Working Directory " , os.getcwd())
        
        yhat_test_raw = pd.read_parquet('yhat_test_raw.parquet.gzip')
        yhat_test = pd.read_parquet('yhat_test.parquet.gzip')

        # reindex of yhat_test_raw for later calculation purpose
        index_Y_test_raw = Y_test_raw.index.tolist()
        yhat_test_raw.index =index_Y_test_raw

        index_Y_test = Y_test.index.tolist()
        yhat_test.index =index_Y_test

    
        # load evaluation metrics
        #######################################################################################################
        EvaluationMetrics_dic = EvaluationMetrics(Y_test = Y_test, X_test_raw =  X_test_raw, Y_test_raw = Y_test_raw, 
                            yhat_test = yhat_test, yhat_test_raw = yhat_test_raw)


        EvaluationMetrics_dic['model_path'] = str(model_path)[46:]
        # print(EvaluationMetrics_dic)

        df = pd.DataFrame.from_dict(EvaluationMetrics_dic, orient = 'index')
        # append this model EM to the total one
        Evaluation_total_df = pd.concat([Evaluation_total_df, df], axis = 1)


    print(Evaluation_total_df)


    os.chdir(path)
    print("Current Working Directory " , os.getcwd())

    filename = 'Evaluation_total_df_'+ str(i)
    Evaluation_total_df.to_excel(filename+'.xlsx')



# %%
