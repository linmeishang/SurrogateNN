# SurrogateNN
This repository contains 5 folders. Their functions are decribed below.
## 1. DataCollection
This folder contains the raw farm data generated from the farm-level model FarmDyn. These data was generated and stored each time in gdx files. 

The file read_gdx.py reads the gdx files in each folder and stores the data in a parquet file. 

At the end, data from all parquet files was combined together into the total_df_(date).parquet.gzip file.

The read_gdx.py can automatically add the newly added folder of farm data to the total_df_(date).parquet.gzip file, without needing to read all gdx files again.

The dataset we used for developing surrogate models contains 163480 farm draws. 
 
## 2. DataPreparation
This folder prepares the data before training, including , spliting train and test data, normalization, and storing the prepared data. 

The data_preparation.py file does the following steps of prepration: 
1) Load the newest total_df_(date).parquet.gzip file from the DataCollection folder.
2) Seperate inputs (X) and outputs (Y) based on the InputOutputArable.xlsx file.
3) Split train and test datasets.
4) Normalize the train and test datasets.
5) Store the datasets (including raw data and normalized data) and the scalers in a DataPreparation_(date) folder. 

## 3. NN
This folder contains the python files of training different architetures of NNs, including:
1) MLP
2) ResNets (18, 34, 50)
3) LSTM
4) BiLSTM

The trained models and thier predictions are stored in the correpsonding DataPreparation_(date) folder. We have only reserved the 12 best models in the DataPreparation_(date) folder for simplication. The trained models can be reloaded and used for predictions directly. 

## 4. SampleSize
This folder seperates the original training set in to different sample sizes, trains models using different sample sizes, and evaluates the performance of all models using our evaluation metrics (EM). Here are the desriptions of all files in this folder:
1) 1_seperate_data.py: It seperates the data into samples sizes of {1000, 5000, 10000, 50000, 100000, 150000}, and it normalizes the training set and test set. Data of each samples size is stored in a Sample(size) (e.g. Sample1000) folder.

2) 2_train_nn.py: It loads the pre-defined the hyperparameters from the 12 best models, trains new models giving random seeds, and stores the trained model in each SampleSize(size) folder. 

3) 3_predict.py: It loads the trained models from each Sample(size) folder, creates a model folder for each model, and saves the predictions in the  model folder. 

4) 4_EM_SampleSize.py: It loads the Evaluation_Metrics.py, evaluates the performances of all models, and stores the results of each SampleSize(size) folder in a excel file, named Evaluation_total_df_Sample(size).xlsx.

5) 5_collect_EM.py: It calculates the avargae performance of 5 random models of each architecture and stores the result in a excel file, name results.xlsx.

## 5. Visualization
This folder contains the code of visualizes the results of inference time and results from the SampleSize experiment of the last step. 
