# SurrogateNN

This github repository shares the data and codes used for developing and evaluating Neural Network surrogates for a detailed farm model FarmDyn (https://github.com/FarmDyn). 

Hardware required: CPU, GPU

Software required: python, GAMS (General Algebraic Modeling System)

Program language: python

Data format: gdx file, pandas dataframe, parquet file

Data size: 346 MB

Size of trained models: 352 KB - 865 MB

Format of trained models: .h5 (keras)

First year available: 2022

Contact information: Linmei Shang, Institute for Food and Resource Economics (ILR), University of Bonn, Nußallee 21, 53115 Bonn, Germany (E-mail address: linmei.shang@ilr.uni-bonn.de)


This repository contains 5 folders. Their functions are decribed below.

## 1. DataCollection
This folder contains the raw data generated from the farm-level model. These data was generated at different times and stored each time in a seperated folder (e.g. 20201208). In each folder, there are many gdx files. Each gdx file is a farm draw. Due to storagte limitation, we only keep an example folder with a few example gdx files.

The file "read_gdx.py" reads the gdx files in each folder and stores the data of each folder in a parquet file (e.g. 20201208.parquet.gzip).

At the end, data from all parquet files was combined together into the file, named "total_df_(date).parquet.gzip". This file can be downloaded by klicking the given link.

The "read_gdx.py" can automatically add the newly added folder of farm data to the "total_df_(date).parquet.gzip" file, without needing to read all gdx files again.

The dataset we used for developing surrogate models contains 163480 farm draws. 
 
## 2. DataPreparation
This folder prepares the data before training. The "data_preparation.py" file does the following steps of prepration: 
1) Load the newest "total_df_(date).parquet.gzip" file from the DataCollection folder.
2) Seperate inputs (X) and outputs (Y) based on the "InputOutputArable.xlsx" file.
3) Split train and test datasets.
4) Normalize the train and test datasets.
5) Store the datasets (including raw data and normalized data) and the scalers in a "DataPreparation_(date)" folder. 

The raw and normalized datasets can be downloaded with the given link. 

## 3. NN
This folder contains the python files of training different architetures of NNs, including:
1) MLP
2) ResNets (18, 34, 50)
3) LSTM
4) BiLSTM

The trained models (best_model.h5) and their predictions are stored in their own model folders under the "DataPreparation_(date)" folder. We only reserved the 4 best models in the "DataPreparation_(date)" folder for minimizing the storage use. The trained models can be reloaded and used for predictions directly. 

## 4. SampleSize
This folder seperates the original training set into different sample sizes, trains models using different sample sizes, and evaluates the performance of all models using our evaluation metrics (EM). Here are the desriptions of the main files in this folder:

1_seperate_data.py: It seperates the data into samples sizes of {1000, 5000, 10000, 50000, 100000, 150000}, and it normalizes the training set and test set. Data of each samples size is stored in a "Sample(size)" folder. We only keep an example of "Sample1000" due to storage limitation.

2_train_nn.py: It loads the pre-defined the hyperparameters from the 12 best models (e.g "mlp_arc.py"), trains new models giving random seeds, and stores the trained model in each "Sample(size)" folder. 

3_predict.py: It loads the trained models from each "Sample(size)" folder, creates a model folder for each model, and saves the predictions in the model folder. 

4_EM_SampleSize.py: It loads the "Evaluation_Metrics.py", evaluates the performances of all models, and stores the results of each "Sample(size)" folder in a excel file, named "Evaluation_total_df_Sample(size).xlsx".

5_collect_EM.py: It calculates the avargae performance of 5 random models of each architecture and stores the result in a excel file, name "results.xlsx".

## 5. Visualization
This folder contains the code of visualizing the results of inference time and results from the SampleSize experiment of the last step. 
