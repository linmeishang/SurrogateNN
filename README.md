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
## 4. SampleSize
## 5. Visualization
