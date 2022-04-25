# This file reads gdx file in python, creates parquet file for each folder, and summarizes them into a whole dataset (total_df_...). 
# When new raw data is generated (i.e. new folders of gdx files are added), this code automatically reads the new data and adds it to the total dataset (total_df_...). 
# This code should work under python 3.6 ("conda activate py36") (might also work in python 3.7 environment).

#%%
import pandas as pd
import gdxpds
from gdxpds import gdx
from collections import OrderedDict
import numpy as np
import glob
import os
import time
from datetime import datetime
import pickle

# new hallo hugo
#%%
# mapperInstance is a function to rename columns so that we can select them
def mapperInstance():

    def gen_i():
        i = 0
        while True:
            yield str(i)
            i = i+1
    gen = gen_i()

    def mapper(s):
        return (s + gen.__next__())
    
    return mapper


#%%
# get_df is a function to read a single gdx file of a single draw from FarmDyn, and transform it into a data frame
def get_df(gdx_file):
   
    p_res = gdxpds.read_gdx.to_dataframe(gdx_file, symbol_name='p_res', gams_dir='N:\soft\GAMS28.3', old_interface=False)
   
    # Rename Columns (1st pass)
    mpr = mapperInstance()
    
    p_res.rename(mpr, axis="columns", inplace=True)
    
    # add a column and give the new column a name (combine all the names before)
    p_res['concat']= pd.Series(p_res[['*2', '*3', '*4', '*5']].fillna('').values.tolist()).map(lambda x: '_'.join(map(str,x)))
    
    df = p_res[['Value6','concat']].T
    
    df = df.rename(columns=df.iloc[1])
    
    df = df.drop(['concat'])
    
    return df

# %%
# get_df_parquet is function to read all gdx files in a folder, concate them and store in as a parquet file
def get_df_parquet(folder):

    all_files = glob.glob(os.path.join(folder + "\\*.gdx"))
    
    df = pd.concat((get_df(gdx_file) for gdx_file in all_files))
   
    df.index = [f'draw_{i}' for i in range(df.shape[0])]

    # replace "2020" with "mean" after month 
    df.columns = df.columns.str.replace('JAN_2020', 'JAN_mean')
    df.columns = df.columns.str.replace('FEB_2020', 'FEB_mean')
    df.columns = df.columns.str.replace('MAR_2020', 'MAR_mean')
    df.columns = df.columns.str.replace('APR_2020', 'APR_mean')
    df.columns = df.columns.str.replace('MAY_2020', 'MAY_mean')
    df.columns = df.columns.str.replace('JUN_2020', 'JUN_mean')
    df.columns = df.columns.str.replace('JUL_2020', 'JUL_mean')
    df.columns = df.columns.str.replace('AUG_2020', 'AUG_mean')
    df.columns = df.columns.str.replace('SEP_2020', 'SEP_mean')
    df.columns = df.columns.str.replace('OCT_2020', 'OCT_mean')
    df.columns = df.columns.str.replace('NOV_2020', 'NOV_mean')
    df.columns = df.columns.str.replace('DEC_2020', 'DEC_mean')

    # delete columns that have "2020" becaue they are same with "mean"
    df = df[df.columns.drop(list(df.filter(regex='_2020')))]
    

    df.to_parquet("D:\\~your work path~\\SurrogateNN\\DataCollection\\"+folder[len(folder)-9:len(folder)-1]+".parquet.gzip",  compression="gzip")
   
    print('file saved')
    return df
# %%
# Get all folders in DataCollection
path = r'D:\~ your work path ~\SurrogateNN\DataCollection'

all_folders = glob.glob(os.path.join(path + '/*/'))
print(all_folders)

#%%
# python finds the latest .parquet.gzip
all_parquets = glob.glob(os.path.join(path+"\\total_df_20*.parquet.gzip"))

print(all_parquets)

if len(all_parquets) == 0:

    print("No total_df yet")

    total_df = pd.DataFrame()

else: 

    total_df = max(all_parquets, key=os.path.getctime)

    total_df = pd.read_parquet(total_df) 

    
print ("Latest total_df is:", total_df)

print("Shape of the latest total_df:", total_df.shape)


#%%
for folder in all_folders:

    filename = folder[len(folder)-9:len(folder)-1]
    
    # if .parquet.gzip exist, do nothing; if not, creat a parquet file by get_df_parquet
    if os.path.isfile(filename+".parquet.gzip"):

        print(filename, "File exist")
        
    else:
        print(filename, "File not exist")

        df = get_df_parquet(folder)
        # print("df:", df)
        # append it into total_df
        total_df = total_df.append(df)

        print(filename, "File is created")

# Rename the indexs of total_df
total_df.index = [f'draw_{i}' for i in range(total_df.shape[0])]


print("shape of total_df:", total_df.shape)
print("Total df: ", total_df)

#%%
# Rename total_df according to time YYMMDD
Date = datetime.now().strftime("%Y%m%d") # use ("%Y%m%d-%H%M%S") for hour-minute-second

total_df.to_parquet("D:\\~ your work path ~\\SurrogateNN\\DataCollection\\total_df_"+Date+".parquet.gzip",  compression="gzip")


print("new total_df saved")
