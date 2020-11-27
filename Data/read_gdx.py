# open gdx file in python and automatically add read folders and add it to exitsing dataframe 
# must type in the command "conda activate py36"

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
#%%

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
# get_df is a function to read a single gdx file of a single draw, and transform it into a data frame
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

    all_files = glob.glob(os.path.join(folder + "expFarms" + "\\*.gdx"))
    
    df = pd.concat((get_df(gdx_file) for gdx_file in all_files))
   
    df.index = [f'draw_{i}' for i in range(df.shape[0])]
    
    df.to_parquet("N:\\agpo\\work2\\MindStep\\SurrogateNN\\Data\\"+folder[len(folder)-9:len(folder)-1]+".parquet.gzip",  compression="gzip")
   
    # print('file saved')
    return df
# %%
# Get all folders in Data
path = r'N:\agpo\work2\MindStep\SurrogateNN\Data'

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

#%%
# Rename total_df_new according to time YYMMDD
Date = datetime.now().strftime("%Y%m%d") # use ("%Y%m%d-%H%M%S") for hour-minute-second

print(Date)

total_df.to_parquet("N:\\agpo\\work2\\MindStep\\SurrogateNN\\Data\\total_df_"+Date+".parquet.gzip",  compression="gzip")

print("new total_df saved")

total_df_new = pd.read_parquet("N:\\agpo\\work2\\MindStep\\SurrogateNN\\Data\\total_df_"+Date+".parquet.gzip")

print(total_df_new)
# %%