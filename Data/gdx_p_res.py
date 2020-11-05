# open gdx file in python
#%%
import pandas as pd
import gdxpds
from gdxpds import gdx
from collections import OrderedDict
import numpy as np
import glob
import os
import time

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
# get_df_pickle is function to read all gdx files in a folder, concate them and store in as a pickle file
def get_df_pickle(folder):

    all_files = glob.glob(os.path.join(folder + "expFarms" + "\\*.gdx"))
    df = pd.concat((get_df(gdx_file) for gdx_file in all_files))
    df.index = [f'draw_{i}' for i in range(df.shape[0])]
    # df.to_pickle(r'N:\\agpo\\work2\\MindStep\\Surrogate_NN\\Data\\folder.pkl')
    df.to_pickle("N:\\agpo\\work2\\MindStep\\Surrogate_NN\\Data\\"+folder[len(folder)-9:len(folder)-1]+".pkl")
    print('file saved')
# %%
# Get all folders in Data
# path = r'N:\agpo\work2\MindStep\Surrogate_NN\Data'
# all_folders_old = glob.glob(os.path.join(path + '/*/'))
#%%
# write a  txt: list of all_folders
# print(all_folders)
with open('all_folders_old.txt', 'w') as f:
    for item in all_folders:
        f.write("%s\n" % item)

for folder in all_folders: 
    get_df_pickle(folder)

# Concate all pickels together as total.pkl
#####
# tomorrow
##### 


#%%
# compare new list and the existing list 

# load the updated all_folders
path = r'N:\agpo\work2\MindStep\Surrogate_NN\Data'
all_folders = glob.glob(os.path.join(path + '/*/'))
print(all_folders)



#%%

# load the existing list from a txt file 
all_folders_old = []

with open('all_folders_old.txt', 'r') as f:
    for line in f:
        currentFolder = line[:-1]
        all_folders_old.append(currentFolder)

print(all_folders_old)

# find the new added folders
new_folders = list(set(all_folders) - set(all_folders_old))

# print(new_folders)

# read new folders and save them into pickle
for folder in new_folders: 
    get_df_pickle(folder)

# concate the new pikles with the existing total.pkl
#####
# tomorrow
##### 



# %%
# store the all_folders at the end
with open('all_folders.txt', 'w') as f:
    for item in all_folders:
        f.write("%s\n" % item)
# %%
