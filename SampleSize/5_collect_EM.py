#%%
# This file collects the results from all excel files of Evaluation_total_df_
from pandas.core.construction import array
from global_file import * 
import xlsxwriter

os.chdir(path)

print("Current Working Directory " , os.getcwd())

# %%
# list all excel file of EM
excel_files = glob.glob(os.path.join(path, "Evaluation_total_df_*.xlsx"))

# order the list according to the sample size
myorder = [0, 4, 1, 5, 2, 3]
ordered_list = [excel_files[i] for i in myorder]
print(ordered_list)


#%%
index_names = ['R2', 'RMSE', 'APE1', 'APE2', 'APE', 'A1', 'A2']
column_names = ['BiLSTM', 'LSTM', 'MLP', 'ResNet']
sample_size = [1000, 5000, 10000, 50000, 100000, 150000]


#%%
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')

for i,j in zip(range(len(index_names)), index_names):

    df_all = pd.DataFrame()
    print(j)

    for k in ordered_list[0:]:

        # Read the excel 
        df = pd.read_excel(k)
        df = df.iloc[0:7, 1:22]

        # transform df into an array
        arr = np.array(df)
 
        # split the array into 4 sub-arrays
        arr_2 = np.hsplit(arr, 4) 

        # stack the 4 sub-array together  
        arr_3 = np.stack(arr_2, axis=0)
 
        # calculates the average
        arr_4 = np.mean(arr_3, axis = -1)

        arr_5 = np.transpose(arr_4)
        print(arr_5)
        
        # save the average result into a df
        df = pd.DataFrame(data=arr_5, columns = column_names, index= index_names)
        # print(df)

        # find the row for each indicator
        df_all = df_all.append(df.iloc[i])
        
    # add a column of sample size to it
    df_all.insert(loc = 0, column='sample size', value = sample_size)  

    # save the results into an excel sheet
    df_all.to_excel(writer, sheet_name = j)


writer.save()
writer.close()



# %%
