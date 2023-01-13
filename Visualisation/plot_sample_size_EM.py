
#%%
import os
from tkinter import font
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
#%%
# load test and train data and predictions for later calculation
path = r'D:\~your work path~\SurrogateNN\SampleSize'
os.chdir(path)
print("Current Working Directory " , os.getcwd())
models = ["MLP", "ResNet", "LSTM", "BiLSTM", ]


#%%
# Define the figure size
fig = plt.figure(figsize=(13, 18))

# Basic setting of the figure
sns.set()
# sns.set(font_scale = 2.0)
sns.set_theme(style="whitegrid")
## If you want to use colors
# Palette= sns.color_palette("colorblind", n_colors=4)
# sns.set_palette(Palette)

# plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Arial"



ax=plt.subplot2grid((3,2),(0,0))

df = pd.read_excel('results.xlsx', sheet_name = 'R2')

markers = ["o", "v", "s", "D"]
line_styles = ["--", ":", "-.", "-"]

for i, j, k in zip(models, markers, line_styles): 
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)
    
ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("$\mathregular{R}^\mathregular{2}$", loc="top", fontsize=14)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title('(a)', y=-0.17)
plt.ylim(0, 1)


ax=plt.subplot2grid((3,2),(0,1))

df = pd.read_excel('results.xlsx', sheet_name = 'RMSE')

for i, j, k in zip(models, markers, line_styles): 
    sns.set_theme(style="whitegrid") 
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)
    
ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("$\mathregular{RMSE}$", loc="top", fontsize=14)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.title('(b)', y=-0.17)
plt.ylim(-0.01, 0.18)



ax=plt.subplot2grid((3,2),(1,0))

df = pd.read_excel('results.xlsx', sheet_name = 'APE')

for i, j, k in zip(models, markers, line_styles): 
    sns.set_theme(style="whitegrid") 
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)
    
ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("$\mathregular{APE}_\mathregular{relationship}$", loc="top", fontsize=14)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.title('(c)', y=-0.17)
plt.ylim(-0.01, 1.1)




ax=plt.subplot2grid((3,2),(1,1))

df = pd.read_excel('results.xlsx', sheet_name = 'A1')

for i, j, k in zip(models, markers, line_styles):
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)

ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("$\mathregular{Accuracy}_\mathregular{corner}$", loc="top", fontsize=14)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

plt.title('(d)', y=-0.17)
plt.ylim(0, 1.1)



ax=plt.subplot2grid((3,2),(2,0))

df = pd.read_excel('results.xlsx', sheet_name = 'A2')

for i, j, k in zip(models, markers, line_styles):
    sns.set_theme(style="whitegrid")    
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)

ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("$\mathregular{Accuracy}_\mathregular{constraint}$", loc="top", fontsize=14)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.title('(e)', y=-0.17)
plt.ylim(0.8, 1.01)
# plt.legend(loc='lower right')


ax=plt.subplot2grid((3,2),(2,1))
# ax=plt.subplot2grid((3,8),(2,0),colspan=4)
# plt.subplot(3, 2, 6)
#ax = plt.subplot(gs[2,0], )

df = pd.read_excel('results.xlsx', sheet_name = 'score')

for i, j, k in zip(models, markers, line_styles):
    sns.set_theme(style="whitegrid") 
    ax = sns.lineplot(x="sample size", y= i, marker=j, markersize=8, data=df, label= i, color = "dimgrey", linestyle=k)

ax.set_xlabel("Size of training set", loc="right")
ax.set_ylabel("Total score", loc="top", fontsize=14)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title('(f)', y=-0.17)
plt.ylim(0, 3.2)


plt.show()
fig.savefig("Plots_EM_SampleSize.png",dpi=300)

# %%

