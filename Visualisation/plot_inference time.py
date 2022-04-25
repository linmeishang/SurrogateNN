
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
path = r'N:\agpo\work2\MindStep\SurrogateNN\Visualisation'
os.chdir(path)
print("Current Working Directory " , os.getcwd())


# %%
df = pd.read_excel('inference_time.xlsx', sheet_name = 'Inference')

sns.set()
sns.set(font_scale= 1.4)
plt.rcParams["font.family"] = "Arial"

line,ax = plt.subplots() 
plt.figure(figsize=(12,9))
ax = sns.barplot(x="Models", y="Inference_time", data=df, color="dodgerblue")

ax.set_xlabel("NNs", loc = "right")
ax.set_ylabel("Inference time per data point (s)", loc="top", fontsize = 18)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15

plt.ylim(-0.001, 0.055)

for item in ax.get_xticklabels():
    item.set_rotation(45)

for i, v in enumerate(df["Inference_time"].iteritems()):
    ax.text(i ,v[1], "{:,}".format(v[1]), color='black', va ='bottom', rotation=45, size = 12)

plt.savefig("Inference time.png", dpi=300) 


# %%
