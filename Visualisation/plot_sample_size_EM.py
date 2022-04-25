
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
path = r'N:\agpo\work2\MindStep\SurrogateNN\SampleSize'
os.chdir(path)
print("Current Working Directory " , os.getcwd())
models = ["MLP", "ResNet", "LSTM", "BiLSTM", ]


#%%
fig = plt.figure(figsize=(16, 20))

# gs = gridspec.GridSpec(3, 3, figure = fig)

sns.set()
sns.set(font_scale = 1.5)
Palette= sns.color_palette("colorblind", n_colors=4)
sns.set_palette(Palette)


plt.rcParams["font.family"] = "Arial"

ax=plt.subplot2grid((3,2),(0,0))
# ax = plt.subplot(gs[0, 0])
# plt.subplot(2, 2, 1)
df = pd.read_excel('results.xlsx', sheet_name = 'R2')

for i in models: 
    
    ax = sns.lineplot(x="sample size", y= i, data=df, label= i, marker="o")
    
ax.set_xlabel("Sample size", loc="right")
ax.set_ylabel("$\mathregular{R}^\mathregular{2}$", loc="top")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title('(a)', y=-0.17)
plt.ylim(0, 1)
# plt.legend(loc='lower right')



# plt.subplot(2, 2, 2)
df = pd.read_excel('demo.xlsx', sheet_name = 'APE')

# ax = plt.subplot(gs[0,1])
ax=plt.subplot2grid((3,2),(0,1))

for i in models: 
    ax = sns.lineplot(x="sample size", y= i, data=df, label= i, marker="o")

ax.set_xlabel("Sample size", loc="right")
ax.set_ylabel("$\mathregular{APE}_\mathregular{relationship}$", loc="top")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.title('(b)', y=-0.17)
plt.ylim(-0.01, 0.43)

# plt.legend(loc='upper right')

ax=plt.subplot2grid((3,2),(1,0))
# ax = plt.subplot(gs[1,0])
# plt.subplot(2, 2, 3)
df = pd.read_excel('demo.xlsx', sheet_name = 'A1')

for i in models: 
    ax = sns.lineplot(x="sample size", y= i, data=df, label= i, marker="o")
ax.set_xlabel("Sample size", loc="right")
ax.set_ylabel("$\mathregular{Accuracy}_\mathregular{corner}$", loc="top")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

plt.title('(c)', y=-0.17)
plt.ylim(0, 1.1)
# plt.legend(loc='lower right')



ax=plt.subplot2grid((3,2),(1,1))
# ax = plt.subplot(gs[1,1])
#plt.subplot(2, 2, 4)

df = pd.read_excel('demo.xlsx', sheet_name = 'A3')

for i in models: 
    
    ax = sns.lineplot(x="sample size", y= i, data=df, label= i, marker="o")

ax.set_xlabel("Sample size", loc="right")
ax.set_ylabel("$\mathregular{Accuracy}_\mathregular{constraint}$", loc="top")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
plt.title('(d)', y=-0.17)
plt.ylim(0.8, 1.01)
# plt.legend(loc='lower right')


ax=plt.subplot2grid((3,8),(2,2),colspan=4)
# plt.subplot(3, 2, 6)
#ax = plt.subplot(gs[2,0], )

df = pd.read_excel('demo.xlsx', sheet_name = 'score')

for i in models: 
    ax = sns.lineplot(x="sample size", y= i, data=df, label= i, marker="o")

ax.set_xlabel("Sample size", loc="right")
ax.set_ylabel("Total score", loc="top")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title('(e)', y=-0.17)
plt.ylim(0, 3.2)


plt.show()
fig.savefig("New_plots_2.png",dpi=300)
