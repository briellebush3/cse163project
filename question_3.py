import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def plot_top_bottom_10_1910(file_name):
    data = pd.read_csv('/Users/davidpark/Desktop/FinalProject_dataset/GLOB.SES.csv')
    data = data.dropna()
    yrs1910 = data['year'] == 1910
    data_1 = data[yrs1910]
    data_1 = data_1.sort_values(by=['SES'], ascending=False)
    
    x = data_1.loc[:, ["SES"]]
    data_1['ses_m'] = (x-x.mean())
    data_1['colors'] = ['red' if x<0 else 'green' for x in data_1['ses_m']]
    data_1.sort_values('ses_m', inplace=True)
    data_1.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_1.index, xmin=0, xmax=data_1.ses_m, color=data_1.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_1.ses_m, data_1.index, data_1.ses_m):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_1.index, data_1.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1910', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def plot_top_bottom_10_1940(file_name):
    data = pd.read_csv('/Users/davidpark/Desktop/FinalProject_dataset/GLOB.SES.csv')
    data = data.dropna()
    yrs1940 = data['year'] == 1940
    data_1 = data[yrs1940]
    data_1 = data_1.sort_values(by=['SES'], ascending=False)
    
    x = data_1.loc[:, ["SES"]]
    data_1['ses_m'] = (x-x.mean())
    data_1['colors'] = ['red' if x<0 else 'green' for x in data_1['ses_m']]
    data_1.sort_values('ses_m', inplace=True)
    data_1.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_1.index, xmin=0, xmax=data_1.ses_m, color=data_1.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_1.ses_m, data_1.index, data_1.ses_m):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_1.index, data_1.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1920', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def plot_top_bottom_10_1970(file_name):
    data = pd.read_csv('/Users/davidpark/Desktop/FinalProject_dataset/GLOB.SES.csv')
    data = data.dropna()
    yrs1970 = data['year'] == 1970
    data_1 = data[yrs1970]
    data_1 = data_1.sort_values(by=['SES'], ascending=False)
    
    x = data_1.loc[:, ["SES"]]
    data_1['ses_m'] = (x-x.mean())
    data_1['colors'] = ['red' if x<0 else 'green' for x in data_1['ses_m']]
    data_1.sort_values('ses_m', inplace=True)
    data_1.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_1.index, xmin=0, xmax=data_1.ses_m, color=data_1.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_1.ses_m, data_1.index, data_1.ses_m):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_1.index, data_1.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1920', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def plot_top_bottom_10_2010(file_name):
    data = pd.read_csv('/Users/davidpark/Desktop/FinalProject_dataset/GLOB.SES.csv')
    data = data.dropna()
    yrs2010 = data['year'] == 2010
    data_1 = data[yrs2010]
    data_1 = data_1.sort_values(by=['SES'], ascending=False)
    
    x = data_1.loc[:, ["SES"]]
    data_1['ses_m'] = (x-x.mean())
    data_1['colors'] = ['red' if x<0 else 'green' for x in data_1['ses_m']]
    data_1.sort_values('ses_m', inplace=True)
    data_1.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_1.index, xmin=0, xmax=data_1.ses_m, color=data_1.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_1.ses_m, data_1.index, data_1.ses_m):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_1.index, data_1.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1920', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    




def main(): 
    plot_top_bottom_10_1910("GLOB.SES.csv")
    plot_top_bottom_10_1940("GLOB.SES.csv")
    plot_top_bottom_10_1970("GLOB.SES.csv")
    plot_top_bottom_10_2010("GLOB.SES.csv")
    


if __name__ == '__main__':
    main()
     