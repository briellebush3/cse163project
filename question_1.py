import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def sub_q1_2(file_name):
    data = pd.read_csv(file_name)
    

    df = data[['country', 'year', 'SES', 'gdppc', 'yrseduc']]

    # First factor: gdppc 
    # Mean values of SES and gdppc since 1900 (filter data)
    yrs1900 = df['year'] >= 1900
    yrs1900 = df[yrs1900]

    # Calculate mean values of SES and gdppc since 1990
    ses_by_coun_f = yrs1900.groupby('country')['SES'].mean()
    gdppc_by_coun = yrs1900.groupby('country')['gdppc'].mean()
    merged = ses_by_coun_f.to_frame().merge(gdppc_by_coun.to_frame(), left_on='country', right_on='country')

    # Second factor: educational attainmnet (years)
    # Calculate mean values of SES and yrseduc from entire period
    yrseduc_by_coun = df.groupby('country')['yrseduc'].mean()
    sub_q1_2.ses_by_coun = df.groupby('country')['SES'].mean()
    merged2 = yrseduc_by_coun.to_frame().merge(sub_q1_2.ses_by_coun.to_frame(), left_on='country', right_on='country')
    
    # If the 'yrseduc' column is not removed all rows with missing data,
    # the calculated data doesn't hsow quite a meaningful result.
    # If we drop all NaN values, then it's more informative 
    merged2 = merged2.dropna()

    # Since SES vs gdppc graph shows too many countries, I would select top and bottom 10 countries by showing their country name. 
    sorted_gdppc = merged.sort_values(by='gdppc', ascending=False)
    top10 = sorted_gdppc.iloc[0:10, :]
    bottom10 = sorted_gdppc.iloc[-11:-1, :]

    sns.set()

    # get coeffs of linear fit
    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(merged['gdppc'],merged['SES'])

    # calculuate r^2 value
    # if the value is closer to 1, it's more valid 
    # if the value is closer to 0, the figure doesn't contribute a meaningful result. 
    # 0.421
    result_1 = stats.linregress(merged['gdppc'], merged['SES'])
    print(f"R-squared: {result_1.rvalue**2:.6f}")

    x_1 = top10['gdppc'].values
    y_1 = top10['SES'].values
    types = top10.reset_index()['country'].values

    x_2 = bottom10['gdppc'].values
    y_2 = bottom10['SES'].values
    types = bottom10.reset_index()['country'].values

    fig, ax = plt.subplots()
    ax.scatter(x_1,y_1)
    ax.scatter(x_2,y_2)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x_1[i], y_1[i]), xytext=(10,-10), fontsize=7, color='r', textcoords='offset points')
        plt.scatter(x_1, y_1)
        ax.annotate(txt, (x_2[i], y_2[i]), xytext=(10,-10), fontsize=7, textcoords='offset points')
        plt.scatter(x_2, y_2)

    # use line_kws to set line label for legend
    ax_1 = sns.regplot(x='gdppc', y='SES', data=merged,
                 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope_1,intercept_1)})

    # plot legend
    ax_1.legend()
    plt.title("Trends in SES in each country over GDP")
    plt.xlabel('GDP per capita')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('GDPPC vs SES.png')


    # This is the same process like the above one. 
    # R^2 : 0.264, which is a half of the first one. 
    # So we say educational attainment does not relatviely contribute the higher socioeconmic status to gdp stats. 

    # adult(older than 19 years of age) is the criteria of filtering the data. If all dots is ploted with naming, it's not recognizable. 
    
    slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(merged2['yrseduc'],merged2['SES'])
    result_2 = stats.linregress(merged2['yrseduc'], merged2['SES'])
    print(f"R-squared: {result_2.rvalue**2:.6f}")

    over_19 = merged2['yrseduc'] > 3
    over_19 = merged2[over_19]
    x = over_19['yrseduc'].values
    y = over_19['SES'].values
    types = over_19.reset_index()['country'].values

    fig, ax = plt.subplots()
    ax.scatter(x,y)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x[i], y[i]), xytext=(10, -10), fontsize=7, color='r', textcoords="offset points")
        plt.scatter(x,y)


    # use line_kws to set line label for legend
    ax_2 = sns.regplot(x='yrseduc', y='SES', data=merged2,
                 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope_2,intercept_2)})
    # plot legend
    ax_2.legend()
    plt.title("Trends in SES in each country over years of education")
    plt.xlabel('Years of education')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('Yrseduc vs SES.png')
    plt.show()


    

def sub_q3(file_name, ses_data):
    data2 = pd.read_csv(file_name)

    data2 = data2.dropna() 

    # data2 = data2.sum(axis=1) / (len(data2.columns))

    data2 = data2.groupby('Country Name')[['1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']].sum().mean(axis=1) 

    merged3 = pd.DataFrame(data2).join(ses_data)
    merged3 = merged3.dropna()
    merged3.rename(columns={0:'Unemployment Rate'}, inplace=True)


    less_5_per = merged3['Unemployment Rate'] < 5
    less_5_per = merged3[less_5_per]
    less_5_per

    slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(merged3['Unemployment Rate'],merged3['SES'])
    result_3 = stats.linregress(merged3['Unemployment Rate'], merged3['SES'])
    print(f"R-squared: {result_3.rvalue**2:.6f}")

 
    x_2 = less_5_per['Unemployment Rate'].values
    y_2 = less_5_per['SES'].values
    types = less_5_per.reset_index()['Country Name'].values

    fig, ax = plt.subplots()
    ax.scatter(x_2,y_2)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x_2[i], y_2[i]), xytext=(10,-10), fontsize=7, color='r', textcoords='offset points')
        plt.scatter(x_2, y_2)

    # use line_kws to set line label for legend
    ax_3 = sns.regplot(x='Unemployment Rate', y='SES', data=merged3,
                 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope_3,intercept_3)})
    # plot legend
    ax_3.legend()
    plt.title("Trends in SES in each country over Unemployment Rate")
    plt.xlabel('Unemployment Rate')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('Unemployment Rate vs SES.png')
    plt.show()


def main(): 
    sub_q1_2("GLOB.SES.csv")
    sub_q3('global_unemployment_rate.csv', sub_q1_2.ses_by_coun)


if __name__ == '__main__':
    main()
     

