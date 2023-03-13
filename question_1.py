"""
Brielle Bush and Jihoon Park
CSE 163 AB
This program uses global socioeconomic status score from 1880 to
2010, unemployment rate and years of education of primary and secondary
schooling. All the data files is CSV format to investigate global socioeconmic
score with gdppc, years of eduction and unemployment rate statistics.
Each row in the dataset corresponds to each country for gdppc,
years of education. This program implements each functions to manipulate
and extract a particular of datasets and plot graphs to visualize the results.
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def sub_q1(file_name):
    """
    Takes a CSV file and filter some columns that will be used in the
    future computing. Compare average values of SES and gdppc over the
    same period and plot scatter graph to visualize it.
    Also, compute the slope of the regression line and R-squared value.
    Returns the top and bottom 10 countries of the merged dataset.
    """
    data = pd.read_csv(file_name)
    df = data[['country', 'year', 'SES', 'gdppc', 'yrseduc']]
    yrs1900 = df['year'] >= 1900
    yrs1900 = df[yrs1900]

    ses_by_coun_1900 = yrs1900.groupby('country')['SES'].mean()
    gdppc_by_coun = yrs1900.groupby('country')['gdppc'].mean()
    merged = ses_by_coun_1900.to_frame().merge(gdppc_by_coun.to_frame(),
                                               left_on='country',
                                               right_on='country')
    merged = merged.dropna()

    sorted_gdppc = merged.sort_values(by='gdppc', ascending=False)
    top10 = sorted_gdppc.iloc[0:10, :]
    bottom10 = sorted_gdppc.iloc[-11:-1, :]
    top10_list = top10.index.tolist()
    bottom10_list = bottom10.index.tolist()
    t_b_list = top10_list + bottom10_list

    sns.set()

    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(
        merged['gdppc'], merged['SES'])
    result_1 = stats.linregress(merged['gdppc'], merged['SES'])
    print(f"R-squared: {result_1.rvalue**2:.3f}")

    x_1 = top10['gdppc'].values
    y_1 = top10['SES'].values
    types = top10.reset_index()['country'].values

    x_2 = bottom10['gdppc'].values
    y_2 = bottom10['SES'].values
    types = bottom10.reset_index()['country'].values

    fig, ax = plt.subplots()
    ax.scatter(x_1, y_1)
    ax.scatter(x_2, y_2)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x_1[i], y_1[i]), xytext=(10, -10),
                    fontsize=7, color='r', textcoords='offset points')
        plt.scatter(x_1, y_1)
        ax.annotate(txt, (x_2[i], y_2[i]), xytext=(10, -10),
                    fontsize=7, textcoords='offset points')
        plt.scatter(x_2, y_2)

    sns.regplot(x='gdppc', y='SES', data=merged,
                line_kws={'label': "y={0:.1f}x+{1:.1f}".
                          format(slope_1, intercept_1)})

    plt.title("Trends in SES in each country over GDP")
    plt.xlabel('GDP per capita')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('GDPPC vs SES.png')
    return t_b_list


def sub_q2(file_name):
    """
    Takes a CSV file and filter some columns that will be used in
    the future computing. Compare average values of SES and years of
    education over the same period and plot scatter graph to visualize it.
    Also, compute the slope of the regression line and R-squared value.
    Returns adults' years of education countries that already filtered.
    """
    data = pd.read_csv(file_name)
    df = data[['country', 'year', 'SES', 'gdppc', 'yrseduc']]
    yrseduc_by_coun = df.groupby('country')['yrseduc'].mean()
    sub_q2.ses_by_coun = df.groupby('country')['SES'].mean()
    merged2 = yrseduc_by_coun.to_frame().merge(sub_q2.ses_by_coun.
                                               to_frame(),
                                               left_on='country',
                                               right_on='country')
    merged2 = merged2.dropna()

    slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(
        merged2['yrseduc'], merged2['SES'])
    result_2 = stats.linregress(merged2['yrseduc'], merged2['SES'])
    print(f"R-squared: {result_2.rvalue**2:.3f}")

    over_19 = merged2['yrseduc'] > 3
    over_19 = merged2[over_19]
    over_19_list = over_19.index.tolist()
    x = over_19['yrseduc'].values
    y = over_19['SES'].values
    types = over_19.reset_index()['country'].values

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x[i], y[i]), xytext=(10, -10),
                    fontsize=7, color='r', textcoords="offset points")
        plt.scatter(x, y)
    sns.regplot(x='yrseduc', y='SES', data=merged2,
                line_kws={'label': "y={0:.1f}x+{1:.1f}".
                          format(slope_2, intercept_2)})

    plt.title("Trends in SES in each country over years of education")
    plt.xlabel('Years of education')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('Yrseduc vs SES.png')
    # plt.show()
    return over_19_list


def sub_q3(file_name, ses_data):
    """
    Takes a CSV file and a filtered dataframe from sub_q2 function.
    Compare average values of SES and unemployment rate over the same
    period and plot scatter graph to visualize it.
    Also, compute the slope of the regression line and R-squared value.
    Returns the filtered dataset countries to list.
    """
    data2 = pd.read_csv(file_name)
    data2 = data2.dropna()
    data2 = data2.groupby('Country Name')[['1991', '1992', '1993',
                                           '1994', '1995', '1996',
                                           '1997', '1998', '1999',
                                           '2000', '2001', '2002',
                                           '2003', '2004', '2005',
                                           '2006', '2007', '2008',
                                           '2009', '2010']].sum().mean(axis=1)

    merged3 = pd.DataFrame(data2).join(ses_data)
    merged3 = merged3.dropna()
    merged3.rename(columns={0: 'Unemployment Rate'}, inplace=True)

    less_5_per = merged3['Unemployment Rate'] < 5
    less_5_per = merged3[less_5_per]
    less_5_per_list = less_5_per.index.tolist()

    slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(
        merged3['Unemployment Rate'], merged3['SES'])
    result_3 = stats.linregress(merged3['Unemployment Rate'], merged3['SES'])
    print(f"R-squared: {result_3.rvalue**2:.3f}")

    x_3 = less_5_per['Unemployment Rate'].values
    y_3 = less_5_per['SES'].values
    types = less_5_per.reset_index()['Country Name'].values

    fig, ax = plt.subplots()
    ax.scatter(x_3, y_3)

    for i, txt in enumerate(types):
        ax.annotate(txt, (x_3[i], y_3[i]), xytext=(10, -10),
                    fontsize=7, color='r', textcoords='offset points')
        plt.scatter(x_3, y_3)

    sns.regplot(x='Unemployment Rate', y='SES', data=merged3,
                line_kws={'label': "y={0:.1f}x+{1:.1f}".
                          format(slope_3, intercept_3)})

    plt.title("Trends in SES in each country over Unemployment Rate")
    plt.xlabel('Unemployment Rate')
    plt.ylabel('Socioeconomic status score')
    plt.savefig('Unemployment Rate vs SES.png')
    plt.show()
    return less_5_per_list


def main():
    sub_q1("GLOB.SES.csv")
    sub_q2("GLOB.SES.csv")
    sub_q3('global_unemployment_rate.csv', sub_q2.ses_by_coun)


if __name__ == '__main__':
    main()
