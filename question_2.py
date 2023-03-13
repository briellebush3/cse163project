"""
Brielle Bush and Jihoon Park
CSE 163 AB
This file contains the code that answers and analyzes our
second research question, how does education attainment contribute
to the national development of a country via its GDP per capita?
We merged two datasets to get the most complete data possible
about educational attainment in countries around the world.
We generated a data visualization that clearly demonstrates our results.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def new_data(file_name, file_name_2):
    """
    This function reads in the two datasets we use to answer
    our second research question. We filtered and merged the
    two datasets to get the country name, type of educational
    attainment, and years. We generated a scatter plot to
    analyze our findings.
    """
    data = pd.read_csv(file_name)
    data2 = pd.read_csv(file_name_2)

    df = data2[['Country Name', 'Series Name', '1970 [YR1970]',
                '1980 [YR1980]', '1990 [YR1990]', '2000 [YR2000]',
                '2010 [YR2010]']]

    # filter for total primary and secondary school for 15+
    pri = df['Series Name'] == "Barro-Lee: Average years of primary schooling,\
          age 15+, total"
    sec = df['Series Name'] == "Barro-Lee: Average years of secondary\
        schooling, age 15+, total"
    df = df[pri | sec]

    # filter for country, year, gdppc
    df2 = data[['country', 'year', 'gdppc']]

    # merged two datasets with having common columns (country & Country Name)
    # It's a redunduncy process but extract columns that we only need.
    # If there's a missing column, you can add it.
    merged = df2.merge(df, left_on='country',
                       right_on='Country Name', how='inner')
    merged = merged[['country', 'year', 'gdppc', '1970 [YR1970]',
                     '1980 [YR1980]', '1990 [YR1990]', '2000 [YR2000]',
                     '2010 [YR2010]']]
    # print(merged)

    # make a scatterplot
    sns.relplot(x='year', y='gdppc', data=merged)

    # plt.legend()
    plt.title("Education Attainment Impact on GDP Per Capita (1970-2010)")
    plt.xlabel('Education Attainment By Year')
    plt.ylabel('GDP Per Capita')
    plt.savefig('Educ vs GDPPC.png', bbox_inches='tight')
    plt.show()


def main():
    new_data("GLOB.SES.csv", "yrseduc_data2.csv")


if __name__ == '__main__':
    main()
