import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
 

def new_data(file_name, file_name_2):
    data = pd.read_csv(file_name)
    data2 = pd.read_csv(file_name_2)

    df = data2[['Country Name', 'Series Name', '1970 [YR1970]', '1980 [YR1980]' , '1990 [YR1990]' , '2000 [YR2000]' , '2010 [YR2010]']]
   

    # filter for total primary and secondary school for 15+ 
    pri = df['Series Name'] == "Barro-Lee: Average years of primary schooling, age 15+, total"
    sec = df['Series Name'] == "Barro-Lee: Average years of secondary schooling, age 15+, total" 
    df = df[pri | sec]
    print(df)


    # filter for country, year, gdppc
    df2 = data[['country', 'year', 'gdppc']]
    print(df2)

    
    # merged two datasets with having common columns (country & Country Name)
    # It's a redunduncy process but extract columns that we only need. 
    # If there's a missing column, you can add it. 
    merged = df2.merge(df, left_on='country', right_on='Country Name', how='inner')
    merged[['country', 'year', 'gdppc', '1970 [YR1970]', '1980 [YR1980]', '1990 [YR1990]','2000 [YR2000]', '2010 [YR2010]']]
    print(merged)

    year = df['1970 [YR1970]'] + df['1980 [YR1980]'] + df['1990 [YR1990]'] + df['2000 [YR2000]'] + df['2010 [YR2010]']
    print(year)

    # make a scatterplot 
    sns.relplot(x='year', y='gdppc', data=merged)

    #plt.legend()
    plt.title("Education Attainment Impact on GDP Per Capita")
    plt.xlabel('Education Attainment')
    plt.ylabel('GDP Per Capita')
    plt.savefig('Educ vs GDPPC.png')


def main(): 
    new_data("GLOB.SES.csv", "yrseduc_data2.csv")
    


if __name__ == '__main__':
    main()