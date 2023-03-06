import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plot
 

def new_data(file_name, file_name_2):
    data = pd.read_csv(file_name)
    data2 = pd.read_csv(file_name_2)

    df = data2[['Country Name', 'Series Name', '1970 [YR1970]', '1980 [YR1980]' , '1990 [YR1990]' , '2000 [YR2000]' , '2010 [YR2010]']]
   

    # filter for total primary and secondary school for 15+ 
    pri = df['Series Name'] == "Barro-Lee: Average years of primary schooling, age 15+, total"
    sec = df['Series Name'] == "Barro-Lee: Average years of secondary schooling, age 15+, total" 
    df = df[pri | sec]
    print(df)

# filter first dataset by country name, GDP, year 
# merge the datasets 
# merged = GLOB_data.merge(total_educ, left_on='', right_on='')


def main(): 
    new_data("GLOB.SES.csv", "yrseduc_data2.csv")
    


if __name__ == '__main__':
    main()