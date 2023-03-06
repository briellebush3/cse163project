import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plot

# find dataset about world education attainment (15+ year olds)
# merge dataset with education column 
# compare the new education column to the GDP column 
# visualize using scatterplot 

def GLOB_data(file_name):
    data = pd.read_csv(file_name)

def new_data(file_name):
    data = pd.read_csv(file_name)

    df = data[['Country Name', 'Series Name', '1980 [YR1980]', '1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]']]

    # filter for total primary and secondary school for 15+ 
    total_educ = df['Series Name'] = "Barro-Lee: Average years of primary schooling, age 15+, total", "Barro-Lee: Average years of secondary schooling, age 15+, total"
    total_educ = df[total_educ]

# merge the datasets
merged = GLOB_data.merge(total_educ, left_on='', right_on='')


def main(): 
    GLOB_data("GLOB.SES.csv")
    new_data("yrseduc_data2.csv")


if __name__ == '__main__':
    main()