import seaborn as sea
import pandas as pd
import matplotlib.pyplot as plot


def main(): 
    read_data("GLOB.SES.csv")

def read_data(file_name):
    data = pd.read_csv(file_name)
    print(data)

if __name__ == '__main__':
    main()
     

# What impact does socioeconomic status have on national development?
# To find this, we will look at the socioeconomic score of each country and compare it to the GDP per capita of that country. 