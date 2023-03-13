"""
Brielle Bush and Jihoon Park
CSE 163 AB
This program uses the global socioeconomic status scores from 1880 to
2010, unemployment rate and years of education of primary and secondary
schooling. All the data files are in CSV format to investigate global socioeconomic
scores with gdppc, years of education, and unemployment rate statistics.
Each row in the dataset corresponds to each country for gdppc, years of
education. This program implements each function to manipulate and extract
a particular of dataset and plot graphs to visualize the results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def plot_top_bottom_10_1910(file_name):
    """
    Takes a CSV file and filters the given dataframe with
    certain conditions in 1910. Calculate the average value of the
    SES numerical value of all countries and subtract the SES score
    of every country from the average value.
    Plot a bar graph to see which country ranks high and bottom position.
    Returns top and bottom 5 countries as a list.
    """
    data = pd.read_csv(file_name)
    data = data.dropna()
    yrs1910 = data['year'] == 1910
    data_1 = data[yrs1910]
    data_1 = data_1.sort_values(by=['SES'], ascending=False)
    top5 = data_1.iloc[0:5, 2].values
    bottom5 = data_1.iloc[-6:-1, 2].values
    t_b_1910 = list(top5) + list(bottom5)

    x = data_1.loc[:, ["SES"]]
    data_1['ses_m'] = (x-x.mean())
    data_1['colors'] = ['red' if x < 0 else 'green' for x in data_1['ses_m']]
    data_1.sort_values('ses_m', inplace=True)
    data_1.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_1.index, xmin=0, xmax=data_1.ses_m,
               color=data_1.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_1.ses_m, data_1.index, data_1.ses_m):
        plt.text(x, y, round(tex, 2),
                 horizontalalignment='right' if x < 0 else 'left',
                 verticalalignment='center',
                 fontdict={'color': 'red' if x < 0 else 'green', 'size': 10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_1.index, data_1.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1910', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    return t_b_1910


def plot_top_bottom_10_1940(file_name):
    """
    Takes a CSV file and filter the given dataframe with a
    certain conditions in 1940. Calculate the average value of the
    SES numerical value of all countries and subtract a SES score
    of every counry from the average value.
    Plot a bar graph to see which country rank high and bottom position.
    Returns top and bottom 5 countries as a list.
    """
    data = pd.read_csv(file_name)
    data = data.dropna()
    yrs1940 = data['year'] == 1940
    data_2 = data[yrs1940]
    data_2 = data_2.sort_values(by=['SES'], ascending=False)
    top5 = data_2.iloc[0:5, 2].values
    bottom5 = data_2.iloc[-6:-1, 2].values
    t_b_1940 = list(top5) + list(bottom5)

    x = data_2.loc[:, ["SES"]]
    data_2['ses_m'] = (x-x.mean())
    data_2['colors'] = ['red' if x < 0 else 'green' for x in data_2['ses_m']]
    data_2.sort_values('ses_m', inplace=True)
    data_2.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_2.index, xmin=0, xmax=data_2.ses_m,
               color=data_2.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_2.ses_m, data_2.index, data_2.ses_m):
        plt.text(x, y, round(tex, 2),
                 horizontalalignment='right' if x < 0 else 'left',
                 verticalalignment='center',
                 fontdict={'color': 'red' if x < 0 else 'green', 'size': 10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_2.index, data_2.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1940', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    return t_b_1940


def plot_top_bottom_10_1970(file_name):
    """
    Takes a CSV file and filter the given dataframe with a
    certain conditions in 1970. Calculate the average value of the
    SES numerical value of all countries and subtract a SES score
    of every counry from the average value.
    Plot a bar graph to see which country rank high and bottom position.
    Returns top and bottom 5 countries as a list.
    """
    data = pd.read_csv(file_name)
    data = data.dropna()
    yrs1970 = data['year'] == 1970
    data_3 = data[yrs1970]
    data_3 = data_3.sort_values(by=['SES'], ascending=False)
    top5 = data_3.iloc[0:5, 2].values
    bottom5 = data_3.iloc[-6:-1, 2].values
    t_b_1970 = list(top5) + list(bottom5)

    x = data_3.loc[:, ["SES"]]
    data_3['ses_m'] = (x-x.mean())
    data_3['colors'] = ['red' if x < 0 else 'green' for x in data_3['ses_m']]
    data_3.sort_values('ses_m', inplace=True)
    data_3.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_3.index, xmin=0, xmax=data_3.ses_m,
               color=data_3.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_3.ses_m, data_3.index, data_3.ses_m):
        plt.text(x, y, round(tex, 2),
                 horizontalalignment='right' if x < 0 else 'left',
                 verticalalignment='center',
                 fontdict={'color': 'red' if x < 0 else 'green', 'size': 10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_3.index, data_3.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 1970', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    return t_b_1970


def plot_top_bottom_10_2010(file_name):
    """
    Takes a CSV file and filter the given dataframe with a
    certain conditions in 2010. Calculate the average value of the
    SES numerical value of all countries and subtract a SES score
    of every counry from the average value.
    Plot a bar graph to see which country rank high and bottom position.
    Returns top and bottom 5 countries as a list.
    """
    data = pd.read_csv(file_name)
    data = data.dropna()
    yrs2010 = data['year'] == 2010
    data_4 = data[yrs2010]
    data_4 = data_4.sort_values(by=['SES'], ascending=False)
    top5 = data_4.iloc[0:5, 2].values
    bottom5 = data_4.iloc[-6:-1, 2].values
    t_b_2010 = list(top5) + list(bottom5)

    x = data_4.loc[:, ["SES"]]
    data_4['ses_m'] = (x-x.mean())
    data_4['colors'] = ['red' if x < 0 else 'green' for x in data_4['ses_m']]
    data_4.sort_values('ses_m', inplace=True)
    data_4.reset_index(inplace=True)

    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=data_4.index, xmin=0, xmax=data_4.ses_m,
               color=data_4.colors, alpha=0.5, linewidth=5)
    for x, y, tex in zip(data_4.ses_m, data_4.index, data_4.ses_m):
        plt.text(x, y, round(tex, 2),
                 horizontalalignment='right' if x < 0 else 'left',
                 verticalalignment='center',
                 fontdict={'color': 'red' if x < 0 else 'green', 'size': 10})

    plt.gca().set(ylabel='$Country$', xlabel='$SES$')
    plt.yticks(data_4.index, data_4.country, fontsize=10)
    plt.title('Diverging Bars of Country SES: 2010', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    return t_b_2010


def top_countries_ml(file_name):
    """
    Takes a CSV file and filter the common top 7 countries between 1910
    and 2010. Use DecisionTreeRegressor ML model to train and test
    the dataset and calcuate mean squared errors. Plot max_depth
    and error graphs.
    """
    data = pd.read_csv(file_name)

    usa = data['country'] == 'United States'
    nz = data['country'] == 'New Zealand'
    aus = data['country'] == 'Australia'
    swit = data['country'] == 'Switzerland'
    can = data['country'] == 'Canada'
    nor = data['country'] == 'Norway'
    ire = data['country'] == 'Ireland'

    ML_data = data[usa | nz | aus | swit | can | nor | ire]

    ML_data = ML_data[['SES', 'gdppc', 'yrseduc']]
    features = ML_data.loc[:, ML_data.columns != 'SES']
    labels = ML_data['SES']
    model = DecisionTreeRegressor()

    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)

    from sklearn.metrics import mean_squared_error
    train_predictions = model.predict(features_train)
    print('Train mean squared error:', mean_squared_error(
        labels_train, train_predictions))
    test_predictions = model.predict(features_test)
    print('Test mean squared error:', mean_squared_error(
        labels_test, test_predictions))

    short_model = DecisionTreeRegressor(max_depth=4)
    short_model.fit(features_train, labels_train)
    train_predictions = short_model.predict(features_train)
    print('Train mean squared error:', mean_squared_error(
        labels_train, train_predictions))
    test_predictions = short_model.predict(features_test)
    print('Test mean squared error:', mean_squared_error(
        labels_test, test_predictions))

    from IPython.display import Image, display

    import graphviz
    from sklearn.tree import export_graphviz

    def plot_tree(model, features, labels):
        dot_data = export_graphviz(model, out_file=None,
                                   feature_names=features.columns,
                                   class_names=labels.unique(),
                                   impurity=False,
                                   filled=True, rounded=True,
                                   special_characters=True)
        graphviz.Source(dot_data).render('tree.gv', format='png')
        display(Image(filename='/Users/davidpark/cse163_project/'
                      'cse163project/tree.gv.png'))
    plot_tree(short_model, features_train, labels_train)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=42)

    errors = []
    for i in range(1, 20):
        model = DecisionTreeRegressor(max_depth=i, random_state=42)
        model.fit(features_train, labels_train)

        pred_train = model.predict(features_train)
        train_err = mean_squared_error(labels_train, pred_train)

        pred_test = model.predict(features_test)
        test_err = mean_squared_error(labels_test, pred_test)

        errors.append({'max depth': i, 'train error': train_err,
                       'test error': test_err})
    errors = pd.DataFrame(errors)

    def plot_errors(errors, column, name):
        sns.relplot(kind='line', x='max depth', y=column, data=errors)
        plt.title(f'{name} error as Max Depth Changes')
        plt.xlabel('Max Depth')
        plt.ylabel(f'{name} Error')
        plt.show()

    plot_errors(errors, 'train error', 'Train')
    plot_errors(errors, 'test error', 'Test')


def bottom_countries_ml(file_name):
    """
    Takes a CSV file and filter the common bottom 7 countries between 1910
    and 2010. Use DecisionTreeRegressor ML model to train and test the dataset
    and calcuate mean squared errors. Plot max_depth and error graphs.
    """
    data = pd.read_csv(file_name)
    nig = data['country'] == 'Niger'
    ang = data['country'] == 'Angola'
    sud = data['country'] == 'Sudan'
    eth = data['country'] == 'Ethiopia'
    sen = data['country'] == 'Senegal'
    mali = data['country'] == 'Mali'
    mala = data['country'] == 'Malawi'

    ML2_data = data[nig | ang | sud | eth | sen | mali | mala]
    ML2_data = ML2_data[['SES', 'gdppc', 'yrseduc']]

    features = ML2_data.loc[:, ML2_data.columns != 'SES']
    labels = ML2_data['SES']
    model = DecisionTreeRegressor()

    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)

    from sklearn.metrics import mean_squared_error
    train_predictions = model.predict(features_train)
    print('Train mean squared error:', mean_squared_error(
        labels_train, train_predictions))
    test_predictions = model.predict(features_test)
    print('Test mean squared error:', mean_squared_error(
        labels_test, test_predictions))

    short_model = DecisionTreeRegressor(max_depth=4)
    short_model.fit(features_train, labels_train)
    train_predictions = short_model.predict(features_train)
    print('Train mean squared error:', mean_squared_error(
        labels_train, train_predictions))
    test_predictions = short_model.predict(features_test)
    print('Test mean squared error:', mean_squared_error(
        labels_test, test_predictions))

    from IPython.display import Image, display
    import graphviz
    from sklearn.tree import export_graphviz

    def plot_tree(model, features, labels):
        dot_data = export_graphviz(model, out_file=None,
                                   feature_names=features.columns,
                                   class_names=labels.unique(),
                                   impurity=False,
                                   filled=True, rounded=True,
                                   special_characters=True)
        graphviz.Source(dot_data).render('tree2.gv', format='png')
        display(Image(filename='/Users/davidpark/cse163_project/'
                      'cse163project/tree2.gv.png'))
    plot_tree(short_model, features_train, labels_train)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=42)

    errors = []
    for i in range(1, 20):
        model = DecisionTreeRegressor(max_depth=i, random_state=42)
        model.fit(features_train, labels_train)

        pred_train = model.predict(features_train)
        train_err = mean_squared_error(labels_train, pred_train)

        pred_test = model.predict(features_test)
        test_err = mean_squared_error(labels_test, pred_test)

        errors.append({'max depth': i, 'train error': train_err,
                       'test error': test_err})
    errors = pd.DataFrame(errors)

    def plot_errors(errors, column, name):
        sns.relplot(kind='line', x='max depth', y=column, data=errors)
        plt.title(f'{name} error as Max Depth Changes')
        plt.xlabel('Max Depth')
        plt.ylabel(f'{name} Error')
        plt.show()
    plot_errors(errors, 'train error', 'Train')
    plot_errors(errors, 'test error', 'Test')


def main():
    plot_top_bottom_10_1910("GLOB.SES.csv")
    plot_top_bottom_10_1940("GLOB.SES.csv")
    plot_top_bottom_10_1970("GLOB.SES.csv")
    plot_top_bottom_10_2010("GLOB.SES.csv")

    top_countries_ml("GLOB.SES.csv")
    bottom_countries_ml("GLOB.SES.csv")


if __name__ == '__main__':
    main()
