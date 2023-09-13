import pandas as pd


def load_wine_quality_data(filename):
    """
    Loading the wine_quality.csv file
    :param filename: path to the csv file
    :return: X (data) and y (labels)
    """
    data = pd.read_csv(filename)

    # Selecting the rows that have 'good' and 'bad' in column 'quality'
    # and converting those string labels respectively to 1 and 0.
    data.loc[(data.quality == 'good'), 'quality'] = 1
    data.loc[(data.quality == 'bad'), 'quality'] = 0

    X = data.iloc[:, :-1]
    # Explicitly cast the pandas object y to an integer
    y = data.iloc[:, -1].astype(int)

    return X, y


def load_Star3642_balanced_data(filename):
    """
    Loading the Star3642_balanced.csv file
    :param filename: path to the csv file
    :return: X (data) and y (labels)
    """

    # For this dataset, if the target column shows '1', it is a Giant star
    # and a '0' indicates a Dwarf star
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)

    return X, y