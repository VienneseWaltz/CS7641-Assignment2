import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def load_wine_quality_data(filename):
    """
    Loading the wine_quality.csv file
    :param filename: path to the csv file
    :return: X (data) and y (labels)
    """
    data = pd.read_csv(filename)
    data.loc[(data.quality == 'good'), 'quality'] = 1
    data.loc[(data.quality == 'bad'), 'quality'] = 0
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values.astype(int)

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
    # Map column 'SpType' from string to integer.
    mapping = {k: v for v, k in enumerate(data.SpType.unique())}
    data['SpType'] = data.SpType.map(mapping)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values.astype(int)

    return X, y


def report_training_result(model_type, dataset_name, y_train, y_train_pred, y_test, y_test_pred):
    print(f"Classification report of {model_type} for {dataset_name}'s training data")
    print(classification_report(y_train, y_train_pred))
    print(f"Classification report of {model_type} for {dataset_name}'s testing data")
    print(classification_report(y_test, y_test_pred))
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy of {model_type} for {dataset_name}'s training data is {'%.4f' % (train_accuracy)}")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy of {model_type} for {dataset_name}'s testing data is {'%.4f' % (test_accuracy)}")
    
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)
    print(f"{model_type}'s AUC for {dataset_name}'s training data = {round(auc_train, 4)}")
    print(f"{model_type}'s AUC for {dataset_name}'s testing data = {round(auc_test, 4)}")
    