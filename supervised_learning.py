from util import load_wine_quality_data, load_Star3642_balanced_data
from decision_tree import decision_tree_learning
from neural_network import neural_network_learning
from adaboost import adaboost_classifier_learning
from svm import svm_learning
from knn import knn_learning


def supervised_learning():
    model_type = []
    training_time = []
    testing_time = []

    ''''
    # Decision Tree learning
    X, y = load_wine_quality_data("data/wine_quality.csv")
    training_time1, testing_time1 = decision_tree_learning("wine_quality", X, y, saveFig=True, verbose=False)
    X, y = load_Star3642_balanced_data("data/Star3642_balanced.csv")
    training_time2, testing_time2 = decision_tree_learning("star_type", X, y, saveFig=True, verbose=False)
    model_type.append("Decision Tree")
    training_time.append(training_time1 + training_time2)
    testing_time.append(testing_time1 + testing_time2)
    '''

    '''
    # Neural Network learning
    X, y = load_wine_quality_data("data/wine_quality.csv")
    training_time1, testing_time1 = neural_network_learning("wine_quality", X, y, saveFig=True, verbose=False)
    X, y = load_Star3642_balanced_data("data/Star3642_balanced.csv")
    training_time2, testing_time2 = neural_network_learning("star_type", X, y, saveFig=True, verbose=False)
    model_type.append("Neural Network")
    training_time.append(training_time1 + training_time2)
    testing_time.append(testing_time1 + testing_time2)
    '''



    '''
    # AdaBoost Classifier learning
    X, y = load_wine_quality_data("data/wine_quality.csv")
    training_time1, testing_time1 = adaboost_classifier_learning("wine_quality", X, y, saveFig=True, verbose=False)
    X, y = load_Star3642_balanced_data("data/Star3642_balanced.csv")
    training_time2, testing_time2 = adaboost_classifier_learning("star_type", X, y, saveFig=True, verbose=False)
    model_type.append("AdaBoost Classifier")
    training_time.append(training_time1 + training_time2)
    testing_time.append(testing_time1 + testing_time2)
    '''


    # SVM Classifier learning
    X, y = load_wine_quality_data("data/wine_quality.csv")
    training_time1, testing_time1 = svm_learning("wine_quality", X, y, saveFig=True, verbose=False)
    X, y = load_Star3642_balanced_data("data/Star3642_balanced.csv")
    training_time2, testing_time2 = svm_learning("star_type", X, y, saveFig=True, verbose=False)
    model_type.append("SVM Classifier")
    training_time.append(training_time1 + training_time2)
    testing_time.append(testing_time1 + testing_time2)


    '''
    # KNN Classifier learning
    X, y = load_wine_quality_data("data/wine_quality.csv")
    training_time1, testing_time1 = knn_learning("wine_quality", X, y, saveFig=True, verbose=False)
    X, y = load_Star3642_balanced_data("data/Star3642_balanced.csv")
    training_time2, testing_time2 = knn_learning("star_type", X, y, saveFig=True, verbose=False)
    model_type.append("SVM Classifier")
    training_time.append(training_time1 + training_time2)
    testing_time.append(testing_time1 + testing_time2)
    '''


if __name__ == "__main__":
    supervised_learning()