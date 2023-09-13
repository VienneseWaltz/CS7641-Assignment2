import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


from util import report_training_result
import time

def knn_learning(dataset_name, X, y, saveFig=False, verbose=False):
    ############
    # Step 1: Standardize the data - Refer to https://amueller.github.io/aml/
    # 01-ml-workflow/03-preprocessing.html on the importane of bringing all
    # features on the same scale.
    ###########
    std_scaler = preprocessing.StandardScaler()
    minmax_scaler = preprocessing.MinMaxScaler()
    X = minmax_scaler.fit_transform(std_scaler.fit_transform(X))

    #######
    # Step 2: Split 80% of the data into a training set and 20% into a testing set
    ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ####################################################################
    # Step 3: Create and train the KNN classifier
    ####################################################################
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_clf.fit(X_train, y_train)
    y_test_pred = knn_clf.predict(X_test)
    y_train_pred = knn_clf.predict(X_train)
    report_training_result("KNN Classifier", dataset_name, y_train, y_train_pred, y_test,
                           y_test_pred)

    #######################################################################
    # Step 4: Validation curve plotted over hyperparameter n_neighbors
    ########################################################################
    n_neighbors_range = np.arange(1, 120)
    train_score, test_score = validation_curve(knn_clf, X_train, y_train, param_name="n_neighbors",
                                               param_range=n_neighbors_range, cv=5)
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting the mean accuracy scores from training and testing scores
    plt.figure(figsize=(20, 10))
    plt.plot(n_neighbors_range, mean_train_score, label="Training Score", color='b')
    plt.plot(n_neighbors_range, mean_test_score, label="Cross Validation Score", color='g')

    plt.grid()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title('Validation curve for KNN(n_neighbors) for ' + dataset_name, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig('figure/ValidationCurve_knn(n_neighbors)' + '_' + dataset_name + '.png', format='png',
                     dpi=120)
        plt.close(fig1)
    else:
        plt.show()


    #############################
    # Step 5: Plotting Learning Curves
    #############################
    training_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(knn_clf, X_train, y_train, train_sizes=training_sizes,
                                                            cv=5)

    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    plt.figure(figsize=(20, 10))
    plt.plot(train_sizes, mean_train_scores, 'o-', color='r', label='Training score')  # red
    plt.plot(train_sizes, mean_test_scores, 'o-', color='g', label='Cross-validation score')  # green

    plt.grid()
    plt.xlabel('Number of training examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Learning curve for knn for ' + dataset_name, fontsize=16,
              fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig2 = plt.gcf()
        fig2.savefig('figure/LearningCurve_knn' + '_' + dataset_name + '.png', format='png',
                     dpi=120)
        plt.close(fig2)
    else:
        plt.show()


    ###################################################################################
    # Step 6: Define the GridSearchCV() procedure to tune over n_neighbors hyperparameter.
    #         Execute the grid search and summarize the best score and parameters.
    #####################################################################################
    # Define the params to tune
    params_to_tune = {'n_neighbors': n_neighbors_range}

    # Define the grid search procedure. This is running 5-fold validation 120 (n_neighbors) times.
    # kNN model is being fitted and predictions are being made 5 x 120 = 600 times.
    grid_search = GridSearchCV(estimator=knn_clf, param_grid=params_to_tune, scoring='accuracy', cv=5, verbose=1, n_jobs=8)

    # Compute the time it takes to complete the training
    t0 = time.time()

    # Execute the grid search
    grid_knn = grid_search.fit(X_train, y_train)

    t1 = time.time()
    training_time = t1 - t0

    # In 4 decimal places
    print(f"Training completed in {'%.4f' % training_time} seconds for KNN classifier for {dataset_name} dataset")

    # Summarize the best score and parameters
    print("Best: %f using %s" % (grid_knn.best_score_, grid_knn.best_params_))

    if verbose:
        # Summarize all the scores that were evaluated
        ## Adapted the use of GridSearchCV() and the cv_results_ attribute from https://scikit-learn.org/stable
        ## /auto_examples/model_selection/plot_grid_search_digits.html
        means = grid_knn.cv_results_['mean_test_score']
        stds = grid_knn.cv_results_['std_test_score']
        params = grid_knn.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    best_knn_params = grid_knn.best_params_
    print(f'The best KNN parameters for {dataset_name} dataset found are: {best_knn_params}')

    # Compute the inference time it takes to complete testing
    t0 = time.time()
    y_test_pred = grid_knn.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print(f"Inference time on the testing data for SVM classifier for {dataset_name}: {'%.4f' % testing_time} seconds")

    y_train_pred = grid_knn.predict(X_train)

    report_training_result("knn (optimized)", dataset_name,
                           y_train, y_train_pred, y_test, y_test_pred)

    return training_time, testing_time






