import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from util import report_training_result
import time

def svm_learning(dataset_name, X, y, saveFig=False, verbose=False):

    #######
    # Step 1: Standardize the data that has been read in
    ######
    X = preprocessing.scale(X)


    #######
    # Step 2: Split 80% of the data into a training set and 20% into a testing set
    ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    ####################################################################
    # Step 3: Create and train the SVM classifier with a linear kernel
    ####################################################################

    # Train and fit a SVM classifier with linear kernel
    svm_clf = svm.SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    y_test_pred = svm_clf.predict(X_test)
    y_train_pred = svm_clf.predict(X_train)
    report_training_result("SVM Classifier with linear kernel", dataset_name, y_train, y_train_pred, y_test, y_test_pred)

    # Train and fit a SVM classifier with polynomial kernel
    svm_clf = svm.SVC(kernel='poly')
    svm_clf.fit(X_train, y_train)
    y_test_pred = svm_clf.predict(X_test)
    y_train_pred = svm_clf.predict(X_train)
    report_training_result("SVM Classifier with polynomial kernel", dataset_name, y_train, y_train_pred, y_test,
                           y_test_pred)


    ################
    # Step 4: Validation Curve for SVM Classifier. C is the regularization parameter that represents
    #         misclassification or error term.
    #################
    # Set the range for C, the regularization parameter
    C_range = np.logspace(-3, 3, 7)

    train_score, test_score = validation_curve(svm_clf, X_train, y_train, param_name="C",
                                               param_range=C_range,
                                               cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting the mean accuracy scores from training and testing scores
    plt.figure(figsize=(20, 10))
    plt.semilogx(C_range, mean_train_score, label="Training score", color='b')
    plt.semilogx(C_range, mean_test_score, label="Cross-validation score", color='g')

    plt.grid()
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Validation curve for SVM classifier(C) for ' + dataset_name, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig('figure/ValidationCurve_svm(polynomial_kernel)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig1)
    else:
        plt.show()


    #############################
    # Step 5: Plotting Learning Curves
    #############################
    training_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(svm_clf, X_train, y_train, train_sizes=training_sizes,
                                                            cv=5)

    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    plt.figure(figsize=(20, 10))
    plt.plot(train_sizes, mean_train_scores, 'o-', color='r', label='Training score')  # red
    plt.plot(train_sizes, mean_test_scores, 'o-', color='g', label='Cross-validation score')  # green

    plt.grid()
    plt.xlabel('Number of training examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Learning curve for SVM Classifier(polynomial kernel) for ' + dataset_name, fontsize=16, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig2 = plt.gcf()
        fig2.savefig('figure/LearningCurve_svm(polynomial_kernel)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig2)
    else:
        plt.show()


    ###################################################################################
    # Step 6: Define the GridSearchCV() procedure with the C parameter to tune. Execute
    #         the grid search and summarize the best score and parameters.
    #####################################################################################
    svm_clf = svm.SVC(kernel='linear')

    # Define the C range
    C_range = np.logspace(-2, 1, 10)

    # Define the kernel range
    kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']

    # Define the grid of values to search
    params_to_tune = {'C': C_range,
                      'kernel': kernel_range}

    # Define the grid search procedure
    grid_search = GridSearchCV(estimator=svm_clf, param_grid=params_to_tune, scoring='accuracy', cv=5, verbose=1, n_jobs=8)

    # Compute the time it takes to complete the training
    t0 = time.time()

    # Execute the grid search
    grid_svm = grid_search.fit(X_train, y_train)

    t1 = time.time()
    training_time = t1 - t0

    # In 4 decimal places
    print(f"Training completed in {'%.4f' % training_time} seconds for SVM classifier for {dataset_name} dataset")

    # Summarize the best score and parameters
    print("Best: %f using %s" % (grid_svm.best_score_, grid_svm.best_params_))

    if verbose:
        # Summarize all the scores that were evaluated
        ## Adapted the use of GridSearchCV() and the cv_results_ attribute from https://scikit-learn.org/stable
        ## /auto_examples/model_selection/plot_grid_search_digits.html
        means = grid_svm.cv_results_['mean_test_score']
        stds = grid_svm.cv_results_['std_test_score']
        params = grid_svm.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    best_svm_params = grid_svm.best_params_
    print(f'The best SVM parameters for {dataset_name} dataset found are: {best_svm_params}')

    # Compute the inference time it takes to complete testing
    t0 = time.time()
    y_test_pred = grid_svm.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print(f"Inference time on the testing data for SVM classifier for {dataset_name}: {'%.4f' % testing_time} seconds")

    y_train_pred = grid_svm.predict(X_train)
    report_training_result("SVM Classifier optimized", dataset_name,
                           y_train, y_train_pred, y_test, y_test_pred)

    return training_time, testing_time











