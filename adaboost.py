import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from util import report_training_result
import time

def adaboost_classifier_learning(dataset_name, X, y, saveFig=False, verbose=False):

    #######
    # Step 1: Standardize the data that has been read in
    ######
    X = preprocessing.scale(X)

    #######
    # Step 2: Split 80% of the data into a training set and 20% into a testing set
    ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ####################################################################################################
    # Step 3: Train and fit the AdaBoost classifier. AdaBoost uses Decision Tree as default Classifier.
    #####################################################################################################
    # Create and train AdaBoost classifier object
    aB_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=7)
    aB_clf.fit(X_train, y_train)
    y_test_pred = aB_clf.predict(X_test)
    y_train_pred = aB_clf.predict(X_train)

    # Evaluate the AdaBoost classifier
    report_training_result("AdaBoostClassifier with Decision Tree(default)", dataset_name, y_train, y_train_pred, y_test, y_test_pred)

    ################
    # Step 4: Validation Curve for AdaBoost Classifier using num_estimators hyperparameter
    #################
    # Set the range for the parameter (from 2 to 100)
    num_estimators_range = np.arange(2, 200, 1)

    train_score, test_score = validation_curve(aB_clf, X_train, y_train, param_name="n_estimators",
                                               param_range=num_estimators_range,
                                               cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting the mean accuracy scores from training and testing scores
    plt.figure(figsize=(20, 10))
    plt.plot(num_estimators_range, mean_train_score, label="Training Score", color='b')
    plt.plot(num_estimators_range, mean_test_score, label="Cross Validation Score", color='g')

    plt.grid()
    plt.xlabel('Number of Weak Learners (n_estimators)')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation curve for AdaBoost Classifier for ' + dataset_name, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig('figure/ValidationCurve_aB(n_estimators)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig1)
    else:
        plt.show()


    ################
    # Validation Curve for AdaBoost Classifier using learning_rate
    #################
    lr_range = np.logspace(-5, 0, 6)
    train_score, test_score = validation_curve(aB_clf, X_train, y_train, param_name="learning_rate",
                                               param_range=lr_range,
                                               cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    plt.figure(figsize=(20, 10))
    plt.semilogx(lr_range, mean_train_score, label='Training score', color='c')
    plt.semilogx(lr_range, mean_test_score, label='Cross-validation score', color='m')

    plt.grid()
    plt.title('Validation curve for AdaBoost Classifier (learning_rate) for ' + dataset_name, fontsize=16, fontweight="bold")
    plt.xlabel('Learning rate', fontsize=12)
    plt.ylabel("Classification score", fontsize=12)
    plt.legend(loc="best")
    if saveFig:
        fig2 = plt.gcf()
        fig2.savefig('figure/ValidationCurve_nn(using learning_rate)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig2)
    else:
        plt.show()

    #############################
    # Step 5: Plotting Learning Curves
    #############################
    training_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(aB_clf, X_train, y_train, train_sizes=training_sizes,
                                                            cv=5)

    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    plt.figure(figsize=(20, 10))
    plt.plot(train_sizes, mean_train_scores, 'o-', color='r', label='Training score')  # red
    plt.plot(train_sizes, mean_test_scores, 'o-', color='g', label='Cross-validation score')  # green

    plt.grid()
    plt.xlabel('Number of training examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Learning curve for AdaBoost Classifier for ' + dataset_name, fontsize=16, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig3 = plt.gcf()
        fig3.savefig('figure/LearningCurve_aBc' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig3)
    else:
        plt.show()

    ###################################################################################
    # Step 6: Use GridSearchCV() within a base estimator for the AdaBoostClassifier
    #####################################################################################
    aB_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=7)

    # Define the grid of values to search
    params_to_tune = {'n_estimators': [10,50,250,1000],
                      'learning_rate': [0.01, 0.1]}

    # Define the grid search procedure
    grid_search = GridSearchCV(estimator=aB_clf, param_grid=params_to_tune, scoring='accuracy', cv=5, verbose=1, n_jobs=8)

    # Compute the time it takes to complete the training
    t0 = time.time()

    # Execute the grid search
    grid_dt = grid_search.fit(X_train, y_train)

    t1 = time.time()
    training_time = t1 - t0

    # In 4 decimal places
    print(f"Training completed in {'%.4f' % training_time} seconds for AdaBoost classifier for {dataset_name} dataset")

    # Summarize the best score and parameters
    print("Best: %f using %s" % (grid_dt.best_score_, grid_dt.best_params_))

    if verbose:
        # Summarize all the scores that were evaluated
        ## Adapted the use of GridSearchCV() and the cv_results_ attribute from https://scikit-learn.org/stable
        ## /auto_examples/model_selection/plot_grid_search_digits.html
        means = grid_dt.cv_results_['mean_test_score']
        stds = grid_dt.cv_results_['std_test_score']
        params = grid_dt.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    best_dt_params = grid_dt.best_params_
    print(f'The best decision tree parameters for {dataset_name} dataset found are: {best_dt_params}')

    # Compute the inference time it takes to complete testing
    t0 = time.time()
    y_test_pred = grid_dt.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print(f"Inference time on the testing data for AdaBoost classifier for {dataset_name}: {'%.4f' % testing_time} seconds")

    y_train_pred = grid_dt.predict(X_train)
    report_training_result("AdaBoost Classifier optimized", dataset_name,
                           y_train, y_train_pred, y_test, y_test_pred)

    return training_time, testing_time











