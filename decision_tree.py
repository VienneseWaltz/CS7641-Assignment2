import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from util import report_training_result

import time


def decision_tree_learning(dataset_name, X, y, saveFig=False, verbose=False):

    #######
    # Step 1: Standardize the data that has been read in
    ######
    X = preprocessing.scale(X)

    #######
    # Step 2: Split 80% of the data into a training set and 20% into a testing set
    ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #########
    # Step 3: Train and fit the Decision Tree
    ########
    dt_clf = tree.DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
    dt_clf.fit(X_train, y_train)
    y_test_pred = dt_clf.predict(X_test)
    y_train_pred = dt_clf.predict(X_train)
    report_training_result("Decision tree (default)", dataset_name, y_train, y_train_pred, y_test, y_test_pred)


    ###################################################################################
    # Step 4: Validation Curve - Plots the train_score and test_score over
    #         hyperparameters max_leaf_nodes and min_samples_leaf.
    #####################################################################################

    ################
    # Validation Curve for Decision Tree using max_leaf_nodes hyperparameter
    #################
    # Set the range for the parameter (from 1 to 21)
    max_leaf_nodes_range = np.arange(2, 100, 1)

    # Calculate the accuracy of training and testing set using the "max_leaf_nodes"
    # parameter with 5-fold cross-validation
    train_score, test_score = validation_curve(dt_clf, X_train, y_train, param_name="max_leaf_nodes",
                                               param_range=max_leaf_nodes_range, cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting the mean accuracy scores from training and testing scores
    plt.figure(figsize=(20,10))
    plt.plot(max_leaf_nodes_range, mean_train_score, label="Training Score", color='b')
    plt.plot(max_leaf_nodes_range, mean_test_score, label="Cross Validation Score", color = 'g')

    plt.grid()
    plt.xlabel('Maximum Leaf Nodes')
    plt.ylabel('Accuracy')
    plt.title('Validataion curve for decision tree (Max leaf nodes) for ' + dataset_name, fontweight="bold")
    plt.legend(loc = 'best')
    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig('figure/ValidationCurve_dt(max_leaf_nodes)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig1)
    else:
        plt.show()


    ###################
    # Validation Curve for Decision Tree using min_samples_leaf hyperparameter
    ###################
    # Set the range for the parameter (from 2 to 21)
    #min_samples_split_range = np.arange(2, 100, 1)
    min_samples_leaf_range = np.arange(2, 100, 1)

    # Calculate the accuracy of training and testing set using the "min_samples_leaf"
    # parameter with 5-fold cross-validation
    train_score, test_score = validation_curve(dt_clf, X_train, y_train,param_name="min_samples_leaf",
                                               param_range=min_samples_leaf_range, cv=5)
    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    plt.figure(figsize=(20,10))
    plt.plot(min_samples_leaf_range, mean_train_score, label='Training score', color='c')         # cyan
    plt.plot(min_samples_leaf_range, mean_test_score, label='Cross-validation score', color='m')  # magenta

    plt.grid()
    plt.xlabel('Minimum Samples at Leaf Nodes')
    plt.ylabel('Accuracy')
    plt.title('Validataion curve for decision tree(Min samples leaf) for ' + dataset_name, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig2 = plt.gcf()
        fig2.savefig('figure/ValidationCurve_dt(min_samples_leaf)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig2)
    else:
        plt.show()


    #######################################################################################
    # Step 5: Use Learning curve to find out if the model (decision tree) is too simple
    #         (biased) or would benefit from more data
    #######################################################################################
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(dt_clf, X_train, y_train, train_sizes=train_sizes, cv=5)
    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    plt.figure(figsize=(20,10))
    plt.plot(train_sizes, mean_train_scores, 'o-', color='r', label='Training score')            # red
    plt.plot(train_sizes, mean_test_scores, 'o-', color='g', label='Cross-validation score')    # green

    # Configure grid lines
    plt.grid()

    plt.xlabel('Number of training examples')
    plt.ylabel('Score')
    plt.title('Learning curve for decision tree for ' + dataset_name, fontweight="bold")
    plt.legend(loc='best')
    if saveFig:
        fig3 = plt.gcf()
        fig3.savefig('figure/LearningCurve_dt' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig3)
    else:
        plt.show()


    ######################################################################################
    # Step 6: Use GridSearchCV() to loop through max_leaf_nodes, min_samples_leaf and
    #         ccp_alpha. Getting the ccp_alpha means essentially we get the complexity
    #         parameter for minimal cost complexity pruning (i.e. post-pruning).
    ######################################################################################
    pruning_path = dt_clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas_range = pruning_path['ccp_alphas']
    params_to_tune = {'max_leaf_nodes': max_leaf_nodes_range,
                      'min_samples_leaf': min_samples_leaf_range,
                      'ccp_alpha':ccp_alphas_range}
    grid_dt = GridSearchCV(estimator=dt_clf, param_grid=params_to_tune, scoring='accuracy', cv=5, verbose=1, n_jobs=8)

    # Compute the time it takes to complete training
    t0 = time.time()
    grid_dt.fit(X_train, y_train)
    t1 = time.time()
    training_time = t1 - t0

    # In 4 decimal places
    print(f"Training completed in {'%.4f' % training_time} seconds for decision tree classifier for {dataset_name} dataset")

    best_dt_params = grid_dt.best_params_
    print(f'The best decision tree parameters for {dataset_name} dataset found are: {best_dt_params}')

    # Grid search scores on training data
    if verbose:
        print(f'Performing grid search scores on training data for {dataset_name}...')
        ## Adapted the use of GridSearchCV() and the cv_results_ attribute from https://scikit-learn.org/stable
        ## /auto_examples/model_selection/plot_grid_search_digits.html
        means = grid_dt.cv_results_['mean_test_score']
        stds = grid_dt.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_dt.cv_results_['params']):
            print(f"{'%0.3f' % (mean)} (+/-{'%0.03f' % (std * 2)}) for {'%r' % (params)}")


    # Compute the inference time it takes to complete testing.
    t0 = time.time()
    y_test_pred = grid_dt.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print(f"Inference time on the testing data for decision tree classifier for {dataset_name}: {'%.4f'% testing_time} seconds")

    y_train_pred = grid_dt.predict(X_train)
    report_training_result("Decision tree optimized", dataset_name,
                           y_train, y_train_pred, y_test, y_test_pred)

    return training_time, testing_time













