import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from util import report_training_result

import time

def neural_network_learning(dataset_name, X, y, saveFig=False, verbose=False):

    #######
    # Step 1: Standardize the features by removing the mean and scaling to unit
    #         variance. fit_transform() used on std_scaler scales the training data
    #         and learns the scaling parameters of that data.
    ######
    std_scaler = preprocessing.StandardScaler()
    X = std_scaler.fit_transform(X)

    #######
    # Step 2: Split 80% of the data into a training set and 20% into a testing set
    ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #########
    # Step 3: Train and fit the neural network with 2 hidden layers of size 5 and 2.
    ########
    nn_clf = MLPClassifier(hidden_layer_sizes=(5,2), max_iter = 1000, random_state=7)
    nn_clf.fit(X_train, y_train)
    y_train_pred = nn_clf.predict(X_train)
    y_test_pred = nn_clf.predict(X_test)

    #nn_test_accuracy = accuracy_score(y_test, y_test_pred)
    #print(f"Test accuracy of neural network is {'%.4f' % (nn_test_accuracy * 100)}")

    report_training_result("Neural network(default)", dataset_name, y_train, y_train_pred, y_test, y_test_pred)



    ###################################################################################
    # Step 4: Validation Curve - Performance of a neural network is extremely sensitive to its
    #         learning rate and regularization parameter (alpha).
    #####################################################################################

    ################
    # Plotting validation curve for neural network using hyperparameter alpha
    #################
    alphas_range = np.logspace(-1, 1, 10)

    # Calculate the accuracy of training and testing set using the "alpha" parameter with 5-fold cross-validation
    train_score, test_score = validation_curve(nn_clf, X_train, y_train, param_name='alpha',
                                               param_range=alphas_range, cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting the mean accuracy scores from training and testing scores
    plt.figure(figsize=(20,10))
    plt.semilogx(alphas_range, mean_train_score, label="Training Score", color='b')
    plt.semilogx(alphas_range, mean_test_score, label="Cross Validation Score", color = 'g')

    plt.grid()
    plt.title('Validation curve for neural network (using Alpha)', fontsize=14, fontweight="bold")
    plt.ylabel('Accuracy')
    plt.xlabel('Alpha (regularization parameter)')
    plt.legend(loc="best")
    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig('figure/ValidationCurve_nn(alpha)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig1)
    else:
        plt.show()

    ##################################################
    # Validation curve using learning rate init range
    ##################################################
    lr_init_range = np.logspace(-8, 1, 8)
    train_score, test_score = validation_curve(nn_clf, X_train, y_train, param_name="learning_rate_init", param_range=lr_init_range,
                                               cv=5)

    # Calculate the mean of training score
    mean_train_score = np.mean(train_score, axis=1)

    # Calculate the mean of testing score
    mean_test_score = np.mean(test_score, axis=1)

    plt.figure(figsize=(20,10))
    plt.semilogx(lr_init_range, mean_train_score, label='Training score')
    plt.semilogx(lr_init_range, mean_test_score, label='Cross-validation score')

    plt.grid()
    plt.title('Validation curve for neural network (using Learning rate)', fontsize=14, fontweight="bold")
    plt.xlabel('Learning rate')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    if saveFig:
        fig2 = plt.gcf()
        fig2.savefig('figure/ValidationCurve_nn(learning_rate)' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig2)
    else:
        plt.show()


    ###############################################################################
    # Step 5: Plot Learning Curve for neural network to find out if the model would benefit from more training data
    ###############################################################################
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(nn_clf, X_train, y_train, train_sizes=train_sizes, cv=5)


    mean_train_scores = np.mean(train_scores, axis=1)

    mean_test_scores = np.mean(test_scores, axis=1)
    plt.figure(figsize=(20,10))
    plt.plot(train_sizes, mean_train_scores, 'o-', label='Training score')
    plt.plot(train_sizes, mean_test_scores, 'o-', label='Cross-validation score')

    plt.grid()
    plt.title('Learning curve for neural network of ' + dataset_name + ' dataset', fontsize=14, fontweight="bold")
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.grid()
    if saveFig:
        fig3 = plt.gcf()
        fig3.savefig('figure/LearningCurve_nn' + '_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig3)
    else:
        plt.show()

    # ##################################################################
    #  Step 6: Grid search using alphas_range and lr_init_range.
    ####################################################################
    hidden_layer_sizes_range = [(10, 8, 6, 2), (10, 5, 2)]
    params_to_tune = {'alpha': alphas_range,
                      'learning_rate_init': lr_init_range,
                      'hidden_layer_sizes': hidden_layer_sizes_range}
    grid_nn = GridSearchCV(nn_clf, param_grid=params_to_tune, scoring='accuracy',
                           cv=5, verbose=1, n_jobs=8)
    t0 = time.time()
    grid_nn.fit(X_train, y_train)
    t1 = time.time()
    training_time = t1 - t0
    print()
    print(f"Training completed in {'%.4f' % training_time} seconds for neural network classifier for {dataset_name} dataset")

    best_nn_params = grid_nn.best_params_
    print(f'The best neural network parameters for {dataset_name} dataset found are: {best_nn_params}')
    print()

    # Grid search scores on training data
    if verbose:
        print(f'Performing grid search scores on training data for {dataset_name}...')
        ## Adapted the use of GridSearchCV() and the cv_results_ attribute from https://scikit-learn.org/stable
        ## /auto_examples/model_selection/plot_grid_search_digits.html
        means = grid_nn.cv_results_['mean_test_score']
        stds = grid_nn.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_nn.cv_results_['params']):
            print(f"{'%0.3f' % (mean)} (+/-{'%0.03f' % (std * 2)}) for {'%r' % (params)}")

    # Compute the inference time it takes to complete testing.
    t0 = time.time()
    y_test_pred = grid_nn.predict(X_test)
    t1 = time.time()
    testing_time = t1 - t0
    print(f"Inference time on the testing data for neural network classifier for {dataset_name}: {'%.4f'% testing_time} seconds")

    y_train_pred = grid_nn.predict(X_train)
    report_training_result("Neural network optimized", dataset_name,
                           y_train, y_train_pred, y_test, y_test_pred)

    ########################################################################################
    # Step 7: Loss Curves - Gives us a snapshot of the training process and the direction
    #                       in which the optimized neural network learns. Observe how training
    #                       error decreases with epochs.
    #########################################################################################
    nn_clf = MLPClassifier(random_state=7, max_iter=1000, warm_start=True)
    nn_clf.set_params(alpha=best_nn_params['alpha'],
                      hidden_layer_sizes=best_nn_params['hidden_layer_sizes'],
                      learning_rate_init=best_nn_params['learning_rate_init'])
    nn_clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    # Plot the loss curve
    plt.plot(nn_clf.loss_curve_)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve for Neural Network of ' + dataset_name + ' dataset', fontweight="bold")
    plt.legend(loc="best")
    if saveFig:
        fig4 = plt.gcf()
        fig4.savefig('figure/LossCurve_neural_network' +'_' + dataset_name + '.png', format='png', dpi=120)
        plt.close(fig4)
    else:
        plt.show()

    return training_time, testing_time











