# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np



def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared N1xD
    :param X_train: set of objects compared against param X N2xD
    Functions calculates Hamming distances between all objects from set X  and all object from set X_train.
    Resulting distances are returned as matrices.
    :return: Distance matrix between objects X and X_train N1xN2
    """
    #N1 = X.shape[0]
    #N2 = X_train.shape[0]
    #D = X.shape[1]

    # below function is not efficient

    #hamming_distance_matrix = []
    #for i in range(0, N1):
    #    hamming_distance_matrix_row = []
    #    for j in range(0, N2):
    #        hamming_distance_value = 0
    #        for k in range(0, D):
    #            if X[i,k] != X_train[j,k]:
    #                hamming_distance_value += 1
    #        hamming_distance_matrix_row.append(hamming_distance_value)
    #    hamming_distance_matrix.append(hamming_distance_matrix_row)
    #return np.array(hamming_distance_matrix)

    hamming_distance_matrix = []
    for x in X:
        hamming_distance_row = []
        for x_train in X_train:
            hamming_distance = 0
            x_non_zero = x.nonzero()[1]
            x_train_non_zero = x_train.nonzero()[1]
            for i in x_non_zero:
                thereIs = False
                for j in x_train_non_zero:
                    if i == j:
                        thereIs = True
                if thereIs == False:
                    hamming_distance += 1
            for j in x_train_non_zero:
                thereIs = False
                for i in x_non_zero:
                    if i == j:
                        thereIs = True
                if thereIs == False:
                    hamming_distance += 1
            hamming_distance_row.append(hamming_distance)
        hamming_distance_matrix.append(hamming_distance_row)
    return np.array(hamming_distance_matrix)


def sort_train_labels_knn(Dist, y):
    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objests X and X_train N1xN2
    :param y: N2-element vector of labels
    :return: Matrix of sorted class labels ( use metgesort algorithm)
    """
    sorted_labels = []
    for dist in Dist:
        topics_order = []
        for value in dist:
            min_value = dist.min()
            min_index = 0
            while True:
                if dist[min_index] == min_value:
                    topics_order.append(y[min_index])
                    #set distance to infinity
                    dist[min_index] = 50000000
                    break
                min_index += 1
        sorted_labels.append(topics_order)
    return np.array(sorted_labels)


def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set N1xN2
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """
    numberOfTopics = y.max()
    N1 = y.shape[0]
    probabilities_matrix = []
    for i in range(0, N1):
        row = []
        for j in range(1, numberOfTopics + 1):
            probability_value = 0
            for l in range(0, k):
                if (y[i][l] == j):
                    probability_value += 1
            row.append(probability_value / k)
        probabilities_matrix.append(row)
    return np.array(probabilities_matrix)

def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels 1xN.
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """
    Nval = y_true.shape[0]
    numberOfTopics = p_y_x.shape[1]
    sum_n = 0
    for i in range(0, Nval):
        for j in range(0, numberOfTopics):
            if p_y_x[i][j] == p_y_x[i].max():
                prediction = j + 1
        if (prediction != y_true[i]):
            sum_n += 1
    return (sum_n / Nval)


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data N1xD
    :param Xtrain: training data N2xD
    :param yval: class labels for validation data 1xN1
    :param ytrain: class labels for training data 1xN2
    :param k_values: values of parameter k that are going to be evaluated
    :return: function makes model selection with knn and results tuple best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """
    # cos nie dziala. Moze funkcja sortujaca? Przyjrzec sie temu.
    # moze reverse engineering pomoze
    errors = []
    best_error = np.inf
    Dist = hamming_distance(Xval, Xtrain)
    y = sort_train_labels_knn(Dist, ytrain)
    for k in k_values:
        print("DEBUG, k in model_selection_knn: ", k)
        p_y_x = p_y_x_knn(y, k)
        current_error = classification_error(p_y_x, yval)
        errors.append(current_error)
        if current_error < best_error:
            best_error = current_error
            best_k = k
    return (best_error, best_k, errors)


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data 1xN
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities 1xM
    """
    N = ytrain.shape[0]
    k = max(ytrain)
    p_y_array = []
    for j in range(0, k):
        sum = 0
        for i in range(0, N):
            if ytrain[i] == j + 1:
                sum += 1
        p_y = sum / N
        p_y_array.append(p_y)
    return np.array(p_y_array)


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data NxD
    :param ytrain: class labels for training data 1xN
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculated probality p(x|y) assuming that x takes binary values and elements
    x are independent from each other. Function returns matrix p_x_y that has size MxD.
    """
    #print(ytrain[0])
    #print(Xtrain[0].getcol(1).toarray()[0][0])
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    K = max(ytrain)
    p_x_y = []
    for k in range(0, K):
        p_x_y_row = []
        for d in range(0, D):
            sum_up = 0
            sum_down = 0
            for n in range(0, N):
                if ytrain[n] == k+1 and Xtrain[n].getcol(d).toarray()[0][0] == True:
                    sum_up += 1
                if ytrain[n] == k+1:
                    sum_down += 1
            sum_up += a - 1
            sum_down += a + b - 2
            theta = sum_up/sum_down
            if type(theta) is np.ndarray:
                p_x_y_row.append(theta[0])
            else:
                p_x_y_row.append(theta)
        p_x_y.append(p_x_y_row)
    return np.array(p_x_y)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities 1xM
    :param p_x_1_y: probability distribution p(x=1|y) - matrix MxD
    :param X: data for probability estimation, matrix NxD
    :return: function calculated probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
     Function returns matrixx p_y_x of size NxM.
    """
    p_y_x = []
    N = X.shape[0]
    M = p_y.shape[0]
    D = X.shape[1]

    for n in range(0, N):
        p_y_x_row = []
        sum = 0
        p_x_y_array = []
        for m in range(0, M):
            p_x_y = 1
            for d in range(0, D):
                p_x_y *= p_x_1_y[m, d]**X[n,d]*(1-p_x_1_y[m, d])**(1-X[n,d])
            p_x_y_array.append(p_x_y)
            sum += p_x_y*p_y[m]
        for m in range(0, M):
            p_y_x_row.append(p_x_y_array[m]*p_y[m]/sum)
        p_y_x.append(p_y_x_row)
    return np.array(p_y_x)
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training setN2xD
    :param Xval: validation setN1xD
    :param ytrain: class labels for training data 1xN2
    :param yval: class labels for validation data 1xN1
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function makes a model selection for Naive Bayes - that is selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors for each pair (a,b)
    """
    error_best = np.inf
    errors = []
    for a in a_values:
        errors_row = []
        for b in b_values:
            current_error = classification_error(p_y_x_nb(estimate_a_priori_nb(ytrain), estimate_p_x_y_nb(Xtrain, ytrain, a, b), Xval), yval)
            errors_row.append(current_error)
            if current_error < error_best:
                error_best = current_error
                best_a = a
                best_b = b
        errors.append(errors_row)
    return (error_best, best_a, best_b, np.array(errors))
    pass
