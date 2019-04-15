# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Logistic regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


import numpy as np

def sigmoid(x):
    '''
    :param x: input vector Nx1
    :return: vector of sigmoid function values calculated for elements x, Nx1
    '''
    sigmoid_values = []
    for input_value in x:
        sigmoid_values.append(1/(1+(np.exp(-input_value))))
    return np.array(sigmoid_values)


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: model parameters Mx1
    :param x_train: training set features NxM
    :param y_train: training set labels Nx1
    :return: function returns tuple (val, grad), where val is a velue of logistic function and grad its gradient over w
    '''
    def partial_derivative(func_arg_first, func_arg_second, func_val_first, func_val_second):
        delta_arg = func_arg_second - func_arg_first
        delta_val = func_val_second - func_val_first
        return delta_val / delta_arg

    def logistic_function(w, x_train, y_train):
        N = x_train.shape[0]
        likelihood = 1;
        for n in range(0, N):
            likelihood *= (sigmoid(np.dot(np.transpose(w), x_train[n])))**y_train[n] * (1 - sigmoid(np.dot(np.transpose(w), x_train[n])))**(1-y_train[n])
        val = -1 / N * np.log(likelihood)
        return val

    value = logistic_function(w, x_train, y_train)

    grad = []
    i = 0
    diff = 0.0000001
    for w_value in w:
    # gradient is just a vector of partial derivatives
        w_modified = list(w)
        w_modified[i] = w_value + 0.5*diff
        w_value_copy = w_value - 0.5*diff
        logistic_function_value = logistic_function(w_modified, x_train, y_train)
        grad_value = partial_derivative(w_value_copy, w_modified[i], value, logistic_function_value)
        grad.append(grad_value*2)
        i += 1
    return (value[0], np.array(grad))


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: objective function that is going to be minimized (call val,grad = obj_fun(w)).
    :param w0: starting point Mx1
    :param epochs: number of epochs / iterations of an algorithm
    :param eta: learning rate
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch
    '''
    w = list(w0)
    func_values = []

    for i in range(0, epochs):
        val,grad = obj_fun(w)
        delta_w = -grad
        w = np.add(w, np.multiply(eta, delta_w))
        val, grad = obj_fun(w)
        func_values.append([val])
    return (w, np.array(func_values))


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: objective function that undergoes optimization. Call val,grad = obj_fun(w,x,y), where x,y indicates mini-batches.
    :param x_train: training data (feature vectors)NxM
    :param y_train: training data (labels) Nx1
    :param w0: starting point Mx1
    :param epochs: number of epochs
    :param eta: learning rate
    :param mini_batch: size of mini-batches
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch. V
    Values of func_values are calculated for entire training set!
    '''
    w = list(w0)
    func_values = []
    #mini-batches division
    mini_batches = [[],[]]
    mini_batches_number = int (x_train.size/mini_batch)
    for i in range(0, mini_batches_number):
        mini_batches[0].append(x_train[mini_batch*i:mini_batch*(i+1)])
        mini_batches[1].append(y_train[mini_batch*i:mini_batch*(i+1)])

    for i in range(0, epochs):
        for j in range(0, mini_batch + 1):
            val, grad = obj_fun(w, mini_batches[0][j], mini_batches[1][j])
            delta_w = -grad
            w = np.add(w, np.dot(eta, delta_w))
        val, grad = obj_fun(w, x_train, y_train)
        func_values.append([val])
    return (w, np.array(func_values))


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: model parameters Mx1
    :param x_train: training set (features) NxM
    :param y_train: training set (labels) Nx1
    :param regularization_lambda: regularization parameters
    :return: function returns tuple(val, grad), where val is a velue of logistic function with regularization l2,
    and grad its gradient over w
    '''
    def partial_derivative(func_arg_first, func_arg_second, func_val_first, func_val_second):
        delta_arg = func_arg_second - func_arg_first
        delta_val = func_val_second - func_val_first
        return delta_val / delta_arg

    def regulated_logistic_function(w):
        L = logistic_cost_function(w, x_train, y_train)[0]
        lamb = regularization_lambda / 2

        w0 = list(w[1:len(w)])
        norm = np.linalg.norm(w0)
        val = L + lamb*norm**2
        return val

    value = regulated_logistic_function(w)
    grad = []
    i = 0
    diff = 0.0000001
    for w_value in w:
        # gradient is just a vector of partial derivatives
        w_modified = list(w)
        w_modified[i] = w_value + 0.5 * diff
        w_value_copy = w_value - 0.5 * diff
        logistic_function_value = regulated_logistic_function(np.array(w_modified))
        grad_value = partial_derivative(w_value_copy, w_modified[i], value, logistic_function_value)
        grad.append(grad_value * 2)
        i += 1

    return (value, np.array(grad))


def prediction(x, w, theta):
    '''
    :param x: observation matrix NxM
    :param w: parameter vector Mx1
    :param theta: classification threshold [0,1]
    :return: function calculates vector y Nx1. Vector is composed of labels {0,1} for observations x
     calculated using model (parameters w) and classification threshold theta
    '''
    y = []
    N = x.shape[0]
    for i in range(0, N):
        prediction = [0]
        likelihood = sigmoid(np.dot(np.transpose(w), x[i]))
        if (likelihood >= theta):
            prediction = [1]
        y.append(prediction)
    return np.array(y)
    pass


def f_measure(y_true, y_pred):
    '''
    :param y_true: vector of ground truth labels Nx1
    :param y_pred: vector of predicted labels Nx1
    :return: value of F-measure
    '''
    N = y_true.shape[0]
    TP = 0
    FP = 0
    FN = 0
    for i in range(0, N):
        if y_true[i] != y_pred[i] and y_pred[i] == [1]:
            FP += 1
        else:
            if y_true[i] != y_pred[i] and y_pred[i] == [0]:
                FN += 1
            else:
                if y_true[i] == y_pred[i] and y_pred[i] == [1]:
                    TP += 1
    return 2*TP/(2*TP+FP+FN)
    pass


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: trainig set (features) NxM
    :param y_train: training set (labels) Nx1
    :param x_val: validation set (features) Nval x M
    :param y_val: validation set (labels) Nval x 1
    :param w0: vector of initial values of w
    :param epochs: number of iterations of SGD
    :param eta: learning rate
    :param mini_batch: mini-batch size
    :param lambdas: list of lambda values that have to be considered in model selection procedure
    :param thetas: list of theta values that have to be considered in model selection procedure
    :return: Functions makes a model selection. It returs tuple (regularization_lambda, theta, w, F), where regularization_lambda
    is the best velue of regularization parameter, theta is the best classification threshold, and w is the best model parameter vector.
    Additionally function returns matrix F, which stores F-measures calculated for each pair (lambda, theta).
    Use SGD and training criterium with l2 regularization for training.
    '''

    def partial(func, *args, **keywords):
        def newfunc(*fargs, **fkeywords):
            newkeywords = keywords.copy()
            newkeywords.update(fkeywords)
            return func(*args, *fargs, **newkeywords)

        newfunc.func = func
        newfunc.args = args
        newfunc.keywords = keywords
        return newfunc

    w = np.array(w0)
    F = []
    max_f = 0

    #print(regularized_logistic_cost_function(w, x_train, y_train, lambdas[2]))

    for lamb in lambdas:
        F_row = []
        w = np.array(w0)
        partialized_logistic_cost_function = partial(regularized_logistic_cost_function, regularization_lambda=lamb)
        w = stochastic_gradient_descent(partialized_logistic_cost_function, x_train, y_train, w, epochs, eta, mini_batch)[0]
        for theta in thetas:
            current_f = f_measure(y_val ,prediction(x_val, w, theta))
            F_row.append(current_f)
            if  current_f > max_f:
                max_f = current_f
                regularization_lambda = lamb
                best_theta = theta
                best_w = list(w)
        F.append(F_row)
    return (regularization_lambda, best_theta, np.array(best_w), np.array(F))
