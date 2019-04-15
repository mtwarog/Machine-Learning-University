# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial

def mean_squared_error(x, y, w):

    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''
    N = x.size
    sum = 0
    #calculating sum
    #TODO refactor to np.sum (there is need to produce actualY vector?)
    index = 0
    for expectedY in y:
        actualY = polynomial(x[index], w)
        sum += (expectedY - actualY)**2
        index += 1
    return (1/N*(sum))


def design_matrix(x_train,M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''
    design_matrix = []
    for input_value in x_train:
        fi_vector = []
        for degree in range(0,M+1):
            fi_vector.append((input_value**degree)[0])
        design_matrix.append(fi_vector)

    design_matrix = np.array(design_matrix)
    return design_matrix


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    fi = design_matrix(x_train, M)
    w = np.transpose(fi)
    w = np.dot(w, fi)
    w = (np.linalg.inv(w))
    w = np.dot(np.dot(w, np.transpose(fi)), y_train)
    err = mean_squared_error(x_train, y_train, w)[0]
    return (w, err)
    pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    fi = design_matrix(x_train, M)
    w = np.transpose(fi)
    w = np.dot(w, fi)
    w = np.add(w, np.multiply(regularization_lambda, np.identity(w.shape[0]))) #np.add(w, np.multiplyScalar(regularization_lambda, np.identity(w.size)))
    w = (np.linalg.inv(w))
    w = np.dot(np.dot(w, np.transpose(fi)), y_train)
    err = mean_squared_error(x_train, y_train, w)[0]
    return (w, err)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''
    current_min_err = np.inf
    for M in M_values:
        least_squares_solution = least_squares(x_train, y_train, M)
        current_w = least_squares_solution[0]
        current_train_err = least_squares_solution[1]
        current_val_err = mean_squared_error(x_val, y_val, current_w)[0]
        if  current_val_err < current_min_err:
            solution_w = current_w
            train_err = current_train_err
            val_err = current_val_err
            current_min_err = current_val_err
    return (solution_w, train_err, val_err)


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''
    current_min_err = np.inf
    for lambda_value in lambda_values:
        least_squares_solution = regularized_least_squares(x_train, y_train, M, lambda_value)
        current_w = least_squares_solution[0]
        current_train_err = least_squares_solution[1]
        current_val_err = mean_squared_error(x_val, y_val, current_w)[0]
        current_lambda = lambda_value
        if current_val_err < current_min_err:
            solution_w = current_w
            train_err = current_train_err
            val_err = current_val_err
            regularization_lambda = current_lambda
            current_min_err = current_val_err
    return (solution_w, train_err, val_err, regularization_lambda)
    pass