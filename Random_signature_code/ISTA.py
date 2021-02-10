# -*- coding: utf-8 -*-
# @Author: weilantian
# @Date:   2021-02-10
# @Last Modified by:   1uci3n
# @Last Modified time: 2021-02-10 15:56:22

# @Python_version: 3.9.1
import numpy as np

def soft_threshold_shrinkage(vector, tao):
    '''soft threshold shrinkage method original version
    Shrinkage a vector.

    Args:
        vector: The vector should be shrinkage.
        tao: The threshold of shrinkage method.
    
    Returns:
        The shrinkage vector.
    '''
    for i in range(vector.size):
        if np.absolute(vector[i]) < tao:
            vector[i] = 0
        else:
            vector[i] = vector[i] - (np.sign(vector[i]) * tao)
    return vector

def soft_threshold_shrinkage_function_for_positive_vector(vector, tao):
    '''soft threshold shrinkage method for positive vextor version
    Shrinkage a positive vector.

    Args:
        vector: The vector should be shrinkage.
        tao: The threshold of shrinkage method.
    
    Returns:
        The shrinkage vector..
    '''
    for i in range(vector.size):
        if vector[i] < tao:
            vector[i] = 0
        else:
            vector[i] = vector[i] - tao
    return vector

def ISTA(y, A, b, tao, maxit = 300, stop_difference = 1e-05):
    '''ISTA method

    Args:
        y: Received signal.
        A: Sensing matrix.
        b: Setp szie.
        tao: Threshold of soft threshold shrinkage method.
        maxit: Max number of iterative. Default: 300.
        stop_difference: Stop the iterative when the difference is less than this number. Default: 1e-05.

    Returns:
        Recovered vector, cost step
    '''
    AT = np.transpose(A)    # AT is the transpose matrix of A
    s = np.zeros(np.shape(A)[1])   # Initialization of recovery results
    r = s + (b * np.matmul(AT, (y - np.matmul(A, s))))
    s = soft_threshold_shrinkage(r, tao)
    step_counter = 1    # Initialization the step counter.
    for _ in range(maxit - 1):
            r = s + (b * np.matmul(AT, (y - np.matmul(A, s))))
            s_current = soft_threshold_shrinkage(r, tao)
            diff = np.linalg.norm(s_current-s, 2)   # Calculate the difference with the previous result, can also use other methods.
            s = s_current
            step_counter = step_counter + 1
            if diff < stop_difference:
                break
    return s, step_counter
