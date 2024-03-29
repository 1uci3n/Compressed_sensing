# -*- coding: utf-8 -*-
# @Author: weilantian
# @Date:   2021-01-04 22:44:06
# @Last Modified by:   1uci3n
# @Last Modified time: 2023-3-21 23:00:35

# @Python_version: 3.9.1

# This toolbox contains generation tools and processing tools for signature code data,
# as well as some related calculation tools such as SNR related calculations.
from statistics import variance
import numpy as np

def get_random_binary_signature_matrix_1_inverse1(user_number, code_length):
    '''
    a simple function to get (1, -1)-binary signature matrix by give user number and code length.
    (0.5,0.5)

    Args:
        user_number: the total number of user.
        code_length: the code length of signature code.

    Returns:
        matrix signature matrix.
    '''
    matrix = 1 - (2 * np.random.binomial(1,0.5, (user_number,code_length)))
    return matrix

def get_random_triad_signature_matrix_0_1_inverse1(user_number, code_length):
    '''
    a simple function to get (0, 1, -1)-signature matrix by give user number and code length.
    (0.5,0.25,0.25)

    Args:
        user_number: the total number of user.
        code_length: the code length of signature code.

    Returns:
        matrix signature matrix.
    '''
    matrix = 1 - (2 * np.random.binomial(1,0.5, (user_number,code_length)))
    return matrix * np.random.binomial(1,0.5, (user_number,code_length))

def get_random_binary_signature_matrix_0_1(user_number, code_length):
    '''
    a simple function to get (0, 1)-binary signature matrix by give user number and code length.
    (0.5,0.5)

    Args:
        user_number: the total number of user.
        code_length: the code length of signature code.

    Returns:
        matrix signature matrix.
    '''
    matrix = np.random.binomial(1,0.5, (user_number,code_length))
    return matrix

def get_random_complex_gauss_matrix(m, n, loc=0, scale=0.5):
    '''
    a function to get complex Gaussian matrix by give the shape

    Args:
        m: numbr of the rows.
        n: number of the columns.
        loc Mean (“centre”) of the distribution
        scale Standard deviation (spread or “width”) of the distribution

    Returns:
        matrix a complex Gaussian matrix.
    '''
    original_gauss_matrix = np.random.normal(loc, scale, (m, n, 2))
    matrix = original_gauss_matrix.view(np.complex128).reshape(m, n)
    return matrix

def get_random_complex_1_inv_1_matrix(m, n):
    '''
    a function to get complex (1,-1)-matrix by give the shape

    Args:
        m: numbr of the rows.
        n: number of the columns.

    Returns:
        matrix a complex (1,-1)-matrix.
    '''
    original_1_inv_1_matrix = (np.random.binomial(1,0.5,(10,10,2)).astype(np.float64)) * 2 - 1
    matrix = original_1_inv_1_matrix.view(np.complex128)
    return matrix

def get_random_complex_1_0_matrix(m, n):
    '''
    a function to get complex (1, 0)-matrix by give the shape

    Args:
        m: numbr of the rows.
        n: number of the columns.

    Returns:
        matrix a complex (1, 0)-matrix.
    '''
    original_1_0_matrix = np.random.binomial(1,0.5,(10,10,2)).astype(np.float64)
    matrix = original_1_0_matrix.view(np.complex128)
    return matrix

def get_gauss_random_signature_matrix(user_number, code_length, mean=0, std=1 ):
    return np.random.normal(mean, std, size=(user_number, code_length))

def get_dataset(size, signature_matrix, acteve_rate=0.1, is_fading=False, fading_scale=1.0, fading_mode=1):
    '''
    a function to generate the random data set by give set size and signature matrix

    Args:
        size: The size of data set.
        signature_matrix: The sensing matrix is used to
            generated the received signal.
        acteve_rate The user's active probability,
            the default value is 0.1.
        is_fading: To control whether the generated data contains fading,
            the default value is 0 (false)
        fading_scale: To set the fading scale,
            the default value is 1.
        fading_mode: To set the fading mode,
            the default value is 1 (Rayleigh fading).

    Returns:
        Y_set received signal, X_set user status, H_set fading coefficient.
    '''
    user_number = np.shape(signature_matrix)[0]
    X_set = np.random.binomial(1, acteve_rate, (size, user_number))
    if is_fading:
        if fading_mode == 1:
            H_set = np.random.rayleigh(fading_scale, np.shape(X_set))
        elif fading_mode == 2:
            H_set = np.random.normal(0, fading_scale, np.shape(X_set))
        else:
            print("error fading mod")
    else:
        H_set = np.ones_like(X_set)
    H_set = H_set * X_set
    Y_set = np.dot(H_set, signature_matrix)
    return Y_set, X_set, H_set

def get_dataset_with_double_side_noise(size, signature_matrix, snr_dB, acteve_rate=0.1, is_fading=False, fading_scale=1.0, fading_mode=1, print_var=False, classic_noise=False):
    '''
    a function to generate the random data set by give set size and signature matrix
    with doubel side noise

    Args:
        size: The size of data set.
        signature_matrix: The sensing matrix is used to
            generated the received signal.
        snr_dB: dB value of snr
        acteve_rate The user's active probability,
            the default value is 0.1.
        is_fading: To control whether the generated data contains fading,
            the default value is 0 (false)
        fading_scale: To set the fading scale,
            the default value is 1.
        fading_mode: To set the fading mode,
            the default value is 1 (Rayleigh fading).
        print_var: print the variance of the noise distribution，
            the default value is False.
        classic_noise: should use the classic formula to calculate the power
            power(signal,2) or not abs(signal)

    Returns:
        Y_set_with_noise received signal with noise, X_set user status vector,
        H_set fading coefficient vector, Y_set received signal without noise
    '''
    Y_set, X_set, H_set = get_dataset(size, signature_matrix, acteve_rate=acteve_rate, is_fading=is_fading, fading_scale=fading_scale, fading_mode=fading_mode)
    if classic_noise:
        single_bit_power = get_average_single_bit_power_of_signature_matrix_old(signature_matrix, np.shape(signature_matrix)[1], acteve_rate=acteve_rate)
    else:
        single_bit_power = get_average_single_bit_power_of_signature_matrix(signature_matrix, np.shape(signature_matrix)[1], acteve_rate=acteve_rate)
    snr_rate = dB_2_rate(snr_dB)
    noise_variance = (single_bit_power / snr_rate) / 2
    Y_set_with_noise = Y_set + np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=Y_set.shape)
    if print_var:
        print(noise_variance)
    return Y_set_with_noise, X_set, H_set, Y_set

def test_power_of_signature_matrix_fixed(signature_matrix, code_length, acteve_rate=0.1, test_data_size=1000):
    if code_length != np.shape(signature_matrix)[1] :
        print('code length error')
        return None
    Y_set, _, _ = get_dataset(test_data_size, signature_matrix, acteve_rate=acteve_rate, is_fading=0)
    sum_power = Y_set.sum()
    entire_signal_power = sum_power / test_data_size
    single_bit_power = entire_signal_power / code_length
    return single_bit_power

def get_average_single_bit_power_of_signature_matrix(signature_matrix, code_length, acteve_rate=0.1):
    if code_length != np.shape(signature_matrix)[1] :
        print('code length error')
        return None
    return np.abs(signature_matrix).mean() * np.shape(signature_matrix)[0] * acteve_rate

def get_average_single_bit_power_of_signature_matrix_old(signature_matrix, code_length, acteve_rate=0.1):
    if code_length != np.shape(signature_matrix)[1] :
        print('code length error')
        return None
    return (np.linalg.norm(signature_matrix, 2, axis=1)**2).mean() * np.shape(signature_matrix)[0] * acteve_rate / code_length

def test_power_of_signature_matrix_err(signature_matrix, code_length, acteve_rate=0.1, test_data_size=1000):
    if code_length != np.shape(signature_matrix)[1] :
        print('code length error')
        return None
    Y_set, _, _ = get_dataset(test_data_size, signature_matrix, acteve_rate=acteve_rate, is_fading=0)
    Y_pow = np.power(Y_set, 2)
    sum_power = Y_pow.sum()
    entire_signal_power = sum_power / test_data_size
    single_bit_power = entire_signal_power / code_length
    return single_bit_power

def dB_2_rate(dB):
    return np.power(10., dB/10.)

def rate_2_dB(rate):
    return 10. * np.log10(rate)

def calculate_mse(v_1, v_2):
    # return np.linalg.norm(v_1 - v_2, 2) / v_1.shape[0]
    return (np.linalg.norm(v_1 - v_2, 2) ** 2) / v_1.shape[0]