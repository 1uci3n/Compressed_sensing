# -*- coding: utf-8 -*-
# @Author: weilantian
# @Date:   2021-01-19
# @Last Modified by:   1uci3n
# @Last Modified time: 2021-06-21 15:18:00

# @Python_version: 3.8.
# @this_version: 1.0

import tensorflow as tf

from tensorflow import keras

class NMSE_Accuracy(keras.metrics.Metric):
    """NMSE_Accuracy: a self defined Keras metric"""
    def __init__(self, name="nmse_accuracy", **kwargs):
        super(NMSE_Accuracy, self).__init__(name=name, **kwargs)
        self.sum_nmse = self.add_weight(name="sum_nmse", initializer="zeros")
        self.sample_number = self.add_weight(name="sample_number", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        denominator = tf.reduce_sum(tf.pow(y_true, 2), 1)
        current_nmse = tf.reduce_sum(tf.pow(diff, 2), 1) / denominator
        current_nmse = tf.where(tf.math.is_nan(current_nmse)|tf.math.is_inf(current_nmse), tf.zeros_like(current_nmse), current_nmse)
        current_nmse = tf.cast(tf.reduce_sum(current_nmse), "float32")
        self.sum_nmse.assign_add(current_nmse)
        current_sample_number = tf.math.count_nonzero(denominator)
        current_sample_number = tf.cast(current_sample_number, "float32")
        self.sample_number.assign_add(current_sample_number)

    def result(self):
        return 10 * (tf.math.log(self.sum_nmse / self.sample_number) / tf.math.log(10.))

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_nmse.assign(0.0)
        self.sample_number.assign(0.0)

class Block_Error_Ratio_Threshold_0_5(keras.metrics.Metric):
    """

    """
    def __init__(self, name="block_error_ratio", **kwargs):
        super(Block_Error_Ratio_Threshold_0_5, self).__init__(name=name, **kwargs)
        self.sum_block = self.add_weight(name="sum_block", initializer="zeros")
        self.sum_error = self.add_weight(name="sum_error", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ones_tensor = tf.ones_like(y_true)
        ones_true = ones_tensor * tf.cast((y_true > 0), "float32")
        ones_output = ones_tensor * tf.cast((y_pred > 0.5), "float32")
        diff = ones_output - ones_true
        sum_err = tf.reduce_sum(diff, 1)
        current_error = tf.math.count_nonzero(sum_err)
        current_error = tf.cast(current_error, "float32")
        self.sum_error.assign_add(current_error)
        current_block = y_true.shape[0]
        current_block = tf.cast(current_block, "float32")
        self.sum_block.assign_add(current_block)

    def result(self):
        return self.sum_error / self.sum_block

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_block.assign(0.0)
        self.sum_error.assign(0.0)

class Bit_Error_Ratio_Threshold_0_5(keras.metrics.Metric):
    """Only for y_true (x or h) greater than zero."""
    def __init__(self, name="bit_error_ratio", **kwargs):
        super(Bit_Error_Ratio_Threshold_0_5, self).__init__(name=name, **kwargs)
        self.sum_bit = self.add_weight(name="sum_bit", initializer="zeros")
        self.sum_error = self.add_weight(name="sum_error", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ones_tensor = tf.ones_like(y_true)
        ones_true = ones_tensor * tf.cast((y_true > 0), "float32")
        ones_output = ones_tensor * tf.cast((y_pred > 0.5), "float32")
        current_error = tf.math.count_nonzero(ones_output - ones_true)
        current_error = tf.cast(current_error, "float32")
        self.sum_error.assign_add(current_error)
        current_bit = tf.size(y_true)
        current_bit = tf.cast(current_bit, "float32")
        self.sum_bit.assign_add(current_bit)

    def result(self):
        return self.sum_error / self.sum_bit

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_bit.assign(0.0)
        self.sum_error.assign(0.0)

class Block_Error_Ratio_Threshold_0_01(keras.metrics.Metric):
    """

    """
    def __init__(self, name="block_error_ratio", **kwargs):
        super(Block_Error_Ratio_Threshold_0_01, self).__init__(name=name, **kwargs)
        self.sum_block = self.add_weight(name="sum_block", initializer="zeros")
        self.sum_error = self.add_weight(name="sum_error", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        to_get_block = tf.reduce_sum(y_true, 1)
        ones_tensor = tf.ones_like(y_true)
        ones_true = ones_tensor * tf.cast((y_true > 0), "float32")
        ones_output = ones_tensor * tf.cast((y_pred > 0.01), "float32")
        diff = ones_output - ones_true
        sum_err = tf.reduce_sum(diff, 1)
        current_error = tf.math.count_nonzero(sum_err)
        current_error = tf.cast(current_error, "float32")
        self.sum_error.assign_add(current_error)
        current_block = tf.math.count_nonzero(to_get_block)
        current_block = tf.cast(current_block, "float32")
        # current_block = y_true.shape[0]
        # current_block = tf.cast(current_block, "float32")
        self.sum_block.assign_add(current_block)

    def result(self):
        return self.sum_error / self.sum_block

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_block.assign(0.0)
        self.sum_error.assign(0.0)

class Bit_Error_Ratio_Threshold_0_01(keras.metrics.Metric):
    """Only for y_true (x or h) greater than zero."""
    def __init__(self, name="bit_error_ratio", **kwargs):
        super(Bit_Error_Ratio_Threshold_0_01, self).__init__(name=name, **kwargs)
        self.sum_bit = self.add_weight(name="sum_bit", initializer="zeros")
        self.sum_error = self.add_weight(name="sum_error", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ones_tensor = tf.ones_like(y_true)
        ones_true = ones_tensor * tf.cast((y_true > 0), "float32")
        ones_output = ones_tensor * tf.cast((y_pred > 0.01), "float32")
        current_error = tf.math.count_nonzero(ones_output - ones_true)
        current_error = tf.cast(current_error, "float32")
        self.sum_error.assign_add(current_error)
        current_bit = tf.size(y_true)
        current_bit = tf.cast(current_bit, "float32")
        self.sum_bit.assign_add(current_bit)

    def result(self):
        return self.sum_error / self.sum_bit

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_bit.assign(0.0)
        self.sum_error.assign(0.0)

# class M_L():