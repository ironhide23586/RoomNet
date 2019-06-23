import os

import tensorflow as tf
import cv2
import numpy as np


class RoomNet:

    def __init__(self, num_classes, im_side, is_training=True,
                 learn_rate=1e-4):
        self.num_classes = num_classes
        self.im_side = im_side
        self.is_training = is_training
        self.learn_rate = learn_rate
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, im_side, im_side, 3],
                                       name='input_x_tensor')
        self.y_tensor = tf.placeholder(tf.int8, shape=None, name='input_y_class_ids')
        self.layers = [self.x_tensor]
        self.init_nn_graph()

    def conv_block(self, x_in, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=True, activation=tf.nn.relu6, pooling=True, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.avg_pool, block_depth=3, make_residual=True,
                   depth_expansion_factor=8):
        if not batch_norm:
            use_bias = True
        else:
            use_bias = False
        curr_layer = []
        input = x_in
        prev_output_channels = x_in.shape[-1] + depth_expansion_factor
        for depth in range(block_depth):
            input = tf.layers.conv2d(input, prev_output_channels, kernel_size, strides=kernel_stride,
                                     use_bias=use_bias, activation=activation, dilation_rate=dilation,
                                     padding=padding)
            prev_output_channels += depth_expansion_factor
            curr_layer.append(input)
            if pooling:
                input = pooling_fn(input, ksize=[1, pool_ksize, pool_ksize, 1],
                                   strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
                curr_layer.append(input)
            if batch_norm:
                input = tf.layers.batch_normalization(input, training=self.is_training)
                curr_layer.append(input)
        if make_residual:
            output = input + x_in
            curr_layer.append(output)
            if batch_norm:
                output = tf.layers.batch_normalization(output, training=self.is_training)
                curr_layer.append(output)
        self.layers.append(curr_layer)
        return output

    def init_nn_graph(self):
        layer_outs = self.conv_block(self.x_tensor, depth_expansion_factor=8)
        layer_outs = self.conv_block(layer_outs, depth_expansion_factor=8, pool_ksize=2, pool_stride=1)
        layer_outs = self.conv_block(layer_outs, depth_expansion_factor=8, pool_ksize=4, pool_stride=1)
        layer_outs = self.conv_block(layer_outs, depth_expansion_factor=8, pool_ksize=4, pool_stride=1)
        layer_outs = self.conv_block(layer_outs, depth_expansion_factor=8, pool_kize=4, pool_stride=1)
        layer_outs = self.conv_block(layer_outs, depth_expansion_factor=8, pool_kize=4, pool_stride=1)
        k = 0
