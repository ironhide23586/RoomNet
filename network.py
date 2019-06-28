import os
from glob import glob

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetMobile

class RoomNet:

    def __init__(self, num_classes, im_side=600, is_training=True, start_step=0,
                 learn_rate=1e-4, l2_regularizer_coeff=1e-2, num_steps=10000):
        self.num_classes = num_classes
        self.im_side = im_side
        self.is_training = is_training
        self.learn_rate = learn_rate
        self.l2_regularizer_coeff = l2_regularizer_coeff
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, im_side, im_side, 3],
                                       name='input_x_tensor')
        self.y_tensor = tf.placeholder(tf.int32, shape=None, name='input_y_class_ids')
        self.layers = [self.x_tensor]
        self.out_op, self.trainable_vars, self.stop_grad_vars = self.init_nn_graph()

        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_tensor,
                                                                      logits=self.out_op)
        l2_losses = [self.l2_regularizer_coeff * tf.nn.l2_loss(v) for v in self.trainable_vars]
        self.reduced_loss = tf.reduce_mean(self.loss_op) + tf.add_n(l2_losses)

        self.start_step = start_step
        self.step = start_step
        self.step_ph = tf.Variable(self.start_step, trainable=False, name='train_step')
        self.learn_rate_tf = tf.train.exponential_decay(self.learn_rate, self.step_ph, num_steps, decay_rate=0.068,
                                                        name='learn_rate')

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_tf)

        # self.train_op = self.opt.minimize(self.reduced_loss, global_step=self.step_ph)
        grads = tf.gradients(self.reduced_loss, self.trainable_vars, stop_gradients=self.stop_grad_vars)
        self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)

        self.outs_softmax_op = tf.nn.softmax(self.out_op)
        self.outs_final = tf.argmax(self.outs_softmax_op, axis=-1)
        self.sess = None

        self.unsaved_vars = [self.step_ph, self.learn_rate_tf]
        self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]

        self.saver = tf.train.Saver(max_to_keep=0, var_list=self.vars_to_keep)
        self.model_folder = 'trained_models'
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.model_fpath_prefix = self.model_folder + '/' + 'roomnet-'

    def init(self):
        if not self.sess:
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save(self, suffix=None):
        if suffix:
            save_fpath = self.model_fpath_prefix + '-' + suffix + '--' + str(self.step)
        else:
            save_fpath = self.model_fpath_prefix + '-' + str(self.step)
        self.saver.save(self.sess, save_fpath)
        print('Model saved at', save_fpath)

    def load(self, model_path=None):
        if model_path is None:
            existing_paths = glob(self.model_folder + '/*.index')
            existing_ids = [int(p.split('--')[-1].replace('.index', '')) for p in existing_paths]
            selected_idx = np.argmax(existing_ids)
            self.step = existing_ids[selected_idx]
            self.start_step = self.step
            model_path = existing_paths[selected_idx].replace('.index', '')
        self.saver.restore(self.sess, model_path)
        step_assign_op = tf.assign(self.step_ph, self.start_step)
        self.sess.run(step_assign_op)
        print('Model restored from', model_path)

    def infer(self, im_in):
        im = ((im_in[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})
        return outs

    def train_step(self, x_in, y):
        x = ((x_in[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        loss, _, step_tf, lr = self.sess.run([self.reduced_loss, self.train_op, self.step_ph, self.learn_rate_tf],
                                             feed_dict={self.x_tensor: x,
                                                        self.y_tensor: y})
        self.step = step_tf
        return loss, step_tf, lr

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=True, activation=tf.nn.relu6, pooling=True, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.avg_pool, block_depth=1, make_residual=True):
        if not batch_norm:
            use_bias = True
        else:
            use_bias = False
        curr_layer = []
        layer_out = x_in
        if block_depth == 1:
            make_residual = False
        for depth in range(block_depth):
            layer_out = tf.layers.conv2d(layer_out, output_filters, kernel_size, strides=kernel_stride,
                                         use_bias=use_bias, activation=activation, dilation_rate=dilation,
                                         padding=padding)
            curr_layer.append(layer_out)
            if pooling:
                layer_out = pooling_fn(layer_out, ksize=[1, pool_ksize, pool_ksize, 1],
                                       strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
                curr_layer.append(layer_out)
            if batch_norm:
                layer_out = tf.layers.batch_normalization(layer_out, training=self.is_training)
                curr_layer.append(layer_out)
            if depth == 0:
                residual_input = layer_out
            output = layer_out
        if make_residual:
            output = output + tf.image.resize_bilinear(residual_input, output.shape[1:3])
            curr_layer.append(output)
            if batch_norm:
                output = tf.layers.batch_normalization(output, training=self.is_training)
                curr_layer.append(output)
        self.layers.append(curr_layer)
        return output

    def dense_block(self, x_in, num_outs, batch_norm=True, biased=False):
        curr_layer = []
        layer_outs = tf.layers.dense(x_in, num_outs, use_bias=biased)
        curr_layer.append(layer_outs)
        layer_outs = tf.nn.relu6(layer_outs)
        curr_layer.append(layer_outs)
        if batch_norm:
            layer_outs = tf.layers.batch_normalization(layer_outs, training=self.is_training)
            curr_layer.append(layer_outs)
        self.layers.append(curr_layer)
        return layer_outs

    def init_nn_graph(self):
        # layer_outs = self.conv_block(self.x_tensor, 8)
        # layer_outs = self.conv_block(layer_outs, 8, pool_ksize=4, pool_stride=1, block_depth=2)
        # layer_outs = self.conv_block(layer_outs, 8, pool_ksize=4, pool_stride=2, block_depth=2)
        # # layer_outs = self.conv_block(layer_outs, 8, pooling=False)
        # layer_outs = self.conv_block(layer_outs, 8, pool_ksize=4, pool_stride=2, block_depth=2)
        # shp = layer_outs.shape
        # flat_len = shp[1] * shp[2] * shp[3]
        # layer_outs = self.dense_block(tf.reshape(layer_outs, [-1, flat_len]), 32)
        # # layer_outs = self.dense_block(layer_outs, 16)
        # layer_outs = self.dense_block(layer_outs, 8)
        # layer_outs = self.dense_block(layer_outs, self.num_classes, batch_norm=False, biased=True)

        net = NASNetMobile(input_tensor=self.x_tensor, input_shape=(self.im_side, self.im_side, 3),
                           include_top=False, weights='imagenet', pooling='avg')
        layer_outs = net.output
        y0 = tf.global_variables()
        layer_outs = self.dense_block(layer_outs, 16)
        layer_outs = self.dense_block(layer_outs, self.num_classes, batch_norm=False, biased=True)
        y1 = tf.global_variables()
        trainable_vars = [v for v in y1 if v not in y0]
        return layer_outs, trainable_vars, net.output

