'''
Author: Souham Biswas
Email: souham.biswas@outlook.com
GitHub: https://github.com/ironhide23586
LinkedIn: https://www.linkedin.com/in/souham

I'm not responsible if your machine catches fire.
'''


import os
from glob import glob

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2


class RoomNet:

    def __init__(self, num_classes, im_side=600, compute_bn_mean_var=True, start_step=0, dropout_enabled=False,
                 learn_rate=1e-4, l2_regularizer_coeff=1e-2, num_steps=10000, dropout_rate=.2,
                 update_batchnorm_means_vars=True, optimized_inference=False, train_batch_size=32,
                 load_training_vars=False):
        self.num_classes = num_classes
        self.im_side = im_side
        self.compute_bn_mean_var = compute_bn_mean_var
        self.optimized_inference = optimized_inference
        self.train_batch_size = train_batch_size
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, im_side, im_side, 3],
                                       name='input_x_tensor')
        self.layers = [self.x_tensor]

        self.start_step = start_step
        self.step = start_step
        self.learn_rate = learn_rate
        self.step_ph = tf.Variable(self.start_step, trainable=False, name='train_step')
        self.learn_rate_tf = tf.train.exponential_decay(self.learn_rate, self.step_ph, num_steps, decay_rate=0.068,
                                                        name='learn_rate')
        self.unsaved_vars = [self.step_ph, self.learn_rate_tf]

        self.sess = None
        if self.optimized_inference:
            self.dropout_enabled = False
            self.out_op, _, _, _ = self.init_nn_graph()
            self.outs_softmax_op = tf.nn.softmax(self.out_op)
            self.outs_final = tf.argmax(self.outs_softmax_op, axis=-1)
            self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]
            self.restorer = tf.train.Saver(var_list=self.vars_to_keep)
            return
        self.dropout_enabled = dropout_enabled
        self.l2_regularizer_coeff = l2_regularizer_coeff
        self.y_tensor = tf.placeholder(tf.int32, shape=None, name='input_y_class_ids')
        if self.dropout_enabled:
            self.dropout_rate = dropout_rate
            self.dropout_rate_tensor = tf.placeholder(tf.float32, shape=())
        self.out_op, self.trainable_vars, self.stop_grad_vars, self.restore_excluded_vars = self.init_nn_graph()
        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_tensor,
                                                                      logits=self.out_op)
        l2_losses = [self.l2_regularizer_coeff * tf.nn.l2_loss(v) for v in self.trainable_vars]
        self.reduced_loss = tf.reduce_mean(self.loss_op) + tf.add_n(l2_losses)

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_tf)
        grads = tf.gradients(self.reduced_loss, self.trainable_vars, stop_gradients=self.stop_grad_vars)

        if update_batchnorm_means_vars:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)
        else:
            self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)

        self.outs_softmax_op = tf.nn.softmax(self.out_op)
        self.outs_final = tf.argmax(self.outs_softmax_op, axis=-1)

        if not load_training_vars:
            self.restore_excluded_vars += [v for v in tf.all_variables() if 'Adam' in v.name or 'power' in v.name]
        else:
            self.restore_excluded_vars = []

        self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]
        self.vars_to_restore = [v for v in self.vars_to_keep if v not in self.restore_excluded_vars]

        self.saver = tf.train.Saver(max_to_keep=0, var_list=self.vars_to_keep)
        self.restorer = tf.train.Saver(var_list=self.vars_to_restore)
        self.model_folder = 'all_trained_models/trained_models'
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.model_fpath_prefix = self.model_folder + '/' + 'roomnet-'

    def init(self):
        if not self.sess:
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save(self, suffix=None):
        if self.optimized_inference:
            self.restorer.save(self.sess, 'roomnet')
            print('Model Saved in optimized inference mode')
            return
        if suffix:
            save_fpath = self.model_fpath_prefix + '-' + suffix + '--' + str(self.step)
        else:
            save_fpath = self.model_fpath_prefix + '-' + str(self.step)
        self.saver.save(self.sess, save_fpath)
        print('Model saved at', save_fpath)

    def load(self, model_path=None):
        if not self.sess:
            self.init()
        if model_path is None:
            if os.path.isdir(self.model_folder):
                existing_paths = glob(self.model_folder + '/*.index')
                if len(existing_paths) == 0:
                    print('No model found to restore from, initializing random weights')
                    return
                existing_ids = [int(p.split('--')[-1].replace('.index', '')) for p in existing_paths]
                selected_idx = np.argmax(existing_ids)
                self.step = existing_ids[selected_idx]
                self.start_step = self.step
                model_path = existing_paths[selected_idx].replace('.index', '')
            else:
                print('No model found to restore from, initializing random weights')
                return
        self.restorer.restore(self.sess, model_path)
        if not self.optimized_inference:
            step_assign_op = tf.assign(self.step_ph, self.start_step)
            self.sess.run(step_assign_op)
        print('Model restored from', model_path)

    def infer(self, im_in):
        im = ((im_in[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        if self.dropout_enabled:
            outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im,
                                                             self.dropout_rate_tensor: 0.})
        else:
            outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})
        return outs

    def center_crop(self, x):
        h, w, _ = x.shape
        offset = abs((w - h) // 2)
        if h < w:
            x_pp = x[:, offset:offset + h, :]
        elif w < h:
            x_pp = x[offset:offset + w, :, :]
        else:
            x_pp = x.copy()
        return x_pp

    def infer_optimized(self, im_in):
        im = self.center_crop(im_in)
        h, w, _ = im.shape
        if h != self.im_side or w != self.im_side:
            im = cv2.resize(im, (self.im_side, self.im_side))
        im = ((im[:, :, [2, 1, 0]] / 255.) * 2) - 1
        im = np.expand_dims(im, 0)
        out_label_idx = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})[0]
        return out_label_idx

    def train_step(self, x_in, y):
        x = ((x_in[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        if self.dropout_enabled:
            loss, _, step_tf, lr = self.sess.run([self.reduced_loss, self.train_op, self.step_ph, self.learn_rate_tf],
                                                 feed_dict={self.x_tensor: x,
                                                            self.y_tensor: y,
                                                            self.dropout_rate_tensor: self.dropout_rate})
        else:
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
                layer_out = tf.layers.batch_normalization(layer_out, training=self.compute_bn_mean_var)
                curr_layer.append(layer_out)
            if depth == 0:
                residual_input = layer_out
            output = layer_out
        if make_residual:
            output = output + tf.image.resize_bilinear(residual_input, output.shape[1:3])
            curr_layer.append(output)
            if batch_norm:
                output = tf.layers.batch_normalization(output, training=self.compute_bn_mean_var)
                curr_layer.append(output)
        if self.dropout_enabled:
            output = tf.nn.dropout(output, rate=self.dropout_rate_tensor)
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
            layer_outs = tf.layers.batch_normalization(layer_outs, training=self.compute_bn_mean_var)
            curr_layer.append(layer_outs)
        if self.dropout_enabled:
            layer_outs = tf.nn.dropout(layer_outs, rate=self.dropout_rate_tensor)
            curr_layer.append(layer_outs)
        self.layers.append(curr_layer)
        return layer_outs

    def ssd_block_box_predictor(self, net, out_channels, scope_idx):
        net = self.ssd_block('BoxEncodingPredictor', net, out_channels, scope_idx)
        net = tf.reshape(net, [self.train_batch_size, -1, 1, 4])
        return net

    def ssd_block_class_predictor(self, net, out_channels, scope_idx):
        net = self.ssd_block('ClassPredictor', net, out_channels, scope_idx)
        net = tf.reshape(net, [self.train_batch_size, -1, self.num_classes])
        return net

    def ssd_block(self, name, net, out_channels, scope_idx):
        net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=[3, 3], stride=1,
                                    scope='BoxPredictor_' + str(scope_idx) + '/' + name + '_depthwise',
                                    padding='SAME', depth_multiplier=1, normalizer_fn=None,
                                    activation_fn=None, biases_initializer=None)
        if self.compute_bn_mean_var:
            net = tf.layers.batch_normalization(net, training=self.compute_bn_mean_var)
        net = slim.conv2d(inputs=net, num_outputs=out_channels, kernel_size=[1, 1],
                          scope='BoxPredictor_' + str(scope_idx) + '/' + name,
                          stride=1, activation_fn=tf.nn.relu6, normalizer_fn=None)
        if self.compute_bn_mean_var:
            net = tf.layers.batch_normalization(net, training=self.compute_bn_mean_var)
        return net

    def ssdlite_nn(self, ssd_endpoints, num_box_points=4):
        detection_outs_all = [self.ssd_block_box_predictor(ssd_endpoints[0], num_box_points * 3, 0)] \
                             + [self.ssd_block_box_predictor(ssd_endpoints[i + 1], num_box_points * 6, i + 1)
                                for i in range(len(ssd_endpoints[1:]))]
        classification_outs_all = [self.ssd_block_class_predictor(ssd_endpoints[0], self.num_classes * 3, 0)] \
                                  + [self.ssd_block_class_predictor(ssd_endpoints[i + 1], self.num_classes * 6, i + 1)
                                     for i in range(len(ssd_endpoints[1:]))]
        detection_out = tf.concat(detection_outs_all, axis=1)
        detection_out = tf.reshape(detection_out, [detection_out.shape[0].value, -1,
                                                   detection_out.shape[-1].value], name='detections_raw')
        classification_out = tf.concat(classification_outs_all, axis=1)
        classification_out_softmax = tf.nn.softmax(classification_out, name='classifications_raw')
        return detection_out, classification_out_softmax

    def init_nn_graph(self):
        layer_outs = self.conv_block(self.x_tensor, 8)
        layer_outs = self.conv_block(layer_outs, 32, pool_ksize=4, pool_stride=1, block_depth=3)
        layer_outs = self.conv_block(layer_outs, 64, pool_ksize=4, pool_stride=2, block_depth=2)
        layer_outs = self.conv_block(layer_outs, 128, pooling=False)
        layer_outs = self.conv_block(layer_outs, 16, pool_ksize=4, pool_stride=2, block_depth=3)



        # shp = layer_outs.shape
        # flat_len = shp[1] * shp[2] * shp[3]
        # v0 = tf.all_variables()
        # layer_outs = self.dense_block(tf.reshape(layer_outs, [-1, flat_len]), 32)
        # layer_outs = self.dense_block(layer_outs, 16)
        # layer_outs = self.dense_block(layer_outs, 8)
        # layer_outs = self.dense_block(layer_outs, self.num_classes, batch_norm=False, biased=True)
        # v1 = tf.all_variables()
        
        trainable_vars = tf.trainable_variables()
        original_vars = []

        # restore_excluded_vars = [v for v in v1 if v not in v0]
        restore_excluded_vars = []
        return layer_outs, trainable_vars, original_vars, restore_excluded_vars
