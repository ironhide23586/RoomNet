import os

import tensorflow as tf
import os


class RoomNet:

    def __init__(self, num_classes, im_side, is_training=True,
                 learn_rate=1e-4):
        self.num_classes = num_classes
        self.im_side = im_side
        self.is_training = is_training
        self.learn_rate = learn_rate
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, im_side, im_side, 3],
                                       name='input_x_tensor')
        self.y_tensor = tf.placeholder(tf.int32, shape=None, name='input_y_class_ids')
        self.layers = [self.x_tensor]
        self.out_op = self.init_nn_graph()
        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_tensor,
                                                                      logits=self.out_op)
        self.opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
        self.train_op = self.opt.minimize(self.loss_op)
        self.outs_softmax_op = tf.nn.softmax(self.out_op)
        self.outs_final = tf.argmax(self.outs_softmax_op)
        self.sess = None
        self.saver = tf.train.Saver(max_to_keep=0)
        self.step = 0
        self.model_folder = 'trained_models'
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.model_fpath_prefix = self.model_folder + '/' + 'roomnet-'

    def init(self):
        if not self.sess:
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save(self):
        save_fpath = self.model_fpath_prefix + '-' + str(self.step)
        self.saver.save(self.sess, save_fpath)
        print('Model saved at', save_fpath)

    def infer(self, im):
        outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im / 255.})
        return outs

    def train_step(self, x, y):
        loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict={self.x_tensor: x / 255.,
                                                                          self.y_tensor: y})
        self.step += 1
        return loss

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=True, activation=tf.nn.relu6, pooling=True, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.avg_pool, block_depth=3, make_residual=True):
        if not batch_norm:
            use_bias = True
        else:
            use_bias = False
        curr_layer = []
        input = x_in
        for depth in range(block_depth):
            input = tf.layers.conv2d(input, output_filters, kernel_size, strides=kernel_stride,
                                     use_bias=use_bias, activation=activation, dilation_rate=dilation,
                                     padding=padding)
            curr_layer.append(input)
            if pooling:
                input = pooling_fn(input, ksize=[1, pool_ksize, pool_ksize, 1],
                                   strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
                curr_layer.append(input)
            if batch_norm:
                input = tf.layers.batch_normalization(input, training=self.is_training)
                curr_layer.append(input)
            if depth == 0:
                residual_input = input
        if make_residual:
            output = input + tf.image.resize_bilinear(residual_input, input.shape[1:3])
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
        layer_outs = self.conv_block(self.x_tensor, 16)
        layer_outs = self.conv_block(layer_outs, 32, pool_ksize=4, pool_stride=1)
        layer_outs = self.conv_block(layer_outs, 64, pool_ksize=4, pool_stride=2)
        layer_outs = self.conv_block(layer_outs, 32, pooling=False)
        layer_outs = self.conv_block(layer_outs, 16, pool_ksize=4, pool_stride=1)
        layer_outs = self.dense_block(tf.reshape(layer_outs, [-1, 1296]), 128)
        layer_outs = self.dense_block(layer_outs, 64)
        layer_outs = self.dense_block(layer_outs, 16)
        layer_outs = self.dense_block(layer_outs, self.num_classes, batch_norm=False, biased=True)
        return layer_outs

