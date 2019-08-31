"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""


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
        self.ssd_endpoints = []
        self.x_tensor = tf.placeholder(tf.float32, shape=(self.train_batch_size, self.im_side, self.im_side, 3),
                                       name='input_x_tensor')
        self.layers = []
        self.layer_var_mappings_ordered = []

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
        self.y_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 4 + 1), name='gt_labels_agg')

        # Number of GT bboxes in each image
        self.num_y_bboxes_per_image_tensor = tf.placeholder(dtype=tf.int32, shape=(None))

        if self.dropout_enabled:
            self.dropout_rate = dropout_rate
            self.dropout_rate_tensor = tf.placeholder(tf.float32, shape=())
        self.out_op, self.trainable_vars, self.stop_grad_vars, self.restore_excluded_vars = self.init_nn_graph()

        self.layername_var_map = {}
        for layer_params in self.layer_var_mappings_ordered:
            for layer_data in layer_params:
                for k, v in layer_data.items():
                    self.layername_var_map[k] = v

        self.loss_op = self.loss_function([self.y_tensor, self.num_y_bboxes_per_image], self.out_op)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_tf)
        grads = tf.gradients(self.loss_op, self.trainable_vars, stop_gradients=self.stop_grad_vars)

        if update_batchnorm_means_vars:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)
        else:
            self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)

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

    def loss_function(self, y_truth, y_pred):
        detection_out, classification_out_softmax, classification_out = y_pred
        gt_bboxes, gt_bboxes_counts = y_truth
        ssd_endpoint_sides = [int(ep.shape[1]) for ep in self.ssd_endpoints]
        anchor_bboxes = self.gen_anchors(ssd_endpoint_sides)
        loss_nr = self.loss_op_tf(detection_out, classification_out, gt_bboxes, gt_bboxes_counts, anchor_bboxes)
        train_vars = [v for v in tf.trainable_variables() if v not in self.stop_grad_vars]
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'bias' not in v.name]) * self.l2_regularizer_coeff
        loss = loss_nr + loss_l2
        return loss

    def compute_iou_centered_yxhw_vectorized_tf(self, bb0, bb1):
        y0_center, x0_center, h0, w0 = tuple([bb0[:, i] for i in range(bb0.shape[1])])
        y1_center, x1_center, h1, w1 = tuple([bb1[:, i] for i in range(bb1.shape[1])])
        boxA = [x0_center - (w0 / 2),
                y0_center - (h0 / 2),
                x0_center + (w0 / 2),
                y0_center + (h0 / 2)]
        boxB = [x1_center - (w1 / 2),
                y1_center - (h1 / 2),
                x1_center + (w1 / 2),
                y1_center + (h1 / 2)]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = tf.reduce_max(tf.concat([tf.expand_dims(boxA[0], 0), tf.expand_dims(boxB[0], 0)], axis=0), axis=0)
        yA = tf.reduce_max(tf.concat([tf.expand_dims(boxA[1], 0), tf.expand_dims(boxB[1], 0)], axis=0), axis=0)
        xB = tf.reduce_min(tf.concat([tf.expand_dims(boxA[2], 0), tf.expand_dims(boxB[2], 0)], axis=0), axis=0)
        yB = tf.reduce_min(tf.concat([tf.expand_dims(boxA[3], 0), tf.expand_dims(boxB[3], 0)], axis=0), axis=0)

        # compute the area of intersection rectangle
        interArea = tf.nn.relu(xB - xA) * tf.nn.relu(yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return tf.maximum(0., tf.cast(iou, tf.float32))

    def class_loss_tf(self, pos_mask, class_gt, local_class_pred, num_pos, logits=False):
        pos_class_preds = tf.boolean_mask(local_class_pred, pos_mask)
        if logits:  # TODO - throws error; needs fix
            gt_expanded = tf.zeros_like(pos_class_preds)
            gt_expanded = tf.scatter_update(gt_expanded, tf.get_variable(tf.tile([class_gt + 1], [num_pos])),
                                            tf.get_variable(tf.tile([1.], [num_pos])))
            pos_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pos_class_preds,
                                                                     labels=gt_expanded)
        else:
            pos_class_loss = tf.reduce_sum(tf.log(pos_class_preds[:, class_gt + 1]))

        # Hard negative mining
        neg_mask = ~pos_mask
        neg_class_preds = tf.boolean_mask(local_class_pred, neg_mask)
        neg_class_probs = neg_class_preds[:, 0]
        num_negs = 3 * num_pos
        neg_class_probs_topk, neg_class_probs_topk_idx = tf.nn.top_k(neg_class_probs, num_negs)
        if logits:  # TODO - throws error; needs fix
            gt_expanded = tf.zeros_like(neg_class_preds)
            gt_expanded = tf.scatter_update(gt_expanded, tf.tile([0.], [num_negs]),
                                            tf.tile([1.], [num_negs]))
            neg_class_preds = tf.gather(neg_class_preds, neg_class_probs_topk_idx)
            neg_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=neg_class_preds,
                                                                     labels=gt_expanded)
        else:
            neg_class_loss = tf.reduce_sum(tf.log(neg_class_probs_topk))

        curr_class_loss = -pos_class_loss - neg_class_loss
        return curr_class_loss

    def loc_loss_tf(self, pos_priors, pos_mask, loc_gt_repeated, local_loc_pred):
        cy_cx_priors = pos_priors[:, :2]
        h_w_priors = pos_priors[:, 2:]
        cy_cx_gt = loc_gt_repeated[:, :2]
        h_w_gt = loc_gt_repeated[:, 2:]

        g_cy_cx = ((cy_cx_gt - cy_cx_priors) / h_w_priors) * 10.
        g_h_w = tf.log(h_w_gt / h_w_priors) * 5.

        pos_loc_truths = tf.concat([g_cy_cx, g_h_w], axis=1)
        pos_loc_preds = tf.boolean_mask(local_loc_pred, pos_mask)

        curr_loc_loss = tf.losses.huber_loss(pos_loc_truths, pos_loc_preds, reduction=tf.losses.Reduction.NONE)
        curr_loc_loss = tf.reduce_sum(curr_loc_loss)
        return curr_loc_loss

    def pos_match_loss_compute_tf(self, pos_priors, pos_mask, loc_gt, class_gt,
                                  local_loc_pred, local_class_pred, num_pos, alpha=1., logits=False):
        curr_loc_loss = self.loc_loss_tf(pos_priors, pos_mask, loc_gt, local_loc_pred)
        curr_class_loss = self.class_loss_tf(pos_mask, class_gt, local_class_pred, num_pos, logits=logits)
        curr_loss = tf.cast((1 / num_pos), tf.float32) * (curr_class_loss + alpha * curr_loc_loss)
        return curr_loss

    def single_gt_loss_tf(self, local_gt, local_loc_pred, local_class_pred, box_priors, iou_thresh=.5, alpha=1.,
                          logits=False, num_box_points=4):
        local_loc_gt = local_gt[:, :num_box_points]
        local_class_gt = local_gt[:, -1]
        loc_gt_repeated = local_loc_gt
        class_gt = tf.cast(local_class_gt[0], tf.int32)

        loc_gt_prior_ious = self.compute_iou_centered_yxhw_vectorized_tf(loc_gt_repeated, box_priors)
        pos_mask = loc_gt_prior_ious > iou_thresh

        pos_priors = tf.boolean_mask(box_priors, pos_mask)
        num_pos = tf.shape(pos_priors)[0]

        loss_compute_fn = lambda: self.pos_match_loss_compute_tf(pos_priors, pos_mask, loc_gt_repeated[:num_pos],
                                                                 class_gt, local_loc_pred, local_class_pred, num_pos,
                                                                 alpha=alpha, logits=logits)
        curr_loss = tf.cond(num_pos > 0, loss_compute_fn, lambda: 0.)
        return curr_loss

    def loss_op_tf_worker(self, loc_preds, class_preds, num_gts, in_batch_idx, num_gt, gts, preds_per_img, box_priors,
                          iou_thresh=.5, alpha=1., logits=False):
        start_idx = tf.reduce_sum(num_gts[:in_batch_idx])
        end_idx = start_idx + num_gt
        local_gt = gts[start_idx: end_idx]
        local_gt_repeated_raw = tf.tile(local_gt, [preds_per_img, 1])

        extraction_idx_gen = lambda i: tf.range(i, preds_per_img * num_gt, num_gt)
        extraction_indices = tf.map_fn(extraction_idx_gen, tf.range(0, num_gt, 1))
        extractor_fn = lambda indices: tf.gather(local_gt_repeated_raw, indices)
        local_gt_repeated = tf.map_fn(extractor_fn, extraction_indices, dtype=tf.float32)

        local_loc_pred_raw = loc_preds[in_batch_idx]
        local_loc_pred = tf.reshape(tf.tile(local_loc_pred_raw, [num_gt, 1]), [num_gt, preds_per_img, -1])

        local_class_pred_raw = class_preds[in_batch_idx]
        local_class_pred = tf.reshape(tf.tile(local_class_pred_raw, [num_gt, 1]), [num_gt, preds_per_img, -1])

        curr_img_loss_fn = lambda idx: self.single_gt_loss_tf(local_gt_repeated[idx], local_loc_pred[idx],
                                                              local_class_pred[idx], box_priors,
                                                              iou_thresh=iou_thresh, alpha=alpha, logits=logits)
        curr_img_losses = tf.map_fn(curr_img_loss_fn, tf.range(0, num_gt), dtype=tf.float32)
        curr_img_loss = tf.reduce_mean(curr_img_losses)
        return curr_img_loss

    def loss_op_tf(self, loc_preds, class_preds, gts, num_gts, box_priors_py, iou_thresh=.5, alpha=1., logits=False):
        box_priors = tf.constant(box_priors_py, dtype=tf.float32)
        batch_losses = []
        preds_per_img = tf.shape(box_priors)[0]
        for in_batch_idx in range(self.train_batch_size):
            num_gt = num_gts[in_batch_idx]
            curr_img_loss_fn = lambda: self.loss_op_tf_worker(loc_preds, class_preds,
                                                              num_gts, in_batch_idx, num_gt, gts,
                                                              preds_per_img, box_priors,
                                                              iou_thresh=iou_thresh, alpha=alpha,
                                                              logits=logits)
            curr_img_loss = tf.cond(num_gt > 0, curr_img_loss_fn, lambda: 0.)
            batch_losses.append(curr_img_loss)
        curr_loss = tf.reduce_mean(batch_losses)
        return curr_loss

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
        y_gts, y_num_gts = y
        if self.dropout_enabled:
            loss, _, step_tf, lr = self.sess.run([self.loss_op, self.train_op, self.step_ph, self.learn_rate_tf],
                                                 feed_dict={self.x_tensor: x,
                                                            self.y_tensor: y_gts,
                                                            self.num_y_bboxes_per_image_tensor: y_num_gts,
                                                            self.dropout_rate_tensor: self.dropout_rate})
        else:
            loss, _, step_tf, lr = self.sess.run([self.loss_op, self.train_op, self.step_ph, self.learn_rate_tf],
                                                 feed_dict={self.x_tensor: x,
                                                            self.y_tensor: y_gts,
                                                            self.num_y_bboxes_per_image_tensor: y_num_gts})
        self.step = step_tf
        return loss, step_tf, lr

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=True, activation=tf.nn.relu6, pooling=True, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.avg_pool, block_depth=1, make_residual=True):
        if not batch_norm:
            use_bias = True
        else:
            use_bias = False
        curr_layer_var_mappings_ordered = []
        curr_layer = []
        layer_out = x_in
        if block_depth == 1:
            make_residual = False
        for depth in range(block_depth):
            v0 = tf.global_variables()
            layer_out = tf.layers.conv2d(layer_out, output_filters, kernel_size, strides=kernel_stride,
                                         use_bias=use_bias, activation=activation, dilation_rate=dilation,
                                         padding=padding)
            v1 = tf.global_variables()
            layer_vars = v1[len(v0):]
            layer_var_mapping = {layer_out.name: layer_vars}
            curr_layer_var_mappings_ordered.append(layer_var_mapping)
            curr_layer.append(layer_out)
            if pooling:
                layer_out = pooling_fn(layer_out, ksize=[1, pool_ksize, pool_ksize, 1],
                                       strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
                curr_layer.append(layer_out)
            if batch_norm:
                v0 = tf.global_variables()
                layer_out = tf.layers.batch_normalization(layer_out, training=self.compute_bn_mean_var)
                v1 = tf.global_variables()
                layer_vars = v1[len(v0):]
                layer_var_mapping = {layer_out.name: layer_vars}
                curr_layer_var_mappings_ordered.append(layer_var_mapping)
                curr_layer.append(layer_out)
            if depth == 0:
                residual_input = layer_out
            output = layer_out
        if make_residual:
            output = output + tf.image.resize_bilinear(residual_input, output.shape[1:3])
            curr_layer.append(output)
            if batch_norm:
                v0 = tf.global_variables()
                output = tf.layers.batch_normalization(output, training=self.compute_bn_mean_var)
                v1 = tf.global_variables()
                layer_vars = v1[len(v0):]
                layer_var_mapping = {output.name: layer_vars}
                curr_layer_var_mappings_ordered.append(layer_var_mapping)
                curr_layer.append(output)
        if self.dropout_enabled:
            output = tf.nn.dropout(output, rate=self.dropout_rate_tensor)
            curr_layer.append(output)
        self.layer_var_mappings_ordered.append(curr_layer_var_mappings_ordered)
        self.layers.append(curr_layer)
        return output

    def dense_block(self, x_in, num_outs, batch_norm=True, biased=False):
        curr_layer_var_mappings_ordered = []
        curr_layer = []
        v0 = tf.global_variables()
        layer_outs = tf.layers.dense(x_in, num_outs, use_bias=biased)
        v1 = tf.global_variables()
        layer_vars = v1[len(v0):]
        layer_var_mapping = {layer_outs.name: layer_vars}
        curr_layer_var_mappings_ordered.append(layer_var_mapping)
        curr_layer.append(layer_outs)
        layer_outs = tf.nn.relu6(layer_outs)
        curr_layer.append(layer_outs)
        if batch_norm:
            v0 = tf.global_variables()
            layer_outs = tf.layers.batch_normalization(layer_outs, training=self.compute_bn_mean_var)
            v1 = tf.global_variables()
            layer_vars = v1[len(v0):]
            layer_var_mapping = {layer_outs.name: layer_vars}
            curr_layer_var_mappings_ordered.append(layer_var_mapping)
            curr_layer.append(layer_outs)
        if self.dropout_enabled:
            layer_outs = tf.nn.dropout(layer_outs, rate=self.dropout_rate_tensor)
            curr_layer.append(layer_outs)
        self.layer_var_mappings_ordered.append(curr_layer_var_mappings_ordered)
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
        curr_layer_var_mappings_ordered = []
        curr_layer = []
        v0 = tf.global_variables()
        net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=[3, 3], stride=1,
                                    scope='BoxPredictor_' + str(scope_idx) + '/' + name + '_depthwise',
                                    padding='SAME', depth_multiplier=1, normalizer_fn=None,
                                    activation_fn=None, biases_initializer=None)
        v1 = tf.global_variables()
        layer_vars = v1[len(v0):]
        layer_var_mapping = {net.name: layer_vars}
        curr_layer_var_mappings_ordered.append(layer_var_mapping)
        curr_layer.append(net)
        if self.compute_bn_mean_var:
            v0 = tf.global_variables()
            net = tf.layers.batch_normalization(net, training=self.compute_bn_mean_var)
            v1 = tf.global_variables()
            layer_vars = v1[len(v0):]
            layer_var_mapping = {net.name: layer_vars}
            curr_layer_var_mappings_ordered.append(layer_var_mapping)
            curr_layer.append(net)
        v0 = tf.global_variables()
        net = slim.conv2d(inputs=net, num_outputs=out_channels, kernel_size=[1, 1],
                          scope='BoxPredictor_' + str(scope_idx) + '/' + name,
                          stride=1, activation_fn=tf.nn.relu6, normalizer_fn=None)
        v1 = tf.global_variables()
        layer_vars = v1[len(v0):]
        layer_var_mapping = {net.name: layer_vars}
        curr_layer_var_mappings_ordered.append(layer_var_mapping)
        curr_layer.append(net)
        if self.compute_bn_mean_var:
            v0 = tf.global_variables()
            net = tf.layers.batch_normalization(net, training=self.compute_bn_mean_var)
            v1 = tf.global_variables()
            layer_vars = v1[len(v0):]
            layer_var_mapping = {net.name: layer_vars}
            curr_layer_var_mappings_ordered.append(layer_var_mapping)
            curr_layer.append(net)
        if self.dropout_enabled:
            output = tf.nn.dropout(net, rate=self.dropout_rate_tensor)
            curr_layer.append(output)
        self.layer_var_mappings_ordered.append(curr_layer_var_mappings_ordered)
        self.layers.append(curr_layer)
        return net

    def gen_anchors(self, fmap_sizes, anchor_bbox_base_dims=None):
        num_fmaps = len(fmap_sizes)
        if anchor_bbox_base_dims is None:
            zero_idx_dims = np.array([[.1, .1],
                                      [0.14142136, 0.28284273],
                                      [0.28284273, 0.14142136]])
            first_idx_dims = np.array([[0.25      , 0.25      ],
                                       [0.17677669, 0.35355338],
                                       [0.35355338, 0.17677669],
                                       [0.14433756, 0.43301269],
                                       [0.43303436, 0.14433035],
                                       [0.28504387, 0.28504387]])
            last_idx_dims = (first_idx_dims.T * ([3.2]*5 + [3.06])).T
            nz_idx_dims = [first_idx_dims, last_idx_dims]
        out_anchor_bboxes = None
        for i in range(num_fmaps):
            grid_elem_size = 1. / fmap_sizes[i]
            range_start = grid_elem_size / 2
            range_end = 1. - range_start
            idx_range = np.linspace(range_start, range_end, fmap_sizes[i])
            for cy in idx_range:
                for cx in idx_range:
                    if i > 0:
                        my_anchors_hw = nz_idx_dims[i - 1]
                    else:
                        my_anchors_hw = zero_idx_dims
                    cy_cx_tiled = np.tile([cy, cx], [my_anchors_hw.shape[0], 1])
                    curr_gridbox_anchors = np.hstack([cy_cx_tiled, my_anchors_hw])
                    if out_anchor_bboxes is None:
                        out_anchor_bboxes = curr_gridbox_anchors
                    else:
                        out_anchor_bboxes = np.vstack([out_anchor_bboxes, curr_gridbox_anchors])
        return out_anchor_bboxes

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
        return detection_out, classification_out_softmax, classification_out

    def init_nn_graph(self):
        layer_outs = self.conv_block(self.x_tensor, 8)
        layer_outs = self.conv_block(layer_outs, 32, pool_ksize=4, pool_stride=1, block_depth=3)
        layer_outs = self.conv_block(layer_outs, 64, pool_ksize=4, pool_stride=2, block_depth=2)
        layer_outs = self.conv_block(layer_outs, 128, pooling=False)
        layer_outs = self.conv_block(layer_outs, 16, pool_ksize=4, pool_stride=2, block_depth=3)
        stop_grad_vars = tf.global_variables()
        self.ssd_endpoints = [self.layers[4][1], self.layers[4][4], self.layers[4][7]]

        v0 = tf.global_variables()
        detection_out, classification_out_softmax, classification_out = self.ssdlite_nn(self.ssd_endpoints)
        v1 = tf.global_variables()
        ssdlite_vars = v1[len(v0):]

        trainable_vars = tf.trainable_variables()
        restore_excluded_vars = ssdlite_vars
        layer_outs = [detection_out, classification_out_softmax, classification_out]
        return layer_outs, trainable_vars, stop_grad_vars, restore_excluded_vars
