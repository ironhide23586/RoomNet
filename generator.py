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


import logging
from threading import Thread
from queue import Queue
import time
import json

import numpy as np
import cv2


class DataFeeder:

    def __init__(self, data_preprocesor, split, batch_size=8, preprocess=True, shuffle=True,
                 batches_per_queue=40, random_crop=True, im_side=300):
        self.data_preprocessor = data_preprocesor
        if split == 'train':
            self.input_fpaths = self.data_preprocessor.train_list
        else:
            self.input_fpaths = self.data_preprocessor.val_list
        self.input_fpaths = np.array(self.input_fpaths)
        logging.info('Initializing DataFeeder Object on ' + str(self.input_fpaths.shape[0]) + ' data objects')
        self.random_crop = random_crop
        self.im_side = im_side
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = 0
        self.batch_iters = 0
        self.data_preprocess = preprocess
        self.epoch_size_total = self.input_fpaths.shape[0]
        if self.batch_size > self.epoch_size_total:
            logging.warning('Batch size exceeds epoch size, setting batch size to epoch size')
            self.batch_size = self.epoch_size_total
        self.batches_per_epoch = self.epoch_size_total // self.batch_size
        self.epoch_size = self.batch_size * self.batches_per_epoch
        self.batch_data_x = []
        self.batch_data_y = []
        self.batch_data_x_fpaths = []
        self.batch_data_y_fpaths = []
        self.batch_fpaths = []
        self.im_coords = None
        self.total_iters = 0
        self.train_state = {'epoch': 1, 'batch': 0, 'total_iters': 0,
                            'previous_epoch_done': False}
        self.start_batch_queue_populater(batches_per_queue=batches_per_queue)

    def random_sliding_square_crop(self, im):
        h, w, _ = im.shape
        if h == w:
            return im.copy(), [0, w], [0, h]
        min_dim = w
        max_dim = h
        if h < w:
            min_dim = h
            max_dim = w
        start_idx = np.random.randint(max_dim - min_dim)
        end_idx = start_idx + min_dim
        if h < w:
            width_range = [start_idx, end_idx]
            height_range = [0, h]
        else:
            width_range = [0, w]
            height_range = [start_idx, end_idx]
        if h < w:
            im_cropped = im[:, start_idx:end_idx, :]
        else:
            im_cropped = im[start_idx:end_idx, :, :]
        return im_cropped, width_range, height_range

    def center_crop(self, x):
        h, w, _ = x.shape
        offset = abs((w - h) // 2)
        if h < w:
            x_pp = x[:, offset:offset + h, :]
            width_range = [offset, offset + h]
            height_range = [0, h]
        elif w < h:
            x_pp = x[offset:offset + w, :, :]
            width_range = [0, w]
            height_range = [offset, offset + h]
        else:
            x_pp = x.copy()
            width_range = [0, w]
            height_range = [0, h]
        return x_pp, width_range, height_range

    def preprocess_set(self, x, y):
        h, w, _ = x.shape
        if self.random_crop:
            x_raw = self.random_sliding_square_crop(x)
        else:
            x_raw = self.center_crop(x)
        x_pp, w_rng, h_rng = x_raw
        org_side = x_pp.shape[0]
        x_pp = cv2.resize(x_pp, (self.im_side, self.im_side))
        # if self.data_preprocess:
        #     # angle = np.random.uniform(0, 360)
        #     # x_pp = np.array(Image.fromarray(x_pp).rotate(angle))
        #     if np.random.uniform() > .5:
        #         x_pp = np.fliplr(x_pp)
        #     if np.random.uniform() > .5:
        #         x_pp = np.flipud(x_pp)
        filtered_ys = []
        y_pp = []
        target_label_ids = [self.data_preprocessor.readable2labelid_map[self.data_preprocessor.target_labels[i]]
                            for i in range(len(self.data_preprocessor.target_labels))]
        for det in y:
            if det['LabelName'] in target_label_ids:
                filtered_ys.append(det)
        num_dets = 0
        for det in filtered_ys:
            xmin = np.clip(w * det['XMin'], w_rng[0], w_rng[1]) - w_rng[0]
            ymin = np.clip(h * det['YMin'], h_rng[0], h_rng[1]) - h_rng[0]
            xmax = np.clip(w * det['XMax'], w_rng[0], w_rng[1]) - w_rng[0]
            ymax = np.clip(h * det['YMax'], h_rng[0], h_rng[1]) - h_rng[0]
            bbox_h = ymax - ymin
            bbox_w = xmax - xmin
            area = bbox_w * bbox_h
            if area < 0:
                continue
            bbox_cy = ymin + bbox_h / 2
            bbox_cx = xmin + bbox_w / 2
            yxhw_bbox_normalized = np.array([bbox_cy, bbox_cx, bbox_h, bbox_w]) / org_side
            yxhw_bbox = (yxhw_bbox_normalized * self.im_side).astype(np.int)

            tlxy = tuple((yxhw_bbox[:2] - yxhw_bbox[2:] / 2)[[1, 0]].astype(np.int))
            brxy = tuple((yxhw_bbox[:2] + yxhw_bbox[2:] / 2)[[1, 0]].astype(np.int))
            label_name_readable = self.data_preprocessor.labelid2readable_map[det['LabelName']]
            cv2.rectangle(x_pp, tlxy, brxy, (0, 255, 0), 2)

            class_idx = self.data_preprocessor.label_mappings[det['LabelName']]['class_idx']
            y_pp.append(list(yxhw_bbox_normalized) + [class_idx])
            num_dets += 1
        y_pp = np.array(y_pp)
        y_gt = y_pp, num_dets
        return x_pp, y_gt

    def read_x(self, fpath):
        return cv2.imread(fpath)

    def read_y(self, fpath):
        return json.load(open(fpath, 'r'))

    def fpath2data(self, batch_fpaths):
        batch_data_x = []
        batch_data_y_bboxes = None
        batch_data_y_counts = []
        batch_data_x_fpaths = []
        batch_data_y_fpaths = []
        for fpath_set in batch_fpaths:
            x_path, y_path = fpath_set.strip().split(' ')
            x_data = self.read_x(x_path)
            y_data = self.read_y(y_path)
            batch_data_x_fpaths.append(x_path)
            batch_data_y_fpaths.append(y_path)
            x_data, y_data = self.preprocess_set(x_data, y_data)
            y_bboxes, y_cnts = y_data
            if batch_data_y_bboxes is None:
                batch_data_y_bboxes = y_bboxes
            else:
                batch_data_y_bboxes = np.vstack([batch_data_y_bboxes, y_bboxes])
            batch_data_y_counts.append(y_cnts)
            batch_data_x.append(x_data)
        batch_data_x = np.array(batch_data_x)
        batch_data_x_fpaths = np.array(batch_data_x_fpaths)
        batch_data_y = batch_data_y_bboxes, batch_data_y_counts
        return batch_data_x, batch_data_y, batch_data_x_fpaths, batch_data_y_fpaths

    def get_data(self, batch_size=None):
        if batch_size is not None:
            logging.info('External batch_size provided, recomputing batches_per_epoch and epoch_size')
            self.batch_size = batch_size
            if self.batch_size > self.epoch_size_total:
                logging.warning('Batch size exceeds epoch size, setting batch size to epoch size')
                self.batch_size = self.epoch_size_total
            self.batches_per_epoch = self.epoch_size_total // self.batch_size
            self.epoch_size = self.batch_size * self.batches_per_epoch
        self.batch_iters += 1
        self.total_iters += 1
        self.epoch_completed = False
        if self.batch_iters > self.batches_per_epoch:
            logging.info('---------> Loading new epoch in to Buffered Batch Reader')
            self.epoch_completed = True
            self.batch_iters = 1
            self.epochs += 1
            if self.shuffle:
                logging.info('Shuffle enabled, so shuffling input at epoch-level')
                np.random.shuffle(self.input_fpaths)
        train_state = {'epoch': self.epochs + 1, 'batch': self.batch_iters, 'total_iters': self.total_iters,
                       'previous_epoch_done': self.epoch_completed}
        start_idx = (self.batch_iters - 1) * self.batch_size
        end_idx = start_idx + self.batch_size
        self.batch_fpaths = self.input_fpaths[start_idx:end_idx]
        batch_data_x, batch_data_y, batch_data_x_fpaths, batch_data_y_fpaths = self.fpath2data(self.batch_fpaths)
        # try:
        #     batch_data_x, batch_data_y, batch_data_x_fpaths, batch_data_y_fpaths = self.fpath2data(self.batch_fpaths)
        # except:
        #     print(self.batch_fpaths)
        return batch_data_x, batch_data_y, batch_data_x_fpaths, batch_data_y_fpaths, train_state

    def __queue_filler_process(self):
        while True:
            if self.buffer.full():
                # logging.info('Buffer full, sleeping for 2 seconds')
                time.sleep(2)
                continue
            self.buffer.put(self.get_data())
            # try:
            #     self.buffer.put(self.get_data())
            # except:
            #     logging.error('Buffer Load error!, retrying...')
            #     continue

    def start_batch_queue_populater(self, batches_per_queue=20):
        logging.info('Starting Populator process for batch buffer')
        self.buffer = Queue(maxsize=batches_per_queue)
        self.queue_filler_thread = Thread(target=self.__queue_filler_process)
        self.queue_filler_thread.start()

    def dequeue(self):
        if not self.buffer.empty():
            self.batch_data_x, self.batch_data_y, self.batch_data_x_fpaths, self.batch_data_y_fpaths, \
            self.train_state = self.buffer.get()
            if self.train_state['previous_epoch_done']:
                logging.info('----------------EPOCH ' + str(self.train_state['epoch'] - 1)
                             + ' COMPLETE----------------')
            logging.info('Epoch ' + str(self.train_state['epoch']) + ', Batch ' + str(self.train_state['batch']))
            return self.batch_data_x, self.batch_data_y
        else:
            logging.warning('Buffer empty, waiting for it to repopulate..')
            while self.buffer.empty():
                continue
            return self.dequeue()
