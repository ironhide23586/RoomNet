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
import json
import shutil
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2

DATA_DIR = './data/Google-Open-Images'


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class GoogleOpenBboxPreprocessor:

    def __init__(self, data_dir, parse=False, images_dirname=None):
        self.data_dir = data_dir
        if images_dirname is None:
            self.input_img_dir = os.sep.join([self.data_dir, 'images', 'train_f'])
            images_dirname = 'train_f'
        else:
            self.input_img_dir = os.sep.join([self.data_dir, 'images', images_dirname])
        self.input_label_dir = self.data_dir + os.sep + 'labels-bbox' + '-' + images_dirname
        self.input_label_viz_dir = self.input_label_dir + '-viz'
        self.train_list_fpath = self.data_dir + os.sep + 'train_list-' + images_dirname + '.txt'
        self.val_list_fpath = self.data_dir + os.sep + 'val_list.txt-' + images_dirname + '.txt'
        self.label_mappings_json_fpath = self.data_dir + os.sep + 'label_mappings-' + images_dirname + '.json'
        self.label_mappings = None
        self.train_
        if os.path.isfile(self.label_mappings_json_fpath):
            self.label_mappings = json.load(open(self.label_mappings_json_fpath, 'r'))
        if os.path.isfile(self.train_list_fpath):


        self.label_file_path = self.data_dir + os.sep + 'train-annotations-bbox.csv'
        self.label_desc_path = self.data_dir + os.sep + 'class-descriptions-boxable.csv'

        self.input_img_paths = glob(self.input_img_dir + os.sep + '*')
        self.input_img_ids = [p.split(os.sep)[-1].split('.')[0] for p in self.input_img_paths]
        self.parsed_json = {}
        self.label_dataframe = None
        self.colnames = []
        self.labelid2readable_map = {}
        self.readable2labelid_map = {}
        self.extract_readable_labels()
        self.label_filter_mask = None
        if parse:
            self.parse_label_data()

    def extract_readable_labels(self):
        with open(self.label_desc_path, 'r') as f:
            raw_data = f.readlines()
        for line in raw_data:
            label_id, label_readable = line.strip().split(',')
            self.readable2labelid_map[label_readable] = label_id
            self.labelid2readable_map[label_id] = label_readable

    def extract_labels_for_id(self, img_id):
        label_fpath = self.input_label_dir + os.sep + img_id + '.json'
        if img_id not in self.input_img_ids or os.path.isfile(label_fpath):
            return
        img_df = self.label_dataframe[self.label_dataframe.ImageID == img_id]
        data_colnames = img_df.columns[1:]
        mat_data = img_df.as_matrix()[:, 1:]
        res = []
        for md in mat_data:
            entry = dict(zip(data_colnames, md))
            res.append(entry)
        json.dump(res, open(label_fpath, 'w'), sort_keys=True, indent=4)

    def viz_id(self, img_id_idx):
        img_id = self.input_img_ids[img_id_idx]
        img_fpath = self.input_img_paths[img_id_idx]
        viz_fpath = self.input_label_viz_dir + os.sep + img_id + '.jpg'
        if os.path.isfile(viz_fpath):
            return
        label_fpath = self.input_label_dir + os.sep + img_id + '.json'
        label_data = json.load(open(label_fpath, 'r'))
        im = cv2.imread(img_fpath)
        h, w, _ = im.shape
        for label in label_data:
            label_name_readable = self.labelid2readable_map[label['LabelName']]
            tlxy_brxy_bbox = (np.array([[label['XMin'], label['YMin']],
                                       [label['XMax'], label['YMax']]]) * (w, h)).astype(np.int)
            cv2.rectangle(im, tuple(tlxy_brxy_bbox[0]), tuple(tlxy_brxy_bbox[1]), (0, 255, 0), 3)
            cv2.putText(im, label_name_readable, tuple(tlxy_brxy_bbox[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (200, 244, 210), 3, cv2.LINE_AA)
        cv2.imwrite(viz_fpath, im)

    def parse_label_data(self, parallel=True):
        force_makedir(self.input_label_dir)
        if len(self.input_img_ids) == len(glob(self.input_label_dir + os.sep + '*')):
            print('Label data already parsed..')
            return
        self.label_dataframe = pd.read_csv(self.label_file_path)
        img_ids = self.input_img_ids
        if not parallel:
            for i in tqdm(range(len(img_ids))):
                self.extract_labels_for_id(img_ids[i])
        else:
            p = Pool(cpu_count())
            p.map(self.extract_labels_for_id, img_ids)
            p.close()

    def viz_labels(self, parallel=True, input_img_ids=None):
        force_makedir(self.input_label_viz_dir)
        if len(self.input_img_ids) == len(glob(self.input_label_viz_dir + os.sep + '*')):
            print('Label visualizations already present..')
            return
        if not parallel:
            for i in tqdm(range(len(self.input_img_ids))):
                self.viz_id(i)
        else:
            p = Pool(cpu_count())
            p.map(self.viz_id, np.arange(len(self.input_img_ids)))
            p.close()

    def label_filter(self, img_id_idx, label_ids, bin_dir_name, copy=True):
        img_id = self.input_img_ids[img_id_idx]
        img_fpath = self.input_img_paths[img_id_idx]
        label_fpath = self.input_label_dir + os.sep + img_id + '.json'
        label_data = json.load(open(label_fpath, 'r'))
        label_ids_gt = [lbl['LabelName'] for lbl in label_data]
        is_present = False
        for id in label_ids:
            is_present = is_present or id in label_ids_gt
        if copy and is_present:
            shutil.copy(img_fpath, bin_dir_name)

    def filter_label_data(self, bin_name, filter_labels_readable, parallel=False, copy=True):
        label_ids = [self.readable2labelid_map[lbl] for lbl in filter_labels_readable]
        self.label_filter_mask = [False] * len(self.input_img_ids)
        bin_dir_path = self.input_img_dir + '-' + bin_name
        if copy:
            force_makedir(bin_dir_path)
        if not parallel:
            for i in tqdm(range(len(self.input_img_ids))):
                self.label_filter(i, label_ids, bin_dir_path, copy=copy)
        else:
            args = [(i, label_ids, bin_dir_path, copy) for i in range(len(self.input_img_ids))]
            p = Pool(cpu_count())
            p.starmap(self.label_filter, args)
            p.close()

    def __gen_list_lines(self, x_fpaths):
        lines = [x_fpath + ' ' + self.input_label_dir + os.sep
                 + '.'.join(x_fpath.split(os.sep)[-1].split('.')[:-1]) + '.json\n' for x_fpath in x_fpaths]
        return lines

    def gen_train_val_lists(self, target_labels, train_frac=.8, shuffle=True):
        if os.path.isfile(self.train_list_fpath) and os.path.isfile(self.val_list_fpath):
            print('Training and validation fpath lists found!, reading from them')
            with open(self.train_list_fpath, 'r') as f:
                all_train_txt = f.readlines()
            with open(self.val_list_fpath, 'r') as f:
                all_val_txt = f.readlines()
        else:
            if shuffle:
                idx = np.arange(len(self.input_img_ids))
                np.random.shuffle(idx)
                self.input_img_ids = list(np.array(self.input_img_ids)[idx])
                self.input_img_paths = list(np.array(self.input_img_paths)[idx])
            num_train_images = int(train_frac * len(self.input_img_ids))
            train_x_fpaths = self.input_img_paths[:num_train_images]
            val_x_fpaths = self.input_img_paths[num_train_images:]
            train_lines = self.__gen_list_lines(train_x_fpaths)
            val_lines = self.__gen_list_lines(val_x_fpaths)
            with open(self.train_list_fpath, 'w') as f:
                f.writelines(train_lines)
            with open(self.val_list_fpath, 'w') as f:
                f.writelines(val_lines)



if __name__ == '__main__':
    bb_pp = GoogleOpenBboxPreprocessor(DATA_DIR, images_dirname='Bathtub_Shower')

    print('Parsing Label Data...')
    bb_pp.parse_label_data(parallel=False)
    print('Done!')

    print('Generating GT visualizations...')
    bb_pp.viz_labels(parallel=True)
    print('Done!')

    print('Generating train-val lists...')
    bb_pp.gen_train_val_lists(['Bathroom', 'Shower'])
    print('Done!')
