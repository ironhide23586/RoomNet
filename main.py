import os
from glob import glob

import tensorflow as tf
import numpy as np
import cv2

from generator import TrainFeeder
from network import RoomNet


DATA_DIR = './REI-Dataset'
TRAIN_LIST_FPATH = 'train_list.txt'
VAL_LIST_FPATH = 'val_list.txt'
IMG_SIDE = 300


def extract_fpaths(data_dir):
    if os.path.isfile(TRAIN_LIST_FPATH) and os.path.isfile(VAL_LIST_FPATH):
        print('Training and validattion fpath lists found!, reading from them')
        with open(TRAIN_LIST_FPATH, 'r') as f:
            all_train_txt = f.readlines()
        with open(VAL_LIST_FPATH, 'r') as f:
            all_val_txt = f.readlines()
        return all_train_txt, all_val_txt
    print('Training and validattion fpath lists not found!, generating them')
    class_dirs = glob(data_dir + '/*')
    path_mappings = {}
    class_sizes = []
    labels = []
    name_id_mappings = {}
    for i in range(len(class_dirs)):
        class_dir = class_dirs[i]
        class_fpaths = glob(class_dir + '/*')
        key = class_dir.split(os.sep)[-1]
        path_mappings[key] = class_fpaths
        class_sizes.append(len(class_fpaths))
        labels.append(key)
        name_id_mappings[key] = i
    smallest_class_id = np.argmin(class_sizes)
    smallest_class_size = class_sizes[smallest_class_id]
    train_path_mappings = {}
    val_path_mappings = {}
    train_class_size = int(.9 * smallest_class_size)
    for i in range(len(class_dirs)):
        paths = path_mappings[labels[i]].copy()
        np.random.shuffle(paths)
        train_paths = paths[:train_class_size]
        val_paths = paths[train_class_size:]
        train_path_mappings[labels[i]] = train_paths
        val_path_mappings[labels[i]] = val_paths
    all_train_txt = []
    all_val_txt = []
    for i in range(len(class_dirs)):
        train_txt = [train_path_mappings[labels[i]][j] + ' ' + str(i) + '\n'
                     for j in range(len(train_path_mappings[labels[i]]))]
        val_txt = [val_path_mappings[labels[i]][j] + ' ' + str(i) + '\n'
                   for j in range(len(val_path_mappings[labels[i]]))]
        all_train_txt += train_txt
        all_val_txt += val_txt
    np.random.shuffle(all_train_txt)
    np.random.shuffle(all_val_txt)
    with open(TRAIN_LIST_FPATH, 'w') as f:
        f.writelines(all_train_txt)
    with open(VAL_LIST_FPATH, 'w') as f:
        f.writelines(all_val_txt)
    return all_train_txt, all_val_txt


if __name__ == '__main__':
    train_fpaths, val_fpaths = extract_fpaths(DATA_DIR)
    train_data_reader = TrainFeeder(train_fpaths, batch_size=32, batches_per_queue=40, shuffle=True,
                                    im_side=IMG_SIDE, random_crop=True)
    val_data_reader = TrainFeeder(train_fpaths, batch_size=32, batches_per_queue=40, shuffle=False,
                                  im_side=IMG_SIDE, random_crop=False)

    x, y = train_data_reader.dequeue()
    nn = RoomNet(num_classes=6, im_side=IMG_SIDE)
    k = 0