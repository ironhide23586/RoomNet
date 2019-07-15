'''
Author: Souham Biswas
Email: souham.biswas@outlook.com
GitHub: https://github.com/ironhide23586
LinkedIn: https://www.linkedin.com/in/souham

I'm not responsible if your machine catches fire.
'''


import os
from glob import glob
import json

import numpy as np

from generator import TrainFeeder
from network import RoomNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DATA_DIR = './REI-Dataset'
TRAIN_LIST_FPATH = 'train_list.txt'
VAL_LIST_FPATH = 'val_list.txt'
TRAIN_STATS_FILE = 'all_train_stats.json'
IMG_SIDE = 224

TRAIN_BATCH_SIZE = 45
TRAIN_STEPS = 100000
SAVE_FREQ = 100
LEARN_RATE = 2e-4
DROPOUT_ENABLED = True
DROPOUT_RATE = .35
L2_REGULARIZATION_COEFF = 6e-2
UPDATE_BATCHNORM_MOVING_VARS = False


def extract_fpaths(data_dir):
    if os.path.isfile(TRAIN_LIST_FPATH) and os.path.isfile(VAL_LIST_FPATH):
        print('Training and validation fpath lists found!, reading from them')
        with open(TRAIN_LIST_FPATH, 'r') as f:
            all_train_txt = f.readlines()
        with open(VAL_LIST_FPATH, 'r') as f:
            all_val_txt = f.readlines()
        return all_train_txt, all_val_txt
    print('Training and validation fpath lists not found!, generating them')
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
    train_data_reader = TrainFeeder(train_fpaths, batch_size=TRAIN_BATCH_SIZE, batches_per_queue=40, shuffle=True,
                                    im_side=IMG_SIDE, random_crop=True)
    val_data_reader = TrainFeeder(val_fpaths, batch_size=64, batches_per_queue=10, shuffle=False,
                                  im_side=IMG_SIDE, random_crop=False)

    nn = RoomNet(num_classes=6, im_side=IMG_SIDE, num_steps=TRAIN_STEPS, learn_rate=LEARN_RATE,
                 dropout_rate=DROPOUT_RATE, l2_regularizer_coeff=L2_REGULARIZATION_COEFF,
                 dropout_enabled=DROPOUT_ENABLED, update_batchnorm_means_vars=UPDATE_BATCHNORM_MOVING_VARS,
                 compute_bn_mean_var=False)
    nn.init()
    nn.load()
    if os.path.isfile(TRAIN_STATS_FILE):
        all_train_stats = json.load(open(TRAIN_STATS_FILE, 'r'))
    else:
        all_train_stats = []
    for train_iter in range(nn.start_step, nn.start_step + TRAIN_STEPS):
        if train_iter % SAVE_FREQ == 0:# and train_iter > nn.start_step:
            x_val, y_val = val_data_reader.dequeue()
            y_vals = list(y_val)
            y_preds = []
            print('Validating model at step', nn.step)
            while not val_data_reader.train_state['previous_epoch_done']:
                y_pred = nn.infer(x_val)
                y_preds += list(y_pred)
                x_val, y_val = val_data_reader.dequeue()
                y_vals += list(y_val)
            print('Inference Complete!')
            y_vals = y_vals[:len(y_preds)]
            acc = accuracy_score(y_vals, y_preds)
            prec, rec, fsc, supp = precision_recall_fscore_support(y_vals, y_preds)
            nn.save(suffix=str(acc))
            train_stats = {'step': int(nn.step), 'accuracy': float(acc),
                           'precisions': list(map(float, list(prec))),
                           'recalls': list(map(float, list(rec))),
                           'f-scores': list(map(float, list(fsc)))}
            all_train_stats.append(train_stats)
            print('Dumping train stats to', TRAIN_STATS_FILE)
            json.dump(all_train_stats, open(TRAIN_STATS_FILE, 'w'), indent=4, sort_keys=True)
        x, y = train_data_reader.dequeue()
        loss, train_step, learn_rate = nn.train_step(x, y)
        print('Step', train_step, 'loss =', loss, 'learn_rate =', learn_rate)
