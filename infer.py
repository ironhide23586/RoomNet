from glob import glob
import os
import shutil

import cv2
from tqdm import tqdm

from network import RoomNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CLASS_LABELS = ['Backyard', 'Bathroom', 'Bedroom', 'Frontyard', 'Kitchen', 'LivingRoom']

INPUT_MODEL_PATH = './final_model/roomnet'
IMG_SIDE = 224

INPUT_IMG_PATH_LIST_FILE = 'val_list.txt'


def read_fpaths(list_fpath):
    with open(list_fpath, 'r') as f:
        data = f.readlines()
    fpath_components = [fpath_set.strip().split(' ') for fpath_set in data]
    im_paths = [' '.join(fpath_component[:-1]) for fpath_component in fpath_components]
    class_id = [int(fpath_component[-1]) for fpath_component in fpath_components]
    n = len(class_id)
    return im_paths, class_id, n


def groundtruth_validation(nn):
    fpaths, labels, num_fpaths = read_fpaths(INPUT_IMG_PATH_LIST_FILE)
    y_preds = []
    print('Inferring Images...')
    for i in tqdm(range(num_fpaths)):
        im = cv2.imread(fpaths[i])
        label_id_pred = nn.infer_optimized(im)
        y_preds.append(label_id_pred)
    y_truths = labels
    acc = accuracy_score(y_truths, y_preds)
    prec, rec, fsc, supp = precision_recall_fscore_support(y_truths, y_preds)
    performance_stats = {'accuracy': float(acc),
                         'precisions': list(map(float, list(prec))),
                         'recalls': list(map(float, list(rec))),
                         'f-scores': list(map(float, list(fsc)))}
    print(performance_stats)


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def classify_im_dir(nn, imgs_dir):
    print('Classifying images in', imgs_dir)
    all_im_paths = glob(imgs_dir + '/*')
    num_fpaths = len(all_im_paths)
    out_dir = imgs_dir + '_classified'
    class_dirs = [out_dir + os.sep + CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]
    for dir in class_dirs:
        force_makedir(dir)
    print('Beginning inference..')
    for i in tqdm(range(num_fpaths)):
        fpath = all_im_paths[i]
        im = cv2.imread(fpath)
        pred_label = CLASS_LABELS[nn.infer_optimized(im)]
        shutil.copy(fpath, out_dir + os.sep + pred_label)


if __name__ == '__main__':
    nn = RoomNet(num_classes=len(CLASS_LABELS), im_side=IMG_SIDE, compute_bn_mean_var=False,
                 optimized_inference=True)
    nn.load(INPUT_MODEL_PATH)

    # stats = groundtruth_validation(nn)
    classify_im_dir(nn, './test_images/set0/images')
