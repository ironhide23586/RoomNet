'''
Author: Souham Biswas
Email: souham.biswas@outlook.com
GitHub: https://github.com/ironhide23586
LinkedIn: https://www.linkedin.com/in/souham

I'm not responsible if your machine catches fire.
'''


from glob import glob
import os
import shutil

import cv2
from tqdm import tqdm
import xlwt

from network import RoomNet


CLASS_LABELS = ['Chimney', 'Under-cabinet']

INPUT_MODEL_PATH = './final_model/roomnet'
INPUT_IMAGES_DIR = './test_images/set3/images'
IMG_SIDE = 224

# INPUT_IMG_PATH_LIST_FILE = 'val_list.txt'


def read_fpaths(list_fpath):
    with open(list_fpath, 'r') as f:
        data = f.readlines()
    fpath_components = [fpath_set.strip().split(' ') for fpath_set in data]
    im_paths = [' '.join(fpath_component[:-1]) for fpath_component in fpath_components]
    class_id = [int(fpath_component[-1]) for fpath_component in fpath_components]
    n = len(class_id)
    return im_paths, class_id, n


def groundtruth_validation(nn):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


def classify_im_dir(nn, imgs_dir, overlay=True):
    print('Classifying images in', imgs_dir)
    all_im_paths = glob(imgs_dir + '/*')
    num_fpaths = len(all_im_paths)
    out_dir = imgs_dir + '_classified'
    xl_fpath = out_dir + '_results.xls'
    class_dirs = [out_dir + os.sep + CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]
    for dir in class_dirs:
        force_makedir(dir)
    print('Beginning inference..')
    excel_file = xlwt.Workbook()
    sheet = excel_file.add_sheet('classification_results')
    sheet.write(0, 0, 'IMAGE_NAME')
    sheet.write(0, 1, 'PREDICTED_LABEL')
    for i in tqdm(range(num_fpaths)):
        fpath = all_im_paths[i]
        im = cv2.imread(fpath)
        infer_outs = nn.infer_optimized(im)
        pred_label = CLASS_LABELS[infer_outs[0][0]]
        pred_conf = infer_outs[1][0][infer_outs[0][0]]
        out_fpath_dir = out_dir + os.sep + pred_label
        print(fpath, '--->', pred_label, pred_conf)
        if overlay:
            h, w, _ = im.shape
            cv2.putText(im, "Predicted Class: " + pred_label, (int(.5 * w), int(.90 * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, (h / 720.) * .85, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(im, "Confidence: " + str(round(pred_conf * 100, 2)) + " %", (int(.5 * w), int(.95 * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, (h / 720.) * .85, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(out_fpath_dir + os.sep + fpath.split(os.sep)[-1], im)
        else:
            shutil.copy(fpath, out_fpath_dir)
        sheet.write(i + 1, 0, fpath.split(os.sep)[-1])
        sheet.write(i + 1, 1, pred_label)
        sheet.write(i + 1, 2, str(pred_conf))
    excel_file.save(xl_fpath)
    return xl_fpath


if __name__ == '__main__':
    nn = RoomNet(num_classes=len(CLASS_LABELS), im_side=IMG_SIDE, compute_bn_mean_var=False,
                 optimized_inference=True)
    nn.load(INPUT_MODEL_PATH)

    # stats = groundtruth_validation(nn)
    xl_out_path = classify_im_dir(nn, INPUT_IMAGES_DIR)
    k = 0
