import time

import cv2

from network import RoomNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CLASS_LABELS = ['Backyard', 'Bathroom', 'Bedroom', 'Frontyard', 'Kitchen', 'LivingRoom']

INPUT_MODEL_PATH = './all_trained_models/trained_models/roomnet--0.8807870370370371--193700'
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
    from tqdm import tqdm
    fpaths, labels, num_fpaths = read_fpaths(INPUT_IMG_PATH_LIST_FILE)
    # y_truths = []
    y_preds = []
    print('Inferring Images...')
    for i in tqdm(range(num_fpaths)):
        # print('Inferring image at', fpaths[i])
        im = cv2.imread(fpaths[i])
        # label_id_truth = labels[i]
        # st = time.time()
        label_id_pred = nn.infer_optimized(im)
        # et = time.time()
        # delta = et - st
        # print('Inference time =', delta)
        # print('Predicted Label =', CLASS_LABELS[label_id_pred], '\n')
        # y_truths.append(label_id_truth)
        y_preds.append(label_id_pred)
    y_truths = labels
    acc = accuracy_score(y_truths, y_preds)
    prec, rec, fsc, supp = precision_recall_fscore_support(y_truths, y_preds)
    performance_stats = {'accuracy': float(acc),
                         'precisions': list(map(float, list(prec))),
                         'recalls': list(map(float, list(rec))),
                         'f-scores': list(map(float, list(fsc)))}
    print(performance_stats)




if __name__ == '__main__':
    nn = RoomNet(num_classes=6, im_side=IMG_SIDE, compute_bn_mean_var=False, optimized_inference=True)
    nn.load(INPUT_MODEL_PATH)

    # stats = groundtruth_validation(nn)


    k = 0