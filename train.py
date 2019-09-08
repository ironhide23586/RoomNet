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
import time

import cv2

from generator import DataFeeder
from network import RoomNet
from preprocess_google_open_dataset import GoogleOpenBboxPreprocessor


DATA_DIR = './data/Google-Open-Images'

IMG_SIDE = 224
TRAIN_BATCH_SIZE = 8
TRAIN_STEPS = 100000
SAVE_FREQ = 150
LEARN_RATE = 7e-5
DROPOUT_ENABLED = False
DROPOUT_RATE = .35
L2_REGULARIZATION_COEFF = 5e-2
UPDATE_BATCHNORM_MOVING_VARS = False
COMPUTE_BN_MEAN_VAR = False


def wait():
    while True:
        time.sleep(300)


if __name__ == '__main__':
    google_open_bbox_preprocessor = GoogleOpenBboxPreprocessor(DATA_DIR, ['Bathtub', 'Shower'],
                                                               images_dirname='Bathtub_Shower', init=True,
                                                               shuffle=True, train_frac=.9)
    train_data_reader = DataFeeder(google_open_bbox_preprocessor, batch_size=TRAIN_BATCH_SIZE, batches_per_queue=10,
                                   shuffle=True, im_side=IMG_SIDE, random_crop=True, preprocess=True, split='train')
    val_data_reader = DataFeeder(google_open_bbox_preprocessor, batch_size=TRAIN_BATCH_SIZE, batches_per_queue=30,
                                 shuffle=False, im_side=IMG_SIDE, random_crop=False, preprocess=False, split='val')

    nn = RoomNet(num_classes=2, im_side=IMG_SIDE, num_steps=TRAIN_STEPS, learn_rate=LEARN_RATE,
                 dropout_rate=DROPOUT_RATE, l2_regularizer_coeff=L2_REGULARIZATION_COEFF,
                 dropout_enabled=DROPOUT_ENABLED, update_batchnorm_means_vars=UPDATE_BATCHNORM_MOVING_VARS,
                 compute_bn_mean_var=COMPUTE_BN_MEAN_VAR, load_training_vars=False,
                 train_batch_size=TRAIN_BATCH_SIZE)
    nn.init()
    nn.load('final_model/roomnet')
    # nn.load('all_trained_models/roomnet--21900')
    # nn.load('all_trained_models/trained_models_object_detection_3/roomnet--151500')

    for train_iter in range(nn.start_step, nn.start_step + TRAIN_STEPS):
        if train_iter % SAVE_FREQ == 0:  # and train_iter > nn.start_step:
            x_val, y_val = val_data_reader.dequeue()
            y_vals = list(y_val)
            y_preds = []
            print('Validating model at step', nn.step)
            while not val_data_reader.train_state['previous_epoch_done']:
                y_pred = nn.infer(x_val)
                y_preds += y_pred
                x_val, y_val = val_data_reader.dequeue()
                y_vals += list(y_val)
            print('Inference Complete!')
            # y_vals = y_vals[:len(y_preds)]
            # acc = accuracy_score(y_vals, y_preds)
            # prec, rec, fsc, supp = precision_recall_fscore_support(y_vals, y_preds)
            # nn.save(suffix=str(acc))
            # train_stats = {'step': int(nn.step), 'accuracy': float(acc),
            #                'precisions': list(map(float, list(prec))),
            #                'recalls': list(map(float, list(rec))),
            #                'f-scores': list(map(float, list(fsc)))}
            for m in range(len(y_preds)):
                dirname = 'rough' + os.sep + 'train-iter-' + str(train_iter)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                yv = y_preds[m]
                im_viz, out_locs_tlxy_brxy, out_locs_tlxy_brxy_normalized, \
                out_class_ids, out_class_names, out_scores = yv
                cv2.imwrite(dirname + os.sep + str(m) + '.png', im_viz)
            print('Saving')
            nn.save()
        x, y = train_data_reader.dequeue()
        loss, train_step, learn_rate = nn.train_step(x, y)
        print('Step', train_step, 'loss =', loss, 'learn_rate =', learn_rate)
