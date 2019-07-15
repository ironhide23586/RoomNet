'''
Author: Souham Biswas
Email: souham.biswas@outlook.com
GitHub: https://github.com/ironhide23586
LinkedIn: https://www.linkedin.com/in/souham
I'm not responsible if your machine catches fire.
'''


import json
import os

import matplotlib.pyplot as plt
import numpy as np


CLASS_LABELS = ['Backyard', 'Bathroom', 'Bedroom', 'Frontyard', 'Kitchen', 'LivingRoom']
OUT_DIR = 'performance_plots'
all_colors = np.array([(244, 35, 231), (69, 69, 69), (219, 219, 0),
                       (0, 0, 142), (0, 79, 100), (119, 10, 32)]).astype(np.float32) / 255.
train_stats_json = 'all_train_stats.json'


if __name__ == '__main__':
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    train_stats = json.load(open(train_stats_json, 'r'))
    overall_acc_path = OUT_DIR + os.sep + 'accuracy_plot.png'
    overall_fsc_path = OUT_DIR + os.sep + 'fscore_plot.png'
    overall_rec_path = OUT_DIR + os.sep + 'recall_plot.png'
    overall_prec_path = OUT_DIR + os.sep + 'precision_plot.png'
    steps = []
    accs = []
    fscs = []
    recs = []
    precs = []
    for stat in train_stats:
        steps.append(stat['step'])
        accs.append(stat['accuracy'])
        fscs.append(stat['f-scores'])
        recs.append(stat['recalls'])
        precs.append(stat['precisions'])
    steps = np.array(steps)
    accs = np.array(accs)
    fscs = np.array(fscs)
    recs = np.array(recs)
    precs = np.array(precs)
    idx = np.argsort(steps)
    steps = steps[idx]
    accs = accs[idx]
    fscs = fscs[idx]
    recs = recs[idx]
    precs = precs[idx]

    plt.clf()
    plt.plot(steps, accs, '-', color='red', label='Classsification Accuracy')
    title_str = 'Model with max overall score is at step ' + str(
                steps[accs.argmax()]) + '\nwith value ' + str(accs.max())
    plt.title(title_str)
    plt.legend(loc='best')
    plt.xlabel('Train Step')
    plt.ylabel('Validation Overall Accuracy over 1839 images')
    plt.savefig(overall_acc_path, bbox_inches='tight', dpi=200)

    plt.clf()
    plt.figure(figsize=(20, 20))
    title_str = 'Best Overall class performers -\n'
    for i in range(len(CLASS_LABELS)):
        plt.plot(steps, fscs[:, i], '-', color=all_colors[i], label=CLASS_LABELS[i])
        best_model_idx = fscs[:, i].argmax()
        best_model_score = fscs[best_model_idx, i]
        best_model_step = steps[best_model_idx]
        title_str += CLASS_LABELS[i] + '---> model at step ' + str(best_model_step) + ' ' + ' with value ' + str(
            best_model_score) + '\n'
    plt.title(title_str)
    plt.legend(loc='best')
    plt.xlabel('Train Step')
    plt.ylabel('Validation Class Overall F-Scores over 1839 images')
    plt.savefig(overall_fsc_path, bbox_inches='tight', dpi=200)

    plt.clf()
    plt.figure(figsize=(20, 20))
    title_str = 'Best Overall class performers -\n'
    for i in range(len(CLASS_LABELS)):
        plt.plot(steps, recs[:, i], '-', color=all_colors[i], label=CLASS_LABELS[i])
        best_model_idx = recs[:, i].argmax()
        best_model_score = recs[best_model_idx, i]
        best_model_step = steps[best_model_idx]
        title_str += CLASS_LABELS[i] + '---> model at step ' + str(best_model_step) + ' ' + ' with value ' + str(
            best_model_score) + '\n'
    plt.title(title_str)
    plt.legend(loc='best')
    plt.xlabel('Train Step')
    plt.ylabel('Validation Class Overall Recalls over 1839 images')
    plt.savefig(overall_rec_path, bbox_inches='tight', dpi=200)

    plt.clf()
    plt.figure(figsize=(20, 20))
    title_str = 'Best Overall class performers -\n'
    for i in range(len(CLASS_LABELS)):
        plt.plot(steps, precs[:, i], '-', color=all_colors[i], label=CLASS_LABELS[i])
        best_model_idx = precs[:, i].argmax()
        best_model_score = precs[best_model_idx, i]
        best_model_step = steps[best_model_idx]
        title_str += CLASS_LABELS[i] + '---> model at step ' + str(best_model_step) + ' ' + ' with value ' + str(
            best_model_score) + '\n'
    plt.title(title_str)
    plt.legend(loc='best')
    plt.xlabel('Train Step')
    plt.ylabel('Validation Class Overall Precisions over 1839 images')
    plt.savefig(overall_prec_path, bbox_inches='tight', dpi=200)
