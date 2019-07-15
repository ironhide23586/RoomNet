'''
Author: Souham Biswas
Email: souham.biswas@outlook.com
GitHub: https://github.com/ironhide23586
LinkedIn: https://www.linkedin.com/in/souham

I'm not responsible if your machine catches fire.
'''


from glob import glob

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = 'all_trained_models/trained_models_custom0'

if __name__ == '__main__':
    model_fpaths = glob(INPUT_DIR + '/*roomnet*.meta')
    overall_acc_path = INPUT_DIR + '_accuracy_plot.png'
    steps = np.array([int(fp.split('--')[-1].replace('.meta', '')) for fp in model_fpaths])
    accs = np.array([float(fp.split('--')[-2]) for fp in model_fpaths])

    idx = np.argsort(steps)
    steps = steps[idx]
    accs = accs[idx]

    plt.clf()
    plt.plot(steps, accs, '-', color='red', label='Classsification Accuracy')
    title_str = 'Model with max overall score is at step ' + str(
                steps[accs.argmax()]) + '\nwith value ' + str(accs.max())
    plt.title(title_str)
    plt.legend(loc='best')
    plt.xlabel('Train Step')
    plt.ylabel('Validation Overall Accuracy over 1839 images')
    plt.savefig(overall_acc_path, bbox_inches='tight', dpi=200)