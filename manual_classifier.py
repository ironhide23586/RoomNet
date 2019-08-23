import os
from glob import glob
import shutil

import cv2


INPUT_DIR = './data/REI-Dataset-reduced/bathroom'


class ImageLabeler:

    def __init__(self, in_dir):
        self.in_dir = in_dir
        self.output_dir = self.in_dir + '-labelled'
        self.log_file_fpath = self.output_dir + os.sep + 'log.txt'
        self.label_file_path = self.output_dir + os.sep + 'labels.csv'
        self.img_paths = glob(self.in_dir + os.sep + '*')
        self.force_makedir(self.output_dir)
        self.num_images = len(self.img_paths)
        self.processed_image_names = []

    @staticmethod
    def show_and_label(im):
        cv2.imshow('image', im)
        key = cv2.waitKey()
        return key

    def force_makedir(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
            self.pl(':: Made directory at ' + dirpath)

    def pl(self, line):
        with open(self.log_file_fpath, 'a+') as f:
            f.write(line + '\n')
            print(line)

    def write_to_csv(self, img_name, label):
        img_str = img_name.split(os.sep)[-1]
        label_str = ','.join([str(lbl) for lbl in label])
        with open(self.label_file_path, 'a+') as f:
            line = img_str + ',' + label_str
            f.write(line + '\n')

    def extract_existing_labels(self):
        if not os.path.isfile(self.label_file_path):
            return []
        with open(self.label_file_path, 'r') as f:
            lines = f.readlines()
        processed_fnames = [l.split(',')[0] for l in lines]
        return processed_fnames

    def preprocess_label(self, label_raw):  # Modify this according to use
        return [str(label_raw)]

    def label2dirname(self, label):  # Modify this according to use
        return str(label[0])

    def run_labeller(self, resume=True, bin_files=True):
        if resume:
            self.pl('Resuming from previous labeling session...')
            self.processed_image_names = self.extract_existing_labels()
            self.pl('Found' + str(len(self.processed_image_names)) + ' labeled images...')
        self.pl('Starting Manual Labeller....')
        for i in range(self.num_images):
            img_path = self.img_paths[i]
            img_fname = img_path.split(os.sep)[-1]
            self.pl('Processing path ' + img_path)
            if img_fname not in self.processed_image_names:
                im = cv2.imread(img_path)
                img_label_raw = self.show_and_label(im)
                if img_label_raw != 27:
                    img_label = self.preprocess_label(img_label_raw)
                    img_label_binname = self.label2dirname(img_label)
                    if bin_files:
                        dst_dirpath = self.output_dir + os.sep + 'binned_files' + os.sep + img_label_binname
                        self.force_makedir(dst_dirpath)
                        shutil.copy(img_path, dst_dirpath)
                        self.pl('Copied ' + img_path + ' to ' + dst_dirpath + os.sep + img_fname)
                    self.write_to_csv(img_fname, img_label)
                    self.processed_image_names.append(img_fname)
                    self.pl(img_fname + '------------------------' + str(img_label))
                else:
                    self.pl('Process aborted, exiting...')
                    exit()
            else:
                self.pl('Already processed, skipping...')
            self.pl('-->' + str(100. * (i + 1) / self.num_images) + ' % complete')
        self.pl('All Labels done! :D')


if __name__ == '__main__':
    labeler = ImageLabeler(INPUT_DIR)
    labeler.run_labeller()
