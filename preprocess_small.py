#!/usr/bin/python
# -*- coding: utf-8 -*-

# preproces Chinese Gigaword 5th ed into train/val/test splits
# output format:
# one paragraph per line with space separated tokens

import os, gzip, argparse, glob
import numpy as np

def char_split(file_handle):

    for line in file_handle:
        words_list = list(line.strip().decode('utf-8'))
        words_seq = [word for word in words_list if word != u' ']
        new_line = []
        seg_index = []
        for word in words_seq:

            for char in word:
                new_line.append(char.encode('utf-8'))
            seg_index.append(len(new_line))
        yield new_line, seg_index
        continue
    # return raw_lines, seg_indices

if __name__ == '__main__':
    # create work directory
    if not os.path.exists('segmentation'):
        os.mkdir('segmentation')

    train_paths = glob.glob('corpora/icwb2-data/training/*.utf8')
    test_paths = glob.glob('corpora/icwb2-data/testing/*.utf8')
    np.random.seed(123)
    for path in test_paths:
        file_name = path.split('/')[-1].split('.')[0]
        print file_name
        with open(path, 'rb') as fread_handle: 
            fraw_handle = open('segmentation/'+file_name+'_raw', 'wb') 
            fseg_handle = open('segmentation/'+file_name+'_seg', 'wb')
            for raw_line, seg_index in char_split(fread_handle):
                fraw_handle.write(' '.join(raw_line))
                fraw_handle.write('\n')
                fseg_handle.write(' '.join([str(index) for index in seg_index]))
                fseg_handle.write('\n')

            fraw_handle.close()
            fseg_handle.close()

    for path in train_paths:
        file_name = path.split('/')[-1].split('.')[0].split('_')[0]
        print file_name
        file_handle_dict = {'train': (open('segmentation/'+file_name+'_train_raw', 'wb'), 
                                    open('segmentation/'+file_name+'_train_seg', 'wb')),
                            'val': (open('segmentation/'+file_name+'_val_raw', 'wb'), 
                                    open('segmentation/'+file_name+'_val_seg', 'wb'))}

        with open(path, 'rb') as fread_handle:


            for raw_line, seg_index in char_split(fread_handle):
                fhandle_tuple = file_handle_dict[np.random.choice(['train', 'val'], p=[0.9, 0.1])]
                fhandle_tuple[0].write(' '.join(raw_line))
                fhandle_tuple[0].write('\n')
                fhandle_tuple[1].write(' '.join([str(index) for index in seg_index]))
                fhandle_tuple[1].write('\n')




