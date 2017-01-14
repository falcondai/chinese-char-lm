#!/usr/bin/python
# -*- coding: utf-8 -*-

# preproces Chinese Gigaword 5th ed into train/val/test splits
# output format:
# one paragraph per line with space separated tokens

import os, gzip, argparse, glob
import numpy as np

def paragraphs(fp):
    # use <P> </P> to identify the start and end of a sentence
    start_tag = '<P>'
    end_tag = '</P>'
    paragraph = u''
    should_append = False
    for line in fp:
        l = line.strip().decode('utf-8')
        if should_append:
            if l == end_tag:
                should_append = False
                yield paragraph
                paragraph = u''
            else:
                paragraph += u' '.join(l) + u' '
        elif line.startswith(start_tag):
            should_append = True

if __name__ == '__main__':
    # create work directory
    if not os.path.exists('work'):
        os.mkdir('work')
    split_p = [0.5, 0.1, 0.4]
    out_fs = [
        open('work/train.txt', 'wb'),
        open('work/val.txt', 'wb'),
        open('work/test.txt', 'wb'),
    ]
    gz_paths = glob.glob('corpora/cmn_gw_5/data/xin_cmn/*.gz')
    np.random.seed(123)
    for gz_path in gz_paths:
        with gzip.open(gz_path, 'rb') as gz_f:
            for paragraph in paragraphs(gz_f):
                f = np.random.choice(out_fs, p=split_p)
                f.write(paragraph.encode('utf-8'))
                f.write('\n')
