#!/usr/bin/env python
# -*- coding:utf-8-*-
import operator, sys

class Vocabulary_Builder(object):
    """docstring for Vocabulary_Builder"""
    def __init__(self, vocabulary_dict = {}):
        super(Vocabulary_Builder, self).__init__()
        self.vocabulary_dict = vocabulary_dict
        self.save_vocabulary_path = './work/dict.txt'
        self.start_tag = '<P>'
        self.end_tag = '</P>'

    def read_in(self, file_path):
        # 1. read through the file
        print "reading..."

        with open(file_path, 'rb') as f:
            doc = f.readlines()
            for line in doc:
                chars = line.split()
                for char in chars:
                    if char in self.vocabulary_dict:
                        self.vocabulary_dict[char] += 1
                    else:
                        self.vocabulary_dict[char] = 1

        print "vocabulary building complete"

        # 2. sorting according to frequency
        # put special symbols in the front
        self.vocabulary_dict[self.start_tag] = sys.maxint
        self.vocabulary_dict[self.end_tag] = sys.maxint
        self.sorted_dict = sorted(self.vocabulary_dict.items(), key=operator.itemgetter(1), reverse=True)

        # 3. saving dictionary to a txt file
    def save(self):
        print "saving..."
        with open(self.save_vocabulary_path, 'wb') as f:
            for char, freq in self.sorted_dict:
                f.write(char)
                f.write('\n')
        print "done"

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--corpus-path', required=True)
    # parser.add_argument('-d', '--dictionary-path')

    train_set_path = sys.argv[1]

    vocab_builder = Vocabulary_Builder()

    vocab_builder.read_in(train_set_path)
