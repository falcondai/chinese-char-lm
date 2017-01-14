import cPickle as pickle
import sys

class Vocabulary_Builder(object):
    """docstring for Vocabulary_Builder"""
    def __init__(self, vocabulary_dict = {}):
        super(Vocabulary_Builder, self).__init__()
        self.vocabulary_dict = vocabulary_dict
        self.save_vocabulary_path = './work/vocabulary_frequency_dictionary.pkl'


    def read_in(self, file_path):
        print "reading..."

        with open(file_path,'r') as f:
            doc = f.readline()
            for line in doc:
                chars = line.split()
                for char in chars:
                    self.vocabulary_dict.get(char, 1) + 1

        print "vocabulary building complete"

        print "saving..."
        with open(self.save_vocabulary_path, 'w') as f:
            pickle.dump(self.vocabulary_dict, f)

        print "done"

if __name__ == '__main__':
    train_set_path = './work/train.txt'

    vocab_builder = Vocabulary_Builder()
    
    vocab_builder.read_in(train_set_path)
    

