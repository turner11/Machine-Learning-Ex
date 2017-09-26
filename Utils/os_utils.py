import os

import cPickle


class File(object):
    """"""

    def __init__(self):
        """"""
        super(File, self).__init__()


    @staticmethod
    def get_text(path):
        with open(path,'r') as f:
            return f.read()

    @staticmethod
    def get_pickle(path):
        with open(path, 'r') as f:
            return cPickle.load(f)

    @staticmethod
    def dump_pickle(path,obj):
        with open(path, 'w') as f:
            cPickle.dump(obj, f)

