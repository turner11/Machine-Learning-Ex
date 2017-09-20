import os


class File(object):
    """"""

    def __init__(self):
        """"""
        super(File, self).__init__()


    @staticmethod
    def get_text(path):
        with open(path,'r') as f:
            return f.read()
