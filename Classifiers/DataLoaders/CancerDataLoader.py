import os

from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader


class CancerDataLoader(AbstractDataLoader):
    """"""
    CSV_NAME = "Cancer.csv"

    def __init__(self):
        """"""
        my_folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(my_folder, self.CSV_NAME)
        super(self.__class__, self).__init__(path,None,0)
        pass

