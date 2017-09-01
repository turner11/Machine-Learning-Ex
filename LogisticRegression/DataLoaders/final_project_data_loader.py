import os

from LogisticRegression.DataLoaders.AbstractDataLoader import AbstractDataLoader


class FinalProjectDataLoader(AbstractDataLoader):
    """"""
    CSV_NAME = "real_project_data.csv"

    def __init__(self):
        """"""
        my_folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(my_folder, self.CSV_NAME)
        super(self.__class__, self).__init__(path,None,20)
        pass

