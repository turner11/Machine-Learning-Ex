import os
import numpy as np
from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader


class CreditDataLoader(AbstractDataLoader):
    """"""
    CSV_NAME = "creditcard.csv"

    def __init__(self):
        """"""
        my_folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(my_folder, self.CSV_NAME)
        super(self.__class__, self).__init__(path, 0, float("inf"), self.filter_data)
        pass

    def filter_data(self, m, ys):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        pos_loc = ys == 1
        neg_loc = ~pos_loc
        pos_count = np.count_nonzero(pos_loc)
        neg_indices = np.where(neg_loc == True)[0]
        pos_indices = np.where(pos_loc == True)[0]
        idx = np.random.choice(neg_indices, size=pos_count)
        return np.concatenate((idx,pos_indices))

