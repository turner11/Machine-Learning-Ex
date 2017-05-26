from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np

class DataLoader(object):
    """"""

    def __init__(self, args):
        """"""
        super(self.__class__, self).__init__()

    @staticmethod
    def from_csv(data_path, classification_column=0, filter_data_func=None):
        # type: (str, int, function) -> ClassifyingData
        try:
            # self.data = np.genfromtxt(data_path, delimiter=',')
            logger.warn("Assuming no headers in csv")
            csv = pd.read_csv(data_path, header=None)
            m = csv.values

            # get the tags
            logger.warn("Assuming tags headers in first column")
            ys_raw = m[:, classification_column]
            val_0_str = ys_raw[0]
            val_1_str = next(x for x in ys_raw if x != val_0_str)
            ys = [0 if x == val_0_str else 1 for x in ys_raw]
            ys = np.array(ys).transpose()

            # make data hold only numerical values
            m = np.delete(m, [classification_column], 1)

            # data should be numerical
            m = m.astype('float32')
            filtered_m = m if filter_data_func is None else filter_data_func(m)
            return ClassifyingData(val_0_str, val_1_str, ys , filtered_m)

        except Exception as ex:
            logger.error("Failed to read data:\t{0}".format(str(ex)))
            raise

class ClassifyingData(object):
    """"""

    def __init__(self, val_0_str, val_1_str, ys , x_mat):
        """"""
        super(self.__class__, self).__init__()
        self.val_0_str = val_0_str
        self.val_1_str = val_1_str
        self.ys = ys
        self.x_mat = x_mat


    