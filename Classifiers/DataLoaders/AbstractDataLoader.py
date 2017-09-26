from Classifiers import rootLogger as logger
import pandas as pd
import numpy as np

from Classifiers.DataLoaders.ClassifyingData import ClassifyingData


class AbstractDataLoader(object):
    """"""

    def __init__(self, data_path, header_row ,classification_column, filter_data_func=None):
        # type: (str, int, int, function) -> None
        """"""
        super(AbstractDataLoader, self).__init__()
        self.data_path = data_path
        self.header_row =header_row
        self.classification_column = classification_column
        self.filter_data_func = filter_data_func or (lambda m,ys: None)

    def load(self):
        # type: (str, int, function) -> ClassifyingData
        try:
            logger.info("Loading data from: {0}".format(self.data_path))
            try:
                loaded_data = pd.read_excel(self.data_path, header=self.header_row)
            except:
                loaded_data = pd.read_csv(self.data_path, header=self.header_row )

            m = loaded_data .values

            # get the tags
            class_clm = m.shape[1]-1 if np.inf ==  self.classification_column else self.classification_column
            ys_raw = m[:, class_clm]
            ys = [0 if x ==  ys_raw[0] else 1 for x in ys_raw]
            ys = np.array(ys).transpose()

            # make data hold only numerical values
            m = np.delete(m, [class_clm], 1)

            # data should be numerical
            m = m.astype('float32')
            idxs = self.filter_data_func(m, ys)
            if idxs is not None:
                m = m[idxs]
                ys = ys[idxs]

            return ClassifyingData(ys, m)

        except Exception as ex:
            logger.error("Failed to read data:\t{0}".format(str(ex)))
            raise



