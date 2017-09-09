from collections import namedtuple

import numpy as np

SlicedData = namedtuple('SlicedData', 'training_set training_y test_set test_y')
class ClassifyingData(object):
    """"""

    @property
    def feature_count(self):
        # column count
        return self.x_mat.shape[1]

    @property
    def samples_count(self):
        # rows count
        if self.x_mat is None or self.x_mat.size == 0:
            return 0

        return len(self.x_mat[:, 0])

    def __init__(self, ys, x_mat, val_0_str=None, val_1_str=None, origin_data=None):
        """"""
        super(self.__class__, self).__init__()
        self.ys = ys
        self.x_mat = np.matrix(x_mat)

        self.__is_normalized = False

        self.val_0_str = val_0_str or ys[0]
        self.val_1_str = val_1_str or next(x for x in ys if x != val_0_str)

        self.origin_data = origin_data

    def normalize(self):
        if self.__is_normalized:
            print('data is already normalized')
            return
        # from sklearn.model_selection import train_test_split
        # train_test_split(X, y, test_size=.4, random_state=42)
        from sklearn.preprocessing import StandardScaler
        self.origin_data = self.x_mat
        normed = StandardScaler().fit_transform(self.x_mat)
        self.x_mat = normed

        # self.__features_avgs = self.data.mean(axis=0)  # input - by column
        # self.__features_std = self.data.std(axis=0)
        # normed = (data - self.__features_avgs) / self.__features_std




    def slice_data(self, training_set_size_percentage=0.7):
        from sklearn.cross_validation import train_test_split

        test_size = 1-training_set_size_percentage
        X_train, X_test, y_train, y_test = \
            train_test_split(self.x_mat, self.ys, test_size=test_size, random_state=42)

        return SlicedData(X_train, y_train, X_test, y_test)

        # trainingset_size = int(self.samples_count * training_set_size_percentage)
        # test_set_size = self.samples_count-trainingset_size
        #
        # all_idxs =set(range(self.samples_count))
        # training_idxs = sorted(np.random.choice(list(all_idxs), size=trainingset_size))
        # test_idsx =  list(all_idxs - set(training_idxs ))
        # # type: (np.ndarray, np.ndarray) -> np.ndarray
        #
        # training_set = data_to_use[training_idxs, :]
        # train_y = self.ys[training_idxs]
        #
        # test_set = data_to_use[test_idsx, :]
        # test_y = self.ys[test_idsx]
        #
        # return SlicedData(training_set, train_y, test_set, test_y)



NULL_OBJECT = ClassifyingData([], np.array([]), "Null Object 0", "Null Object 1")