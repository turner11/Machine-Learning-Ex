from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np


class LogisticClassifier(object):
    """"""

    @property
    def feature_count(self):
        # column count
        return len(self.data[0, :])

    @property
    def samples_count(self):
        # rows count
        return len(self.data[:, 0])

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data
        logger.info("Got {0} features for {1} samples".format(self.feature_count, self.samples_count))
        self.set_normalized_data()

    @property
    def normalized_data(self):
        return self.__normalized_data

    def __init__(self):
        """"""
        super(self.__class__, self).__init__()
        self.__data = None
        self.__normalized_data = None

        self.val_0_str = None
        self.val_1_str = None
        self.ys = None
        self.__model = None

    def load_data_from_csv(self, data_path, classification_column=0):
        try:
            # self.data = np.genfromtxt(data_path, delimiter=',')
            logger.warn("Assuming no headers in csv")
            csv = pd.read_csv(data_path, header=None)
            m = csv.values

            # get the tags
            ys_raw = m[:, classification_column]
            self.val_0_str = ys_raw[0]
            self.val_1_str = next(x for x in ys_raw if x != self.val_0_str)
            self.ys = [0 if x == self.val_0_str else 1 for x in ys_raw]

            # make data hold only numerical values
            m = np.delete(m, [classification_column], 1)

            # data should be numerical
            m = m.astype('float32')
            self.data = m

        except Exception as ex:
            logger.error("Failed to read data:\t{0}".format(str(ex)))
            raise

            # def __str__(self, ):
            #     pass

    def set_normalized_data(self):
        df = self.data
        input_avgs = df.mean(axis=0)  # input - by column
        input_std = df.std(axis=0)
        self.__normalized_data = (df - input_avgs) / input_std

    def train(self, training_set_size_percentage=0.6, trainingset_size=None):

        if trainingset_size is None:
            if training_set_size_percentage is None or training_set_size_percentage <= 0 or training_set_size_percentage >= 1:
                raise Exception("percentage must be within the (0,1) range")

        trainingset_size = int(self.samples_count * training_set_size_percentage)
        t_samples = self.normalized_data[:trainingset_size, :]
        t_y = self.ys[:trainingset_size]
        model = self._train(t_samples,t_y)
        self.__model = model
        return self.__model

    def get_initial_model(self, size):
        logger.warn("using an initial all 0 model")
        return [0] * size

    def _train(self, t_samples, t_y):
        sample_count = len(t_y)
        assert t_samples.shape[0] ==sample_count
        logger.info("Starting to train on {0} features and {1} samples.".format(t_samples.shape[1], sample_count))
        model = self.get_initial_model(len(t_y))
        model
