from collections import namedtuple

from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SlicedData = namedtuple('SlicedData', 'training_set training_y test_set test_y')
ModelScore = namedtuple("ModelScore", "precision recall accuracy f_measure")


class AbstractClassifier(object):
    """"""

    @property
    def feature_count(self):
        # column count
        return self.normalized_data.shape[1]

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
        self.__normalized_data = self.normalize_data(self.data)
        logger.info("Got {0} features for {1} samples".format(self.feature_count, self.samples_count))

    @property
    def normalized_data(self):
        return self.__normalized_data

    def __init__(self, gradient_step_size=0.01):
        """"""
        super(AbstractClassifier, self).__init__()
        self.__data = None
        self.__normalized_data = None

        self.val_0_str = None
        self.val_1_str = None
        self.ys = None
        self._model = None

    def _predict(self, normed):
        raise NotImplementedError("prediction must be implemented by concrete classifier")

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
            ys = [0 if x == self.val_0_str else 1 for x in ys_raw]
            self.ys = np.matrix(ys).transpose()

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

    def normalize_data(self, data):
        input_avgs = data.mean(axis=0)  # input - by column
        input_std = data.std(axis=0)
        normed = (data - input_avgs) / input_std
        # Adding 1 as the first feature for all
        with_ones = np.zeros((normed.shape[0], normed.shape[1] + 1))  # cerating the new matrix
        with_ones[:, 1:] = normed  # populating all but first column
        with_ones[:, 0:1] = 1  # adding 1's in first column
        return np.matrix(with_ones)

    def classify(self, data, is_data_normalized=False):
        normed = data if is_data_normalized else self.normalize_data(data)
        prediction = self._predict(normed)
        return prediction

    def slice_data(self, training_set_size_percentage=0.6, trainingset_size=None):
        if trainingset_size is None:
            if training_set_size_percentage is None or training_set_size_percentage <= 0 or training_set_size_percentage >= 1:
                raise Exception("percentage must be within the (0,1) range")

        trainingset_size = int(self.samples_count * training_set_size_percentage)

        training_set = self.normalized_data[:trainingset_size, :]
        train_y = self.ys[:trainingset_size]

        test_set = self.normalized_data[trainingset_size + 1:, :]
        test_y = self.ys[trainingset_size + 1:, :]

        return SlicedData(training_set, train_y, test_set, test_y)

    def train(self, training_set_size_percentage=0.6, trainingset_size=None):
        sliced_data = self.slice_data(training_set_size_percentage, trainingset_size)

        model = self._train(sliced_data.training_set, sliced_data.training_y)
        self._model = model

        model_score = self.get_model_score(sliced_data.test_set, sliced_data.test_y)
        self.log_score(model_score, prefix="Score for test set")

        model_score = self.get_model_score(sliced_data.training_set, sliced_data.training_y)
        self.log_score(model_score, prefix="Score for training set:")
        return self._model

    def get_model_score(self, test_set, test_y):
        prediction = self.classify(test_set, is_data_normalized=True)

        pos = test_y == 1
        neg = test_y == 0

        # True class A (TA) - correctly classified into class A
        tp = np.count_nonzero(prediction[pos] == 1)
        # False class A (FA) - incorrectly classified into class A
        fn = np.count_nonzero(prediction[pos] == 0)
        # True class B (TB) - correctly classified into class B
        tn = np.count_nonzero(prediction[neg] == 0)
        # False class B (FB) - incorrectly classified into class B
        fp = np.count_nonzero(prediction[neg] == 1)

        precision = float(tp) / (tp + fn)
        recall = float(tp) / (tp + fp)

        # You might also need accuracy and F-measure:
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        f_measure = 2 * (float(precision * recall) / (precision + recall))

        return ModelScore(precision=precision, recall=recall, accuracy=accuracy, f_measure=f_measure)

    def _train(self, t_samples, t_y):
        raise NotImplementedError("training must be implemented by concrete classifier")

    def log_score(self, model_score, prefix=""):
        prefix = (prefix or "Model score") + " ({0})".format(self)
        logger.info(prefix + ":\nprecision: {0}\nrecall: {1}\naccuracy: {2}\nf_measure: {3}"
                    .format(model_score.precision, model_score.recall, model_score.accuracy, model_score.f_measure))

    def __str__(self):
        return str(self.__class__.__name__)
