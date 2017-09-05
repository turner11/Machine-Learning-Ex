from collections import namedtuple
from Classifiers import rootLogger as logger
import numpy as np

from Classifiers.DataLoaders.ClassifyingData import ClassifyingData
from Utils.Event import EventHook

draw_plots = False
draw_data=True
TRAINING_SET_SIZE_PERCENTAGE = 0.6
SlicedData = namedtuple('SlicedData', 'training_set training_y test_set test_y')


class ModelScore(object):
    def __init__(self, precision=None, recall=None, accuracy=None, f_measure=None):
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.f_measure = f_measure

    def __str__(self):
        return "precision: {0:.4f}\t" \
               "recall: {1:.4f}\t" \
               "accuracy: {2:.4f}\t" \
               "f_measure: {3:.4f}\t".format(self.precision,
                                             self.recall,
                                             self.accuracy,
                                             self.f_measure)

    def __eq__(self, other):
        eq = self.precision == other.precision and \
             self.recall == other.recall and \
             self.accuracy == other.accuracy and \
             self.f_measure == other.f_measure

        return eq

    def has_nans(self):
        stats = [self.precision, self.recall, self.accuracy, self.f_measure]
        return any([np.math.isnan(s) for s in stats])


class AbstractClassifier(object):
    """"""

    @property
    def training_set_size_percentage(self):
        return TRAINING_SET_SIZE_PERCENTAGE
    @property
    def feature_count(self):
        # column count
        return self.normalized_data.shape[1]

    @property
    def samples_count(self):
        # rows count
        if len(self.data) == 0:
            return 0

        return len(self.data[:, 0])

    @property
    def data(self):
        return self.input_data.x_mat

    @property
    def normalized_data(self):
        return self.__normalized_data

    @property
    def val_0_str(self):
        return self.input_data.val_0_str

    @property
    def val_1_str(self):
        return self.input_data.val_1_str

    @property
    def ys(self):
        return self.input_data.ys

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, value):
        self.__input_data = value
        self.__event_data_loaded_internal()

    def __init__(self, draw_plots=False):
        """"""
        self.draw_plots = draw_plots
        self.__event_data_loaded_internal = EventHook()
        self.__event_data_loaded_internal += self.__data_loaded_handler
        self.event_data_loaded = EventHook()
        super(AbstractClassifier, self).__init__()


        self.__input_data = ClassifyingData("Null Object 0", "Null Object 1", [], np.array([]))

        self.__data = None
        self.__normalized_data = None

        self.__features_avgs = None
        self.__features_std = None

        self._model = None
        self.score = None


    def _predict(self, normed_data):
        raise NotImplementedError("prediction must be implemented by concrete classifier")

    def set_data(self, input_data):
        # type: (ClassifyingData) -> None
        self.input_data = input_data

    def normalize_data(self, data):
        # from sklearn.preprocessing import StandardScaler
        # normed  = StandardScaler().fit_transform(data)
        normed = (data - self.__features_avgs) / self.__features_std
        # Adding 1 as the first feature for all
        return np.matrix(normed)

    def classify(self, data, is_data_normalized=False):
        normed_data = data if is_data_normalized else self.normalize_data(data)
        prediction = self._predict(normed_data)
        return np.array(prediction).reshape(-1)


    def slice_data(self, training_set_size_percentage=0.6, normalized=True):

        trainingset_size = int(self.samples_count * training_set_size_percentage)
        test_set_size = self.samples_count-trainingset_size

        data_to_use = self.normalized_data if normalized else self.data
        all_idxs =set(range(self.samples_count))
        training_idxs = sorted(np.random.choice(list(all_idxs), size=trainingset_size))
        test_idsx =  list(all_idxs - set(training_idxs ))
        # type: (np.ndarray, np.ndarray) -> np.ndarray

        training_set = data_to_use[training_idxs, :]
        train_y = self.ys[training_idxs]

        test_set = data_to_use[test_idsx, :]
        test_y = self.ys[test_idsx]

        return SlicedData(training_set, train_y, test_set, test_y)

    def train(self, training_set_size_percentage=0.7):
        sliced_data = self.slice_data(training_set_size_percentage)

        model = self._train(sliced_data.training_set, sliced_data.training_y)
        self._model = model

        model_score = self.get_model_score(sliced_data.test_set, sliced_data.test_y)
        self.log_score(model_score, prefix="Score for test set")

        self.score = model_score

        train_score = self.get_model_score(sliced_data.training_set, sliced_data.training_y)
        self.log_score(train_score, prefix="Score for training set:")

        return self._model

    def get_model_score(self, test_set, test_y, prediction=None):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> ModelScore
        prediction = prediction if prediction is not None else  self.classify(test_set, is_data_normalized=True)

        pos = np.array(test_y == 1)
        neg = np.array(test_y == 0)

        # True class A (TA) - correctly classified into class A
        tp = np.count_nonzero(prediction[pos] == 1)
        # False class A (FA) - incorrectly classified into class A
        fn = np.count_nonzero(prediction[pos] == 0)
        # True class B (TB) - correctly classified into class B
        tn = np.count_nonzero(prediction[neg] == 0)
        # False class B (FB) - incorrectly classified into class B
        fp = np.count_nonzero(prediction[neg] == 1)

        precision = float(tp) / (tp + fn) if tp + fn != 0 else np.nan
        recall = float(tp) / (tp + fp) if tp + fp != 0 else np.nan

        # You might also need accuracy and F-measure:
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        f_measure = 2 * (float(precision * recall) / (precision + recall)) if precision + recall != 0 else np.nan

        return ModelScore(precision=precision, recall=recall, accuracy=accuracy, f_measure=f_measure)

    def _train(self, t_samples, t_y):
        raise NotImplementedError("training must be implemented by concrete classifier")

    def log_score(self, model_score, prefix=""):

        line_sep = "---------------------------------------------------"
        prefix = (prefix or "Model score") + " ({0}) ".format(self)
        logger.info(line_sep)
        logger.info(prefix + ":\n")
        logger.info(str(model_score))
        logger.info(line_sep)

    def __data_loaded_handler(self):
        self.__features_avgs = self.data.mean(axis=0)  # input - by column
        self.__features_std = self.data.std(axis=0)

        # from sklearn.model_selection import train_test_split
        # train_test_split(X, y, test_size=.4, random_state=42)



        self.__normalized_data = self.normalize_data(self.data)
        logger.info("Got {0} features for {1} samples".format(self.feature_count, self.samples_count))
        self.event_data_loaded(self, self.normalized_data, self.ys)

        if draw_data:
            from DataVisualization.Visualyzer import Visualyzer
            Visualyzer.PlotPCA(self.normalized_data, self.ys, dim=3)

    def __str__(self):
        return str(self.__class__.__name__)
