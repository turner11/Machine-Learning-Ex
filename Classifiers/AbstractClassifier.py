from collections import namedtuple
from Classifiers import rootLogger as logger
import numpy as np

from Classifiers.DataLoaders.ClassifyingData import ClassifyingData, NULL_OBJECT
from Utils.Event import EventHook

draw_plots = False
draw_data = True
TRAINING_SET_SIZE_PERCENTAGE = 0.6


class ModelScore(object):
    def __init__(self, precision=None, recall=None, accuracy=None, f_measure=None):
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.f_measure = f_measure


    def __str__(self):
        try:
            str =  "precision: {0:.4f}\t" \
                   "recall: {1:.4f}\t" \
                   "accuracy: {2:.4f}\t" \
                   "f_measure: {3:.4f}\t".format(self.precision,
                                                 self.recall,
                                                 self.accuracy,
                                                 self.f_measure)
        except:
            # in case of nones...
            str = "precision: {0}\t" \
                  "recall: {1}\t" \
                  "accuracy: {2}\t" \
                  "f_measure: {3}\t".format(self.precision,
                                                self.recall,
                                                self.accuracy,
                                                self.f_measure)
        return str

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
    def feature_count(self):
        # type: () -> int
        # column count
        return self.data.shape[1]

    @property
    def samples_count(self):
        # type: () -> int
        return self.input_data.samples_count

    @property
    def data(self):
        # type: () -> np.matrix
        data = self.input_data.x_mat
        if data is not None \
                and self._data_mask is not None \
                and data.shape[1] > max(self._data_mask):
            data = data[:,self._data_mask]
        return data

    @property
    def val_0_str(self):
        # type: () -> str
        return self.input_data.val_0_str

    @property
    def val_1_str(self):
        # type: () -> str
        return self.input_data.val_1_str

    @property
    def ys(self):
        # type: () -> np.ndarray
        return self.input_data.ys

    @property
    def input_data(self):
        # type: () -> ClassifyingData
        return self.__input_data

    @input_data.setter
    def input_data(self, value):
        # type: (ClassifyingData) -> None
        self.__input_data = value
        self._data_mask = value.get_mask(self)
        self.__event_data_loaded_internal()

    def __init__(self, draw_plots=False):
        """"""
        self.draw_plots = draw_plots
        self.__event_data_loaded_internal = EventHook()
        self.__event_data_loaded_internal += self.__data_loaded_handler
        self.event_data_loaded = EventHook()
        super(AbstractClassifier, self).__init__()

        self.__input_data = NULL_OBJECT

        self.__data = None
        self._data_mask = None
        self._model = None
        self.score = None

    def _predict(self, data):
        # type: (ClassifyingData) -> np.ndarray
        raise NotImplementedError("prediction must be implemented by concrete classifier")

    def set_data(self, input_data):
        # type: (ClassifyingData) -> None
        self.input_data = input_data

    def classify(self, data):
        # type: (np.matrix) -> np.ndarray
        affective_data = data
        if isinstance(data, ClassifyingData):
            data.normalize()
            affective_data = data.x_mat
        prediction = self._predict(affective_data)
        return np.array(prediction)
        # return np.array(prediction).reshape(-1)

    def train(self, training_set_size_percentage=0.7, show_logs=True):
        # type: (float, bool) -> (object,ClassifyingData)
        sliced_data = self.input_data.slice_data(training_set_size_percentage)

        model = self._train(sliced_data.training_set, sliced_data.training_y)
        self._model = model

        if sliced_data.test_set.size > 0:
            model_score = self.get_model_score(sliced_data.test_set, sliced_data.test_y)
            if show_logs:
                self.log_score(model_score, prefix="Score for test set")

            self.score = model_score

        train_score = None
        if show_logs:
            train_score = self.get_model_score(sliced_data.training_set, sliced_data.training_y)
            self.log_score(train_score, prefix="Score for training set:")

        return self._model, sliced_data, train_score

    def get_model_score(self, test_set, test_y, prediction=None):
        # type: (np.matrix, np.ndarray, np.ndarray) -> ModelScore
        prediction = prediction if prediction is not None \
            else  self.classify(test_set)

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
        self.input_data.normalize()
        self.event_data_loaded(self, self.data, self.ys)
        logger.info("Got {0} features for {1} samples".format(self.feature_count, self.samples_count))

        if draw_data:
            from DataVisualization.Visualyzer import Visualyzer
            Visualyzer.PlotPCA(self.data, self.ys, dim=3)

    def __str__(self):
        return str(self.__class__.__name__)
