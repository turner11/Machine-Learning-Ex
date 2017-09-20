# coding=utf-8
import numpy as np

from Classifiers.AbstractClassifier import AbstractClassifier, ModelScore
from Classifiers.AbstractLogisticClassifier import AbstractLogisticClassifier
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers import rootLogger as logger


class Ensemble(AbstractClassifier):
    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self,threshold=0.82):
        super(Ensemble, self).__init__()
        self.classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
        self.threshold = threshold


    def _train(self, t_samples, t_y):
        models = []
        for clf in self.classifiers:
            try:
                model = clf._train(t_samples, t_y)
                models.append(model)
            except:
                logger.error("{0} got an error while training internal {1}".format(self,clf))
                raise
        return models

    def _predict(self, normed_data):
        predictions = []
        for clf in self.classifiers:
            try:
                logger.debug("{0} classifying using instance: {1}".format(self, clf))
                p = clf.classify(normed_data)
                logger.debug("{0} Done".format(clf))
                predictions.append(p)
            except:
                logger.error("{0} got an error while predicting using internal {1}".format(self,clf))
                raise


        prediction = self.__consolidate_predications(predictions)
        return prediction

    def __consolidate_predications(self, predictions):
        all_predictions_same_length = len(set([len(p) for p in predictions])) == 1
        assert all_predictions_same_length

        m = np.matrix(predictions)
        avgs = np.mean(m, axis=0)
        my_prediction = avgs > self.threshold
        #convert to int...
        my_prediction = np.asarray(my_prediction*1).reshape(-1)
        return my_prediction

        # predictors_count = len(predictions)
        # all_predictors = range(0, predictors_count)
        # determnistic_idxs = [i for i, p_set in enumerate(predictions) if len(np.unique(p_set)) == 2]
        # indexes_to_include = all_predictors \
        #     if len(determnistic_idxs) * 1.0 / predictors_count > 0.5 \
        #     else [idx for idx in all_predictors if idx not in determnistic_idxs]
        # preds = []
        # predictions_count = len(predictions[0])
        # for idx in range(0, predictions_count):
        #     preds_1 = np.asarray([p[idx][1] for p in predictions])
        #     preds_1 = preds_1[indexes_to_include]
        #     decsision = np.average(preds_1) > self.threshold
        #     preds.append(decsision)
        #
        # return np.asarray(preds) * 1

    def set_data(self, input_data):
        # type: (ClassifyingData) -> None
        super(Ensemble, self).set_data(input_data)
        for clf in self.classifiers:
            clf.set_data(input_data)

    def __repr__(self):
        return "{0}()".format(self.name)
