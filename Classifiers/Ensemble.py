# coding=utf-8
import numpy as np

from Classifiers.AbstractClassifier import AbstractClassifier, ModelScore
from Classifiers.AbstractLogisticClassifier import AbstractLogisticClassifier
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers import rootLogger as logger


class Ensemble(AbstractClassifier):
    @property
    def name(self):
        return self.clf.__class__.__name__

    def __init__(self):
        super(Ensemble, self).__init__()
        self.classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()


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
        threshold = 0.5
        m = np.matrix(predictions)
        avgs = np.mean(m, axis=0)
        my_prediction = avgs > threshold
        #convert to int...
        my_prediction = my_prediction*1
        return my_prediction

    def set_data(self, input_data):
        super(Ensemble, self).set_data(input_data)
        for clf in self.classifiers:
            clf.set_data(input_data)

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)
