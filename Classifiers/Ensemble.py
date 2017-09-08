# coding=utf-8
import numpy as np

from Classifiers.AbstractClassifier import AbstractClassifier, ModelScore
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


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
            model = clf._train(t_samples, t_y)
            models.append(model)
        return models

    def _predict(self, normed_data):
        predictions = []
        for clf in self.classifiers:
            p = clf.classify(normed_data)
            predictions.append(p)

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