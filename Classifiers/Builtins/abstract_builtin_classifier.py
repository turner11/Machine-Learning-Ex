# coding=utf-8
import numpy as np


from Classifiers.AbstractClassifier import AbstractClassifier


class AbstractBuiltinClassifier(AbstractClassifier):

    def __init__(self):
        super(AbstractBuiltinClassifier, self).__init__()
        self.clf = None
        self.set_classifier()

    def _get_classifier_internal(self, **kwargs):
        raise NotImplementedError("this should be set by concreate classifier")

    def set_classifier(self, **kwargs):
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.clf = self._get_classifier_internal(**kwargs)
        if self.samples_count > 0:
            self.train()


    def _train(self, t_samples, t_y):
        a = self.clf.fit(t_samples, t_y)
        return a


    def _predict(self, normed_data):
        p = self.clf.predict(normed_data)
        return p
