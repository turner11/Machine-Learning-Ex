# coding=utf-8
import numpy as np
from sklearn import svm

from LogisticRegression.AbstractClassifier import AbstractClassifier


class SvmClassifier(AbstractClassifier):
    def __init__(self):
        super(SvmClassifier, self).__init__()
        self.set_classifier()

    def set_classifier(self, c=1.0, kernel='rbf', degree=3):
        # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
        self.clf = svm.SVC(C=1.0, kernel='rbf', degree=3,cache_size=1000)

    def _train(self, t_samples, t_y):
        a = self.clf.fit(t_samples, t_y)
        return a


    def _predict(self, normed_data):
        p = self.clf.predict(normed_data)
        return p
