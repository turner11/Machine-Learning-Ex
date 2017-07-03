# coding=utf-8
import numpy as np
from sklearn import svm

from LogisticRegression.AbstractClassifier import AbstractClassifier


class SvmClassifier(AbstractClassifier):

    def __init__(self):
        super(SvmClassifier, self).__init__()
        self.set_classifier(c=1.0, kernel='rbf', degree=4,gamma='auto')

    def set_classifier(self, c=None, kernel=None, degree=None,gamma =None):
        # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
        c = c or self.clf.C
        kernel = kernel or self.clf.kernel or 'rbf'
        degree = degree or self.clf.degree or 4
        gamma = gamma or self.clf.gamma or 'auto'
        cache_size = 1000
        self.clf = svm.SVC(C=c, kernel=kernel,cache_size=cache_size , gamma =gamma,degree=degree)
        if self.samples_count > 0:
            self.train()

    def _train(self, t_samples, t_y):
        a = self.clf.fit(t_samples, t_y)
        return a


    def _predict(self, normed_data):
        p = self.clf.predict(normed_data)
        return p
