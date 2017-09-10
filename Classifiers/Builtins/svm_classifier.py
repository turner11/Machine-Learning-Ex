# coding=utf-8
import numpy as np
from sklearn import svm

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class SvmClassifier(AbstractBuiltinClassifier):

    def __init__(self):
        super(SvmClassifier, self).__init__()
        self.set_classifier(C=0.5, kernel='rbf', degree=6,gamma='auto')

    def _get_classifier_internal(self, C=None, kernel=None, degree=None,gamma =None):
        # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
        c = C or (self.clf.C if self.clf is not None else None)
        kernel = kernel or (self.clf.kernel if self.clf is not None else None) or 'rbf'
        degree = degree or (self.clf.degree if self.clf is not None else None) or 4
        gamma = gamma or   (self.clf.gamma  if self.clf is not None else None) or 'auto'
        cache_size = 1000
        clf = svm.SVC(C=c, kernel=kernel,cache_size=cache_size , gamma =gamma,degree=degree)

        return clf

    def __str__(self):
        return "{0}: (C={1},k={2},d={3},g={4})"\
                    .format(super(SvmClassifier,self).__str__()
                            ,self.clf.C
                            , self.clf.kernel
                            , self.clf.degree
                            , self.clf.gamma)

