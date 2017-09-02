# coding=utf-8
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class Gaussian_NB(AbstractBuiltinClassifier):

    def __init__(self):
        super(Gaussian_NB, self).__init__()


    def _get_classifier_internal(self):
        return GaussianNB()



