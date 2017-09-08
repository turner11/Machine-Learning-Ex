
from sklearn.linear_model import LogisticRegression

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class Logistic_Regression(AbstractBuiltinClassifier):
    """"""
    DEFAULT_MAX_ITER = 100

    def __init__(self, max_iter=None):
        """"""
        self.max_iter = max_iter or self.DEFAULT_MAX_ITER
        super(Logistic_Regression, self).__init__()


    def set_classifier(self, **kwargs):
        self.clf = LogisticRegression(max_iter = self.max_iter, solver="sag")









