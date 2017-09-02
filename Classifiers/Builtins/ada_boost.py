# coding=utf-8
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class AdaBoost(AbstractBuiltinClassifier):

    def __init__(self):
        super(AdaBoost, self).__init__()


    def _get_classifier_internal(self,
                                base_estimator=None,
                                 n_estimators=50,
                                 learning_rate=1.,
                                 algorithm='SAMME.R',
                                random_state=None):
        # base_estimator = None,
        # n_estimators = 50,
        # learning_rate = 1.,
        # algorithm = 'SAMME.R',
        # random_state = None
        return AdaBoostClassifier(  base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    algorithm=algorithm,
                                    random_state=random_state)



