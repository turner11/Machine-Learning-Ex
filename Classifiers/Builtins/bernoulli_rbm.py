# coding=utf-8
from sklearn import svm
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from sklearn.neural_network import BernoulliRBM


class Bernoulli_RBM(AbstractBuiltinClassifier):

    def __init__(self):
        super(Bernoulli_RBM, self).__init__()


    def _get_classifier_internal(self,n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None):
        return BernoulliRBM(n_components=n_components,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            n_iter=n_iter,
                            verbose=verbose,
                            random_state=random_state)


