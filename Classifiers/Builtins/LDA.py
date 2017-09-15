# coding=utf-8
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
# from sklearn.lda import LDA as sk_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sk_lda



class LDA(AbstractBuiltinClassifier):

    def __init__(self):
        super(LDA, self).__init__()


    def _get_classifier_internal(self):
        lda = sk_lda()
        return lda



