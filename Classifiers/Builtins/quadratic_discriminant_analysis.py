# coding=utf-8
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class Quadratic_Discriminant_Analysis(AbstractBuiltinClassifier):

    def __init__(self):
        super(Quadratic_Discriminant_Analysis, self).__init__()


    def _get_classifier_internal(self,
                                priors=None,
                                 reg_param=0., store_covariances=False,
                                 tol=1.0e-4):
        # priors=None,
        # reg_param=0.,
        # store_covariances=False,
        # tol=1.0e-4
        return QuadraticDiscriminantAnalysis(priors =priors ,
                                            reg_param =reg_param ,
                                            store_covariances =store_covariances ,
                                            tol =tol)



