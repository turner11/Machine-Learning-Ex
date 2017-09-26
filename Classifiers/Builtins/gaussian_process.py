# coding=utf-8
from sklearn import svm
from sklearn.gaussian_process.gaussian_process import MACHINE_EPSILON

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from sklearn.gaussian_process import GaussianProcess
import numpy as np


class Gaussian_Process(AbstractBuiltinClassifier):

    def __init__(self):
        super(Gaussian_Process, self).__init__()


    def _get_classifier_internal(self,
                                 regr='constant', corr='squared_exponential', beta0=None,
                                 storage_mode='full', verbose=False, theta0=1e-1,
                                 thetaL=None, thetaU=None, optimizer='fmin_cobyla',
                                 random_start=1, normalize=True,
                                 nugget=10. * MACHINE_EPSILON, random_state=None):

        # regr = 'constant'
        # corr = 'squared_exponential'
        # beta0 = None,
        # storage_mode = 'full'
        # verbose = False
        # theta0 = 1e-1
        # thetaL = None
        # thetaU = None
        # optimizer = 'fmin_cobyla'
        # random_start = 1
        # normalize = True
        # nugget = 10. * np.finfo(float).eps
        # random_state = NoneRandomForestClassifier

        # GaussianProcessRegressor
        return GaussianProcess(regr=regr
                                ,corr=corr
                                ,beta0=beta0
                                ,storage_mode=storage_mode
                                ,verbose=verbose
                                ,theta0=theta0
                                ,thetaL=thetaL
                                ,thetaU=thetaU
                                ,optimizer=optimizer
                                ,random_start=random_start
                                ,normalize=normalize
                                ,nugget=nugget
                                ,random_state=random_state )

        

