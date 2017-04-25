from collections import defaultdict

from LogisticRegression.AbstractClassifier import AbstractClassifier
from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GaussianGenerativeClassifier(AbstractClassifier):
    """"""

    def __init__(self, gradient_step_size=0.01):
        """"""
        super(GaussianGenerativeClassifier, self).__init__()

    def _train(self, t_samples, t_y):


        # #   m : array_like
        # #        A 1-D selor 2-D array containing multiple variables and observations.
        # #        Each row of `m` represents a variable, and each column a single
        # #        observation of all those variables. Also see `rowvar` below.
        # m = t_samples.transpose()
        #
        # cov = np.cov(m)
        # cov_det = np.linalg.det(cov)
        #
        # model_prefix = 1.0/(np.sqrt(2*np.pi)*np.power(cov_det,0.5))
        y_list = list(np.array(t_y).reshape(-1,))
        all_classes = np.unique(y_list)

        #
        average_by_class = dict((cls,np.mean(t_samples[t_y == cls])) for cls in all_classes)
        average_by_class = defaultdict(lambda :None, average_by_class)
        m = len(t_y)
        one_over_m = 1.0/m
        def internal_predict(x):
            diffs = [x - avg for cls, avg in average_by_class]
            prod_diffs = np.prod(diffs)
            sum_prod_diffs = np.sum(prod_diffs)
            prediction = one_over_m * sum_prod_diffs
            return prediction




        return  internal_predict()
        # all_classes = set(t_y)
        # models = defaultdict(lambda: None)
        # for cls in all_classes:
        #     idxs = t_y == cls
        #     data = t_samples[idxs]
        #     curr_model = self.__build_model(data)
        #     models[cls] = curr_model
        #
        # return models

    def _predict(self, normed_data):
        internal_predict_func = self._model
        predictions = [internal_predict_func(features) for features in normed_data ]
        return predictions

    # def __build_model(self, data):
    #     """
    #     Computes a gaussian model for cls's data
    #     """
    #
    #
    #     ## This is the gaussian model for a single variable
    #     # avg = np.average(data)
    #     # std = np.std(data)
    #     #
    #     # exp_power = (-0.5)*np.power(((data - avg)/std),2)
    #     # exp = np.exp(exp_power)
    #     # gaussian = exp/(np.sqrt(2*np.pi)*std)
    #     # return gaussian
    #     raise NotImplementedError
