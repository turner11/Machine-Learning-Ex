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

        assert len(all_classes) == 2, "Current implementationh supports only 2 classes"

        # tuples of (class, indices)
        cidx = [(cls,[i for i,c in enumerate(y_list)if c == cls]) for cls in all_classes]
        average_by_class = [(cls, np.mean(t_samples[idxs])) for cls, idxs in cidx]
        average_by_class = defaultdict(lambda :None, average_by_class)
        m = len(y_list)
        one_over_m = 1.0/m




        cov = np.cov(t_samples,rowvar=False)
        cov_det = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        predix_denominator = np.sqrt(2 * np.pi) *np.sqrt(cov_det)
        model_prefix = 1.0 / predix_denominator
        avg_0, avg_1 = average_by_class[0], average_by_class[1]



        def internal_predict(xs):
            sample_count = xs.shape[0]
            results = []
            for x in xs:

                best_likelihood = -np.Infinity
                chosen_class = None

                for cls,avg in average_by_class.items():
                    exp_power = -0.5 * ((x - avg) * inv_cov * (x - avg).transpose())
                    model_result = model_prefix * np.exp(exp_power)

                    if model_result  > best_likelihood:
                        best_likelihood = model_result
                        chosen_class = cls

                results.append(chosen_class)

            return results if len(results) != 1 else results[0]

        # x_0 = t_samples[0]
        # x_19 = t_samples[19]
        #
        # test = internal_predict(t_samples)
        # y_0 = internal_predict(x_0)
        # y_19 = internal_predict(x_19)

        return internal_predict





        # def internal_predict(x):
        #     diffs = [x - avg for cls, avg in average_by_class.items()]
        #     prod_diffs = np.prod(diffs)
        #     sum_prod_diffs = np.sum(prod_diffs)
        #     prediction = one_over_m * sum_prod_diffs
        #     return prediction




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
