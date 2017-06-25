from collections import namedtuple

from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LogisticRegression.AbstractClassifier import AbstractClassifier, draw_plots


class AbstractLogisticClassifier(AbstractClassifier):
    """"""

    DEFAULT_THRESHOLD = 0.5
    DEFAULT__GRAD_STEP = 0.01

    def __init__(self, gradient_step_size=0.01):
        """"""
        super(AbstractLogisticClassifier, self).__init__()
        self.threshold = self.DEFAULT_THRESHOLD

        self.iterations = 1000
        self.gradient_step_size = gradient_step_size or self.DEFAULT__GRAD_STEP




    def normalize_data(self, data):
        normed = super(AbstractLogisticClassifier, self).normalize_data(data)
        with_ones = np.zeros((normed.shape[0], normed.shape[1] + 1))  # cerating the new matrix
        with_ones[:, 1:] = normed  # populating all but first column
        with_ones[:, 0:1] = 1  # adding 1's in first column
        return np.matrix(with_ones)

    def _predict(self, normed_data):
        hx = normed_data * self._model.transpose()
        prediction = hx >= self.threshold
        int_prediction  = prediction + 0
        return int_prediction

    def __get_initial_model(self, size):
        logger.warn("using an initial all 0 model")
        model = [0] * size
        model = np.asarray(model).transpose()
        model = np.matrix(model)
        return model

    def _train(self, t_samples, t_y):
        tag_count = t_y.shape[0]
        sample_count = t_samples.shape[0]
        feature_count = t_samples.shape[1]
        assert sample_count == tag_count, "length of samples did not match length of tags"

        logger.info("Starting to train on {0} features and {1} samples.".format(feature_count, sample_count))
        initial_model = self.__get_initial_model(self.feature_count)

        J, grad = self.cost_function(initial_model, t_samples, t_y)

        logger.info('Cost at initial theta : {0}'.format(J))
        # logger.info('Gradient at initial theta: \n{0}'.format(grad))


        model = self.optimize(lambda th: self.cost_function(theta=th, X=t_samples, y=t_y), initial_model,
                              max_iteration_count=self.iterations)



        return model

    def cost_function(self, theta, X, y):
        """
               Computes cost and gradient for logistic regression using theta as the parameter
               """
        assert theta.shape[1] == X.shape[1], "dimensions of data (X) and model (theta) did not agree. cannot " \
                                             "determine number of features "
        y_fixed = np.matrix(y).transpose()
        m = len(y)  # number of training examples
        n = theta.shape[1]  # number of features
        return self._cost_function(theta, X,y_fixed)

    def _cost_function(self, theta, X, y):
        # [J, grad]
        """
        Computes cost and gradient for logistic regression using theta as the parameter
        """
        m = len(y)  # number of training examples
        # need to return the following variables
        J = 0
        n = theta.shape[1]  # number of features
        # Compute the cost of a particular choice of theta.
        # set J to the cost.
        # Compute the partial derivatives
        # Note: grad should have the same dimensions as theta
        raise NotImplementedError("A concreate logistic regression class should implement the cost function")

    def sigmoid(self, z):
        """
        Compute sigmoid function J = SIGMOID(z) 
            Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
        """
        g = np.zeros(shape=z.shape)
        mz = -z
        e_exp = np.exp(mz)
        g = 1. / (1. + e_exp)
        return g
        # plot(z, g, 'g', 'LineWidth', 2, 'MarkerSize', 4);

    def optimize(self, cost_function_for_model, initial_model, max_iteration_count=500,
                 stop_at_cost=0.01):
        """
        :param cost_function_for_model: the function for testing the cost for a specific model [cost, gradient = function(iterable_model)]
        :param stop_at_cost: at what condition should the optimizing stop 

        :type initial_model: iterable
        :type cost_function_for_model: function
        :type max_iteration_count: int
        """
        # TODO: stop_at_cost should be a percentage of enhancement...
        iter_count = 0
        cost = np.Inf
        model = initial_model

        js = [None] * max_iteration_count
        while (iter_count < max_iteration_count and cost > stop_at_cost):
            cost, grad = cost_function_for_model(model)
            model = model + self.gradient_step_size * grad

            # logger.debug("iter:{0} ; cost: {1}".format(iter_count, cost))
            js[iter_count] = cost
            iter_count += 1

        js = [j for i, j in enumerate(js) if i < iter_count]
        if iter_count == max_iteration_count:
            logger.warn("Exited optimization due to max iteration achieved ({0})".format(iter_count))
            logger.warn("cost was {0}; (max cost - {1})".format(cost, stop_at_cost))

        if self.draw_plots:
            plt.figure()  # new figure
            ax = plt.plot(js)
            plt.title("Cost by iterations ({0})".format(self))
            plt.draw()
            plt.show()

        return model

