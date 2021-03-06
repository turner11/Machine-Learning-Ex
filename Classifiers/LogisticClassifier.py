from collections import namedtuple

from Classifiers import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Classifiers.AbstractLogisticClassifier import AbstractLogisticClassifier


class LogisticClassifier(AbstractLogisticClassifier):
    """"""

    def __init__(self, gradient_step_size=None):
        """"""
        super(self.__class__, self).__init__(gradient_step_size)

    def _cost_function(self, theta, X, y):
        # [J, grad]
        """
        Computes cost and gradient for logistic regression using theta as the parameter
        """
        n = theta.shape[1]  #number of features
        # need to return the following variables
        J = 0
        grad = np.zeros(n)
        # Compute the cost of a particular choice of theta.
        # setting J to the cost.
        # Compute the partial derivatives
        # setting grad to the partial derivatives of the cost
        # Note: grad should have the same dimensions as theta
        hx = X * theta.transpose()
        sigmoidH = self.sigmoid(hx)

        diff = y - sigmoidH
        square_diff =np.power(diff, 2)
        summedCost = np.sum(square_diff)
        J = summedCost /2.0

        # -----------------Grad
        diff_x = diff.transpose() * X
        grad = diff_x

        return J, grad


