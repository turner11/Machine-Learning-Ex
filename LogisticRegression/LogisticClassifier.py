from collections import namedtuple

from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LogisticRegression.AbstractLogisticClassifier import AbstractLogisticClassifier


class LogisticClassifier(AbstractLogisticClassifier):
    """"""

    def __init__(self, gradient_step_size=1):
        """"""
        super(self.__class__, self).__init__(gradient_step_size)

    def cost_function(self, theta, X, y):
        # [J, grad]
        """
        Computes cost and gradient for logistic regression using theta as the parameter
        """
        m = len(y)  # number of training examples
        # need to return the following variables
        J = 0
        grad = np.zeros(m)
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


