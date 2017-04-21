from collections import namedtuple

from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LogisticRegression.AbstractLogisticClassifier import AbstractLogisticClassifier



class LogisticClassifier_coursera(AbstractLogisticClassifier):
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

        pos_logic = y == 1
        neg_logic = y == 0
        pos_vals = sigmoidH[pos_logic]
        neg_vals = sigmoidH[neg_logic]

        # makes sure that only predictions of POSITIVE(y == 1) is taken in considerations
        cost_FalsePos = -np.log(pos_vals)

        # makes sure that only predictions of NEGITIVE(y == 0) is taken in considerations
        cost_FalseNeg = -np.log(1 - neg_vals)
        summedCost = np.sum(cost_FalseNeg) + np.sum(cost_FalsePos)
        J = (1.0 / m) * summedCost

        # -----------------Grad
        diff = y - sigmoidH
        diffX = diff.transpose() * X
        grad = (1.0 / m) * diffX
        return J, grad

