from Classifiers.AbstractLogisticClassifier import AbstractLogisticClassifier
from sklearn.linear_model import LogisticRegression
class Builtin_LR(AbstractLogisticClassifier):
    """"""

    def __init__(self, gradient_step_size=None):
        """"""
        super(self.__class__, self).__init__(gradient_step_size)
        self.logistic = LogisticRegression(max_iter = self.iterations, solver="sag")
        pass

    def _train(self, t_samples, t_y):
        self.logistic.fit(t_samples, t_y)

    def _predict(self, normed_data):
        prediction = self.logistic.predict(normed_data)
        # prediction1 = self.logistic.predict_proba(normed_data)
        return prediction




