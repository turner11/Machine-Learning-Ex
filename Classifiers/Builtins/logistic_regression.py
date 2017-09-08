from Classifiers.AbstractLogisticClassifier import AbstractLogisticClassifier
from sklearn.linear_model import LogisticRegression
class Logistic_Regression(AbstractLogisticClassifier):
    """"""

    def __init__(self, gradient_step_size=None):
        """"""
        super(Logistic_Regression, self).__init__(gradient_step_size)
        self.logistic = LogisticRegression(max_iter = self.iterations, solver="sag")
        pass

    def _train(self, t_samples, t_y):
        model = None
        try:
            model = self.logistic.fit(t_samples, t_y)
        except Exception as ex:
            data = self.add_one_column(t_samples)
            model = self.logistic.fit(t_samples, t_y)
        return model

    def _predict(self, normed_data):
        # HACK for ex...
        affective_data = normed_data
        if normed_data.shape[1] == self.feature_count -1:
            affective_data = self.add_one_column(normed_data)
        prediction = self.logistic.predict(affective_data)
        # prediction1 = self.logistic.predict_proba(normed_data)
        return prediction




