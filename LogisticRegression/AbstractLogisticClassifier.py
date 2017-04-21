from collections import namedtuple

from LogisticRegression import rootLogger as logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SlicedData = namedtuple('SlicedData', 'training_set training_y test_set test_y')
ModelScore = namedtuple("ModelScore", "precision recall accuracy f_measure")


class AbstractLogisticClassifier(object):
    """"""
    DEFAULT_THRESHOLD = 0.5

    @property
    def feature_count(self):
        # column count
        return self.normalized_data.shape[1]

    @property
    def samples_count(self):
        # rows count
        return len(self.data[:, 0])

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data
        self.__normalized_data = self.normalize_data(self.data)
        logger.info("Got {0} features for {1} samples".format(self.feature_count, self.samples_count))

    @property
    def normalized_data(self):
        return self.__normalized_data

    def __init__(self, gradient_step_size=0.01):
        """"""
        super(AbstractLogisticClassifier, self).__init__()
        self.__data = None
        self.__normalized_data = None

        self.val_0_str = None
        self.val_1_str = None
        self.ys = None
        self.__model = None
        self.threshold = self.DEFAULT_THRESHOLD
        self.gradient_step_size = gradient_step_size

    def load_data_from_csv(self, data_path, classification_column=0):
        try:
            # self.data = np.genfromtxt(data_path, delimiter=',')
            logger.warn("Assuming no headers in csv")
            csv = pd.read_csv(data_path, header=None)
            m = csv.values

            # get the tags
            ys_raw = m[:, classification_column]
            self.val_0_str = ys_raw[0]
            self.val_1_str = next(x for x in ys_raw if x != self.val_0_str)
            ys = [0 if x == self.val_0_str else 1 for x in ys_raw]
            self.ys = np.matrix(ys).transpose()

            # make data hold only numerical values
            m = np.delete(m, [classification_column], 1)

            # data should be numerical
            m = m.astype('float32')
            self.data = m

        except Exception as ex:
            logger.error("Failed to read data:\t{0}".format(str(ex)))
            raise

            # def __str__(self, ):
            #     pass



    def normalize_data(self, data):
        input_avgs = data.mean(axis=0)  # input - by column
        input_std = data.std(axis=0)
        normed = (data - input_avgs) / input_std
        # Adding 1 as the first feature for all
        with_ones = np.zeros((normed.shape[0], normed.shape[1] + 1)) # cerating the new matrix
        with_ones[:, 1:] = normed #populating all but first column
        with_ones[:, 0:1] = 1 # adding 1's in first column
        return np.matrix(with_ones)

    def classify(self, data, is_data_normalized=False):
        normed = data if is_data_normalized else self.normalize_data(data)
        hx = normed * self.__model.transpose()
        prediction = hx >= self.threshold
        return prediction

    def slice_data(self, training_set_size_percentage=0.6, trainingset_size=None):
        if trainingset_size is None:
            if training_set_size_percentage is None or training_set_size_percentage <= 0 or training_set_size_percentage >= 1:
                raise Exception("percentage must be within the (0,1) range")

        trainingset_size = int(self.samples_count * training_set_size_percentage)

        training_set = self.normalized_data[:trainingset_size, :]
        train_y = self.ys[:trainingset_size]

        test_set = self.normalized_data[trainingset_size + 1:, :]
        test_y = self.ys[trainingset_size + 1:, :]

        return SlicedData(training_set, train_y, test_set, test_y)

    def train(self, training_set_size_percentage=0.6, trainingset_size=None):
        sliced_data = self.slice_data(training_set_size_percentage, trainingset_size)

        model = self._train(sliced_data.training_set, sliced_data.training_y)
        self.__model = model

        model_score = self.get_model_score(sliced_data.test_set, sliced_data.test_y)
        self.log_score(model_score,prefix="Score for test set")

        model_score = self.get_model_score(sliced_data.training_set, sliced_data.training_y)
        self.log_score(model_score,prefix="Score for training set:")


        return self.__model

    def get_model_score(self, test_set, test_y):
        prediction = self.classify(test_set, is_data_normalized=True)

        pos = test_y == 1
        neg = test_y == 0

        # True class A (TA) - correctly classified into class A
        tp = np.count_nonzero(prediction[pos] == 1)
        # False class A (FA) - incorrectly classified into class A
        fn = np.count_nonzero(prediction[pos] == 0)
        # True class B (TB) - correctly classified into class B
        tn = np.count_nonzero(prediction[neg] == 0)
        # False class B (FB) - incorrectly classified into class B
        fp = np.count_nonzero(prediction[neg] == 1)

        precision = float(tp) / (tp + fn)
        recall = float(tp) / (tp + fp)

        # You might also need accuracy and F-measure:
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        f_measure = 2 * (float(precision * recall) / (precision + recall))

        return ModelScore(precision=precision, recall=recall, accuracy=accuracy, f_measure=f_measure)

    def get_initial_model(self, size):
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
        initial_model = self.get_initial_model(self.feature_count)

        J, grad = self.cost_function(initial_model, t_samples, t_y)

        logger.info('Cost at initial theta : {0}'.format(J))
        logger.info('Gradient at initial theta: \n{0}'.format(grad))

        model = self.optimize(lambda th: self.cost_function(theta=th, X=t_samples, y=t_y), initial_model,
                              max_itteration_count=500)

        return model

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
        # set J to the cost.
        # Compute the partial derivatives
        # Note: grad should have the same dimensions as theta
        raise NotImplemented

    def cost_function_coursera(self, theta, X, y):
        raise NotImplementedError("Subclasses should implement this!")
        # [J, grad]
        """
        Computes cost and gradient for logistic regression using theta as the parameter
        """
        m = len(y)  # number of training examples
        # need to return the following variables
        J = None
        grad = np.zeros(m)
        # Compute the cost of a particular choice of theta.
        # setting J to the cost.
        # grad should be the partial derivatives
        # Note: grad should have the same dimensions as theta

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

    def optimize(self, cost_function_for_model, initial_model, max_itteration_count=500,
                 stop_at_cost=0.01):
        iter_count = 0
        cost = np.Inf
        model = initial_model

        js = [None] * max_itteration_count
        while (iter_count < max_itteration_count and cost > stop_at_cost):
            cost, grad = cost_function_for_model(model)
            model = model + self.gradient_step_size * grad

            # logger.debug("iter:{0} ; cost: {1}".format(iter_count, cost))
            js[iter_count] = cost
            iter_count += 1

        js = [j for i, j in enumerate(js) if i < iter_count]
        if iter_count == max_itteration_count:
            logger.warn("Exited optimization due to max iteration achieved ({0})".format(iter_count))
            logger.warn("cost was {0}; (max cost - {1})".format(cost, stop_at_cost))

        plt.figure() # new figure
        ax = plt.plot(js)
        plt.title("Cost by iterations ({0})".format(self))
        plt.draw()
        plt.show()

        return model

    def log_score(self, model_score, prefix = ""):
        prefix = (prefix or "Model score") + " ({0})".format(self)
        logger.info(prefix+":\nprecision: {0}\nrecall: {1}\naccuracy: {2}\nf_measure: {3}"
                   .format(model_score.precision, model_score.recall , model_score.accuracy , model_score.f_measure))

    def __str__(self):
        return  str(self.__class__.__name__)
