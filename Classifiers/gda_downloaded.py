# coding=utf-8

import sys
import numpy as np

from Classifiers.AbstractClassifier import AbstractClassifier


class GaussianDiscriminantAnalysis(AbstractClassifier):

    def __init__(self):
        super(GaussianDiscriminantAnalysis, self).__init__()
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None


    def _train(self, t_samples, t_y, lamb=1.0):
        m, n = t_samples.shape

        I0 = [j for j, y in enumerate(t_y) if y == 0]
        I1 = [j for j, y in enumerate(t_y) if y == 1]

        self.phi = 1.0 * len(I1) / m

        self.mu0 = np.mean(t_samples[I0], 0)
        self.mu1 = np.mean(t_samples[I1], 0)

        r = np.matrix(np.zeros([m, n]))
        r[I0] = t_samples[I0] - self.mu0
        r[I1] = t_samples[I1] - self.mu1

        self.sigma = (1.0 / m) * (r.T * r)

    def _predict(self, normed_data):
        r0 = normed_data - self.mu0
        r1 = normed_data - self.mu1

        z0 = np.sum(np.multiply(r0 * self.sigma.I, r0), 1)
        z1 = np.sum(np.multiply(r1 * self.sigma.I, r1), 1)

        s = self.phi * np.exp(-0.5 * z1) - (1.0 - self.phi) * np.exp(-0.5 * z0)

        Y_pred = np.sign(s)
        Y_pred[np.where(Y_pred == -1)] = 0

        return Y_pred

        # A = np.column_stack([Y, Y_pred])
        # print A

        # P = np.matrix(np.zeros(Y.shape))
        # P[np.where(Y == Y_pred)] = 1
        #
        # return 1.0 * P.sum() / len(Y)


if __name__ == '__main__':
    train_path = 'data/heart_scale.train'
    test_path = 'data/heart_scale.test'

    X_train, Y_train = read_dense_data(open(train_path))
    X_test, Y_test = read_dense_data(open(test_path))

    X_train = np.matrix(X_train)
    Y_train = [int(y) for y in Y_train]
    Y_train = np.matrix(Y_train).T
    Y_train[np.where(Y_train == -1)] = 0

    X_test = np.matrix(X_test)
    Y_test = [int(y) for y in Y_test]
    Y_test = np.matrix(Y_test).T
    Y_test[np.where(Y_test == -1)] = 0

    clf = GaussianDiscriminantAnalysis()
    clf.train(X_train, Y_train)

    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for Gaussian Discriminant Analysis : %lf%%' % (100.0 * acc_train)
    print >> sys.stderr, 'Test accuracy for Gaussian Discriminant Analysis : %lf%%' % (100.0 * acc_test)