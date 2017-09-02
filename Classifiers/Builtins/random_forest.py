# coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class Random_Forest(AbstractBuiltinClassifier):

    def __init__(self):
        super(Random_Forest, self).__init__()
        self.clf = None
        self.set_classifier()

    def _get_classifier_internal(self, n_estimators=None,
                             criterion=None,
                             max_depth=None,
                             min_samples_split=None,
                             min_samples_leaf=None,
                             min_weight_fraction_leaf=None,
                             max_features=None,
                             max_leaf_nodes=None,
                             bootstrap=None,
                             oob_score=None,
                             n_jobs=None,
                             random_state=None,
                             verbose=None,
                             warm_start=None,
                             class_weight=None):
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        
        clf = RandomForestClassifier()
        self_clf = self.clf or clf

        clf.n_estimators= n_estimators or self_clf.n_estimators
        clf.criterion= criterion or self_clf.criterion
        clf.max_depth= max_depth or self_clf.max_depth
        clf.min_samples_split= min_samples_split or self_clf.min_samples_split
        clf.min_samples_leaf= min_samples_leaf or self_clf.min_samples_leaf
        clf.min_weight_fraction_leaf= min_weight_fraction_leaf or self_clf.min_weight_fraction_leaf
        clf.max_features= max_features or self_clf.max_features
        clf.max_leaf_nodes= max_leaf_nodes or self_clf.max_leaf_nodes
        clf.bootstrap= bootstrap or self_clf.bootstrap
        clf.oob_score= oob_score or self_clf.oob_score
        clf.n_jobs= n_jobs or self_clf.n_jobs
        clf.random_state= random_state or self_clf.random_state
        clf.verbose= verbose or self_clf.verbose
        clf.warm_start= warm_start or self_clf.warm_start
        clf.class_weight= class_weight or self_clf.class_weight


        return clf

