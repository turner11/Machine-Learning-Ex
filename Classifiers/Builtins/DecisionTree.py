# coding=utf-8
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier


class DecisionTree(AbstractBuiltinClassifier):

    def __init__(self):
        super(DecisionTree, self).__init__()


    def _get_classifier_internal(self,
                                 criterion="gini",
                                 splitter="best",
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.,
                                max_features=None,
                                random_state=None,
                                max_leaf_nodes=None,
                                class_weight=None,
                                presort=None):
        # criterion = "gini",
        # splitter = "best",
        # max_depth = None,
        # min_samples_split = 2,
        # min_samples_leaf = 1,
        # min_weight_fraction_leaf = 0.,
        # max_features = None,
        # random_state = None,
        # max_leaf_nodes = None,
        # class_weight = None,
        # presort = False
        return DecisionTreeClassifier(
                                    criterion=criterion,
                                    splitter=splitter,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                    max_features=max_features,
                                    random_state=random_state,
                                    max_leaf_nodes=max_leaf_nodes,
                                    class_weight=class_weight,
                                    presort=presort)



