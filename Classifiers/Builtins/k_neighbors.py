from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from sklearn.neighbors import KNeighborsClassifier


class K_Neighbors(AbstractBuiltinClassifier):
    """"""

    def __init__(self, gradient_step_size=None):
        """"""
        super(K_Neighbors, self).__init__()


    def _get_classifier_internal(self,n_neighbors=5):
        return KNeighborsClassifier(n_neighbors=n_neighbors)




