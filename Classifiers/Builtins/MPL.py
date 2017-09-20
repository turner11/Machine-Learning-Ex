# coding=utf-8
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from sklearn.neural_network import MLPClassifier

class NNetwork(AbstractBuiltinClassifier):

    def __init__(self,hidden_layer_sizes=(100,)):
        hidden_layer_sizes = (hidden_layer_sizes,) if isinstance(hidden_layer_sizes,int) else hidden_layer_sizes
        self.hidden_layer_sizes=hidden_layer_sizes
        super(NNetwork, self).__init__()


    def _get_classifier_internal(self,**kwargs):
        clf = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes,**kwargs)
        return  clf

    def __str__(self):
        return "{0}(hidden_layer_sizes={1})".format(self.__class__.__name__,self.hidden_layer_sizes)





