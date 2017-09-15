# coding=utf-8


from Classifiers.AbstractClassifier import AbstractClassifier


class AbstractBuiltinClassifier(AbstractClassifier):
    @property
    def name(self):
        return self.clf.__class__.__name__

    def __init__(self):
        super(AbstractBuiltinClassifier, self).__init__()
        self.clf = None
        self.set_classifier()

    def _get_classifier_internal(self, **kwargs):
        raise NotImplementedError("this should be set by concreate classifier")

    def set_classifier(self, **kwargs):
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.clf = self._get_classifier_internal(**kwargs)
        if self.samples_count > 0:
            self.train()

    def _train(self, t_samples, t_y):
        a = self.clf.fit(t_samples, t_y)
        return a

    def _predict(self, normed_data):
        p = self.clf.predict(normed_data)
        return p

    @classmethod
    def get_all_classifiers(cls):
        # type: () -> [AbstractClassifier]
        from Classifiers.Builtins.svm_classifier import SvmClassifier
        linear_svm = SvmClassifier()
        linear_svm.set_classifier(kernel="linear", C=0.025)
        # from . import *

        # rbf_svm = SvmClassifier()
        # rbf_svm.set_classifier(kernel='rbf', gamma=2, C=1)

        subs = [s() for s in cls.__subclasses__()]
        subs = [s for s in subs if str(s).lower() != "namexxx"]
        additionals = [linear_svm]#, rbf_svm]
        ret =  sorted(subs + additionals, key=lambda s: str(s))
        return ret

    @classmethod
    def get_all_working_classifiers(cls):
        # type: () -> [AbstractClassifier]
        classifiers = cls.get_all_classifiers()
        from Classifiers.Builtins.bernoulli_rbm import Bernoulli_RBM
        from Classifiers.Builtins.gaussian_process import Gaussian_Process
        from Classifiers.Builtins.ada_boost import AdaBoost
        from Classifiers.Builtins.gaussian_nb import Gaussian_NB
        from Classifiers.Builtins.random_forest import Random_Forest
        from Classifiers.Builtins.DecisionTree import DecisionTree
        classifiers = [c for c in classifiers if
                       c.__class__ not in [Bernoulli_RBM, Gaussian_Process
                                           ##from here, just testing if better of without...
                                           # ,AdaBoost
                                           # ,Gaussian_NB
                                           # ,Random_Forest
                                           # ,DecisionTree
                                           ]]


        return classifiers


    # def __str__(self):
    #     return self.name

    def __repr__(self):
        return self.name
