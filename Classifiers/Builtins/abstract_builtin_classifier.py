# coding=utf-8
import os

from Classifiers.AbstractClassifier import AbstractClassifier
from Utils.os_utils import File


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

    def predict_proba(self, normed_data):
        # p = self.clf.predict(normed_data)
        if hasattr(self.clf,'predict_proba'):
            p = self.clf.predict_proba(normed_data)
        else:
            p = None


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
        ens_path = os.path.join(os.getcwd(), 'plots\\20170926_082631\\classifiers\\plots\\20170926_082631\\best\\')
        file_names =os.listdir(ens_path)
        ensembles = [File.get_pickle(os.path.join(ens_path,fn)) for fn in file_names]

        ens_thresholds = {'best_Ensemble_20.classifier': 0.8,
                          'less_overfit_Ensemble_28.classifier': 0.6,
                          'Ensemble_52.classifier':0.5,
                          'Ensemble_45.classifier': 0.6,
                          'Ensemble_25.classifier': 0.8,
                          'Ensemble_57.classifier': 0.8,
                          'Ensemble_54.classifier': 0.8,
                          }
        assert len(ens_thresholds) == len(ensembles)

        for i, e in enumerate(ensembles):
            e.source_file = file_names [i]
            e.threshold = ens_thresholds[e.source_file]





        additionals += ensembles
        ret =  sorted(subs + additionals, key=lambda s: str(s))




        return ret

    @classmethod
    def get_all_working_classifiers(cls):
        # type: () -> [AbstractClassifier]
        classifiers = cls.get_all_classifiers()
        from Classifiers.Builtins.ada_boost import AdaBoost
        from Classifiers.Builtins.quadratic_discriminant_analysis import Quadratic_Discriminant_Analysis
        from Classifiers.Builtins.logistic_regression import Logistic_Regression
        from Classifiers.Builtins.svm_classifier import SvmClassifier
        from Classifiers.Ensemble import Ensemble
        from Classifiers.Builtins.k_neighbors import K_Neighbors
        from Classifiers.Builtins.gaussian_nb import Gaussian_NB
        from Classifiers.Builtins.random_forest import Random_Forest
        from Classifiers.Builtins.DecisionTree import DecisionTree
        from Classifiers.Builtins.LDA import LDA
        from Classifiers.Builtins.MPL import NNetwork
        # classifiers = [c for c in classifiers if
        #                c.__class__ not in [Bernoulli_RBM, Gaussian_Process
        #                                    ##from here, just testing if better of without...
        #                                    # ,AdaBoost
        #                                    # ,Gaussian_NB
        #                                     ,Random_Forest
        #                                     ,DecisionTree
        #                                    ]]

        from Classifiers.Ensemble import Ensemble
        classifiers = [c for c in classifiers
                       if c.__class__ in [Ensemble, Logistic_Regression,Quadratic_Discriminant_Analysis]
                       or (isinstance(c, SvmClassifier) and c.clf.C == 0.025 and c.clf.degree == 6)]


        return classifiers


    # def __str__(self):
    #     return self.name

    def __repr__(self):
        return self.name
