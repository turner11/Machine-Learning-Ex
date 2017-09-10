import os

from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader

class FinalProjectDataLoader(AbstractDataLoader):
    """"""
    CSV_NAME = "real_project_data.csv"

    def __init__(self,set_mask=True):
        """"""
        my_folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(my_folder, self.CSV_NAME)
        super(self.__class__, self).__init__(path,None,20)
        self.set_mask=set_mask
        pass

    def load(self):
        classifying_data = super(FinalProjectDataLoader, self).load()
        if self.set_mask:
            classifying_data.get_mask = self.get_mask
        return classifying_data


    @staticmethod
    def get_mask(classifier):
        if classmethod is None:
            return None
        from Classifiers.Builtins.ada_boost import AdaBoost
        from Classifiers.Builtins.quadratic_discriminant_analysis import Quadratic_Discriminant_Analysis
        from Classifiers.Builtins.logistic_regression import Logistic_Regression
        from Classifiers.Builtins.svm_classifier import SvmClassifier
        from Classifiers.Ensemble import Ensemble
        from Classifiers.Builtins.k_neighbors import K_Neighbors
        from Classifiers.Builtins.gaussian_nb import Gaussian_NB
        from Classifiers.Builtins.random_forest import Random_Forest
        from Classifiers.Builtins.DecisionTree import DecisionTree
        masks = {
            Quadratic_Discriminant_Analysis: [7, 9, 4]
            , Logistic_Regression: [7, 9, 16]
            , Ensemble:None
            , K_Neighbors: [3, 4, 2, 12, 1, 18, 7, 19]
            , AdaBoost: [7, 1, 18, 4, 19, 11, 5, 9, 16]
            # , Gaussian_NB: [7, 13, 19]
            , Random_Forest: [7, 11, 10, 8, 17, 14, 1]
            , DecisionTree: [7, 1, 18, 4]
            # , SvmClassifier: {'rbf': [7, 9, 14, 15, 1, 5, 4], 'linear': [7, 0, 19, 1, 4, 3, 11, 5]}
            , SvmClassifier: {'rbf': [7, 9, 14, 1, 15, 5], 'linear': [7, 0, 19, 1, 4, 3, 11, 5]}
        }

        clf_class = classifier.__class__
        ret = None
        if clf_class in masks:
            mask = masks[clf_class]
            if isinstance(classifier, SvmClassifier):
                ret = mask[classifier.clf.kernel]
            else:
                ret = mask

        return ret








