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

        if classifier is None:
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
        from Classifiers.Builtins.LDA import LDA
        from Classifiers.Builtins.MPL import NNetwork
        masks = {
            Quadratic_Discriminant_Analysis: [7, 9, 4, 5, 18, 1, 15, 6, 8, 11, 3, 19, 2, 13, 17, 16, 14, 12, 0, 10][0:17]
            , Logistic_Regression: [7, 9, 16, 11, 15, 8, 1, 4, 5, 6, 0, 14, 17, 12, 2, 18, 10, 3, 13, 19][0:12]
            , Ensemble:None
            , NNetwork:[7, 13, 9, 1, 4, 11, 17, 5, 6, 8, 16, 2, 10, 12, 3, 0, 18, 19, 15, 14][0:14]
            #[7, 9, 11, 1, 2, 4, 5, 17, 8, 0, 15, 10, 6, 13, 16, 19, 12, 18, 14, 3][0:7]
            # [7, 9, 5, 8, 1, 17, 11, 10, 3, 4, 12, 0, 19, 6, 2, 18, 15, 13, 16, 14]


            , AdaBoost: [7, 1, 15, 6, 14, 5, 0, 19, 12, 8, 3, 11, 13, 16, 17, 2, 10, 9, 4, 18][0:17] #Overfit alarm!
            , Random_Forest: [2, 16, 6, 10, 5, 8, 15, 3, 14, 0, 1, 4, 9, 19, 12, 17, 11, 13, 18, 7][0:9]
            , DecisionTree: [7, 8, 15, 19, 14, 6, 10, 12, 9, 0, 11, 3, 2, 16, 13, 17, 5, 4, 18, 1][0:15]
            , K_Neighbors: [3, 4, 0, 1, 18, 2, 6, 13, 10, 12, 7, 14, 16, 9, 19, 8, 11, 5, 17, 15][0:14]
            , LDA:[7, 17, 3, 1, 0, 6, 8, 9, 16, 2, 4, 5, 10, 11, 13, 18, 14, 19, 15, 12][0:11]
            , SvmClassifier: {
                               'rbf': [7, 9, 14, 1, 15, 5, 4, 6, 16, 17, 13, 8, 19, 2, 0, 10, 11, 3, 12, 18][0:6],
                                'linear': [7, 0, 19, 1, 4, 3, 11, 5, 2, 8, 13, 15, 16, 17, 9, 10, 14, 12, 18, 6][0:12]
                             }

            , Gaussian_NB: [7, 13, 19, 9, 10, 0, 4, 14, 1, 6, 8, 2, 16, 11, 18, 3, 17, 12, 15, 5][0:7] #TODO: Is it doing any good??
            # , SvmClassifier: {'rbf': [7, 9, 14, 15, 1, 5, 4], 'linear': [7, 0, 19, 1, 4, 3, 11, 5]}
        }

        clf_class = classifier.__class__
        ret = None
        if clf_class in masks:
            mask = masks[clf_class]
            if isinstance(classifier, SvmClassifier):
                ret = mask[classifier.clf.kernel]
            else:
                ret = mask

        return ret or None








