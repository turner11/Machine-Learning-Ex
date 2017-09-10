import os
from sklearn import decomposition

from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader


class FinalProjectDataLoaderPCA(AbstractDataLoader):
    """"""
    CSV_NAME = "real_project_data.csv"

    def __init__(self, dim=2):
        """"""
        self.dim = dim
        my_folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(my_folder, self.CSV_NAME)
        super(self.__class__, self).__init__(path,None,20)

    def load(self):
        classifying_data =  super(FinalProjectDataLoaderPCA, self).load()
        X = classifying_data.x_mat
        pca = decomposition.PCA(n_components=self.dim)
        pca.fit(X)
        X_pca = pca.transform(X)
        classifying_data.x_mat = X_pca
        return classifying_data
