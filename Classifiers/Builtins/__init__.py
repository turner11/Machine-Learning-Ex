from Classifiers.Builtins import logistic_regression, \
    random_forest, \
    svm_classifier, \
    DecisionTree, \
    ada_boost, \
    bernoulli_rbm, \
    gaussian_process, \
    k_neighbors, \
    gaussian_nb, \
    quadratic_discriminant_analysis,\
    LDA




Logistic_Regression = logistic_regression.Logistic_Regression
Random_Forest = random_forest.Random_Forest
SvmClassifier = svm_classifier.SvmClassifier
DecisionTree = DecisionTree.DecisionTree
AdaBoost = ada_boost.AdaBoost
BernoulliRBM = bernoulli_rbm.BernoulliRBM
GaussianProcess = gaussian_process.GaussianProcess
K_Neighbors = k_neighbors.K_Neighbors
GaussianNB = gaussian_nb.GaussianNB
Quadratic_Discriminant_Analysis = quadratic_discriminant_analysis.Quadratic_Discriminant_Analysis
LDA = LDA.LDA


# from os.path import dirname, basename, isfile
# import glob
# modules = glob.glob(dirname(__file__)+"/*.py")
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]