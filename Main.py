import matplotlib.pyplot as plt
import numpy as np


from Classifiers.Builtins import Logistic_Regression as bi_lr, Random_Forest as bi_rf
from Classifiers.gda_downloaded import GaussianDiscriminantAnalysis

from Classifiers.LogisticClassifier import LogisticClassifier as lc
from Classifiers.LogisticClassifier_coursera import LogisticClassifier_coursera as lc_coursera
from run_scripts.find_best_features import run_best_n_fitures


def main():
    n_features = 4
    run_best_n_fitures(n=n_features, classifier=bi_rf())

    run_best_n_fitures(n=n_features, classifier=bi_lr())
    return
    run_best_n_fitures(n=n_features , classifier=lc_coursera())
    return
    run_best_n_fitures(n=n_features ,classifier=GaussianDiscriminantAnalysis())
    return
    # LogisticClassifier           -----------------------------------------------
    logc = lc()
    run_classifier(logc)

    # GaussianGenerativeClassifier -----------------------------------------------
    downloaded_gc = GaussianDiscriminantAnalysis()
    run_classifier(downloaded_gc)

    # GaussianGenerativeClassifier -----------------------------------------------
    # gc = GaussianGenerativeClassifier()
    # run_classifier(gc)

    # Coursera -----------------------------------------------
    logc_coursera = lc_coursera()
    run_classifier(logc_coursera)

    str()
    str()













if __name__ == "__main__":
    # BenchmarkSVM.run()
    main()
