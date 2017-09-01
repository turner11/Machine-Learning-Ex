import os
import matplotlib.pyplot as plt

from copy import deepcopy

import BenchmarkSVM
from DataVisualization.Visualyzer import Visualyzer
from Classifiers.DataLoaders import AbstractDataLoader, CancerDataLoader, ClassifyingData,final_project_data_loader

from Classifiers.DataLoaders import CancerDataLoader
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers.SvmClassifier import SvmClassifier
from Classifiers.builin_LR import Builtin_LR
from Classifiers.gda_downloaded import GaussianDiscriminantAnalysis
from Classifiers import rootLogger as logger, AbstractClassifier

import numpy as np


from Classifiers.LogisticClassifier import LogisticClassifier as lc
from Classifiers.LogisticClassifier_coursera import LogisticClassifier_coursera as lc_coursera
from Classifiers.GaussianGenerativeClassifier import GaussianGenerativeClassifier




def main():
    n_features = 5
    run_best_n_fitures(n=n_features, classifier=Builtin_LR())
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


def classifier_loaded_data_handler(classifier, X, y):
    import logging
    # Code source: Gael Varoquaux
    # License: BSD 3 clause
    print "Got Data from {0}. Plotting PCA data.".format(classifier)
    dim = 3
    Visualyzer.PlotPCA(X, y, dim)


def get_next_best_feature(classifier, selected_idxs, data_loader=None):
    # type: (AbstractClassifier, list, AbstractDataLoader) -> (int,int)
    data_loader = data_loader or get_default_data_loader()

    input_data = data_loader.load()
    orig_x = deepcopy(input_data.x_mat)

    selected_idxs = set(selected_idxs)
    col_idxs = set(range(orig_x.shape[1]))

    idxs_to_check = col_idxs - selected_idxs

    results = []
    classifier.event_data_loaded.clear()
    for col_idx in idxs_to_check:
        features_idxs = list(selected_idxs.union({col_idx}))
        input_data.x_mat = deepcopy(orig_x)
        input_data.x_mat = np.array([c[features_idxs] for c in input_data.x_mat])
        classifier.set_data(input_data)
        # Visualyzer.display_heat_map(logc.normalized_data, logc.ys)
        percentage_for_300 = 0.528
        classifier.train(training_set_size_percentage=percentage_for_300)
        score = classifier.score
        results.append((col_idx, score))

    results = [tpl for tpl in results if not tpl[1].has_nans()]
    results.sort(key=lambda tpl: tpl[1].f_measure, reverse=True)
    best_tpl = results[0]
    best_feature = best_tpl[0]
    score = best_tpl[1].f_measure
    logger.info("Next best feature was: '{0}' with a score of: '{1}')".format(best_feature, score))

    return best_feature,score


def run_best_n_fitures(n=5, classifier=None):
    classifier = classifier or GaussianDiscriminantAnalysis()#lc_coursera()

    selected_idxs = []
    scores = []
    while (len(selected_idxs)) < n:
        next_best, score = get_next_best_feature(classifier, selected_idxs)
        if next_best is None:
            logger.warn("Got a none value for next best index. aborting...")
            break
        selected_idxs.append(next_best)
        scores.append(score)



    logger.info("Top {0} fetures: {1}".format(n,selected_idxs))
    plt.figure()  # new figure
    ax = plt.plot(range(1,len(scores)+1),scores)
    plt.title("results by number of features ({0})".format(classifier))
    plt.ylim([0.9, 1])
    plt.draw()
    plt.show()

    return selected_idxs


def run_classifier(logc, data_loader=None):
    data_loader = data_loader or get_default_data_loader()
    logc.event_data_loaded += classifier_loaded_data_handler

    data = data_loader.load()
    logc.set_data(data)
    # Visualyzer.display_heat_map(logc.normalized_data, logc.ys)
    # training_set_percentage  = percentage_for_300 = 0.528
    training_set_percentage = 0.7
    logc.train(training_set_size_percentage=training_set_percentage)

    # verifying...----------------------------------

    sliced_data = logc.slice_data(training_set_size_percentage=training_set_percentage , normalized=False)

    # get the not yet normed data set that was used for training
    test_set = sliced_data.test_set
    ground_truth = np.array(sliced_data.test_y)
    ys = logc.classify(test_set)
    # did we get same classification?

    score = logc.get_model_score(test_set, ground_truth, prediction=ys)
    # assert score == logc.score, "Got difference in score:\n{0}\n{1}".format(score,logc.score)
    # try:
    #     print ("{0}: alpha: {1}; iterations: {2}".format(logc, logc.gradient_step_size, logc.iterations))
    # except:
    #     pass
    str()





if __name__ == "__main__":

    # import pickle
    # with open('points.pickle','r') as f:
    #     points = pickle.load(f)
    # xa, ya, za = points
    # Visualyzer.plotSufrfce_EXP(xa, ya,za)
    BenchmarkSVM.run()
    if False:
        main()
