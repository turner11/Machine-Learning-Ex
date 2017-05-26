import os
import matplotlib.pyplot as plt

from copy import deepcopy

from DataVisualization.Visualyzer import Visualyzer
from LogisticRegression.DataLoader import DataLoader, ClassifyingData
from LogisticRegression.gda_downloaded import GaussianDiscriminantAnalysis
from LogisticRegression import rootLogger as logger, AbstractClassifier

import numpy as np

CSV_NAME = "Data.csv"
from LogisticRegression.LogisticClassifier import LogisticClassifier as lc
from LogisticRegression.LogisticClassifier_coursera import LogisticClassifier_coursera as lc_coursera
from LogisticRegression.GaussianGenerativeClassifier import GaussianGenerativeClassifier


def main():
    n_features = 5
    run_best_n_fitures(n=n_features ,classifier=GaussianDiscriminantAnalysis())
    run_best_n_fitures(n=n_features , classifier=lc_coursera())
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
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn import decomposition
    centers = [[1, 1], [-1, -1], [1, -1]]
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    unique_labels = np.unique(y)
    labels = [("Label: " + str(lbl), lbl) for lbl in unique_labels]

    for name, label in labels:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)

    cm = plt.cm.get_cmap('RdYlBu')  # plt.cm.spectral
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cm)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()


def get_next_best_feature(classifier, data_path, selected_idxs):
    # type: (AbstractClassifier, str, list) -> (int,int)

    input_data = DataLoader.from_csv(data_path)
    orig_x = deepcopy(input_data.x_mat)

    selected_idxs = set(selected_idxs)
    col_idxs = set(range(orig_x.shape[1]))

    idxs_to_check = col_idxs - selected_idxs

    results = []
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
    path = os.path.join(os.path.curdir, CSV_NAME)
    selected_idxs = []

    scores = []
    while (len(selected_idxs)) < n:
        next_best, score = get_next_best_feature(classifier, path, selected_idxs)
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


def run_classifier(logc):
    logc.event_data_loaded += classifier_loaded_data_handler
    path = os.path.join(os.path.curdir, CSV_NAME)
    data = DataLoader.from_csv(path)
    logc.set_data(data)
    # Visualyzer.display_heat_map(logc.normalized_data, logc.ys)
    percentage_for_300 = 0.528
    logc.train(training_set_size_percentage=percentage_for_300)

    # verifying...----------------------------------

    sliced_data = logc.slice_data(training_set_size_percentage=percentage_for_300, normalized=False)

    # get the not yet normed data set that was used for training
    test_set = sliced_data.test_set
    ground_truth = np.array(sliced_data.test_y)
    ys = logc.classify(test_set)
    # did we get same classification?

    score = logc.get_model_score(test_set, ground_truth, prediction=ys)
    # assert score == logc.score, "Got difference in score:\n{0}\n{1}".format(score,logc.score)
    try:
        print ("{0}: alpha: {1}; iterations: {2}".format(logc, logc.gradient_step_size, logc.iterations))
    except:
        pass
    str()


if __name__ == "__main__":
    main()
