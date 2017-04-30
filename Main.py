import os
from DataVisualization.Visualyzer import Visualyzer
from LogisticRegression.gda_downloaded import GaussianDiscriminantAnalysis
import numpy as np

CSV_NAME = "Data.csv"
from LogisticRegression.LogisticClassifier import LogisticClassifier as lc
from LogisticRegression.LogisticClassifier_coursera import LogisticClassifier_coursera as lc_coursera
from LogisticRegression.GaussianGenerativeClassifier import GaussianGenerativeClassifier


def main():
    logc = lc()
    run_classifier(logc)
    downloaded_gc = GaussianDiscriminantAnalysis()
    run_classifier(downloaded_gc)

    gc = GaussianGenerativeClassifier()
    run_classifier(gc)


    logc_coursera = lc_coursera()
    run_classifier(logc_coursera)

    str()


def classifier_loaded_data(classifier, X,y):
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
    labels = [("Label: "+str(lbl), lbl) for lbl in unique_labels]

    for name, label in labels :
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)

    cm = plt.cm.get_cmap('RdYlBu')#plt.cm.spectral
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cm)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()


def run_classifier(logc):
    logc.event_data_loaded += classifier_loaded_data
    path = os.path.join(os.path.curdir, CSV_NAME)
    logc.load_data_from_csv(path)
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

    score = logc.get_model_score(test_set, ground_truth, prediction=ys )
    assert score == logc.score, "Got difference in score:\n{0}\n{1}".format(score,logc.score)
    str()


if __name__ == "__main__":
    main()
