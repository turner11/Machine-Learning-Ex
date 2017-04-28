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


def run_classifier(logc):
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
    assert str(score) == str(logc.score), "Got difference in score..."
    str()


if __name__ == "__main__":
    main()
