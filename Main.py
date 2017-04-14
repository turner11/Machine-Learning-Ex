import os
from DataVisualization.Visualyzer import Visualyzer


CSV_NAME = "Data.csv"
from LogisticRegression.LogisticClassifier import LogisticClassifier as lc
def main():
    path = os.path.join(os.path.curdir,CSV_NAME)
    logc = lc()
    logc.load_data_from_csv(path)
    # Visualyzer.display_heat_map(logc.normalized_data, logc.ys)

    model = logc.train(training_set_size_percentage=0.528)
    sliced_data = logc.slice_data(training_set_size_percentage=0.528)

    test_set = logc.data[300:, :]
    test_y = logc.ys[300:, :]
    ys = logc.classify(test_set)

    diffs = ys == test_y
    import numpy as np
    diff_len = len(diffs) - np.count_nonzero(diffs)
    assert  diff_len == 8, "Expected to have a difference of 8... (was {0})".format(diff_len)

    str()

if __name__ == "__main__":
    main()