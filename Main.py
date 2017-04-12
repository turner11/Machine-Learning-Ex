import os
from DataVisualization.Visualyzer import Visualyzer


CSV_NAME = "Data.csv"
from LogisticRegression.LogisticClassifier import LogisticClassifier as lc
def main():
    path = os.path.join(os.path.curdir,CSV_NAME)
    logc = lc()
    logc.load_data_from_csv(path)
    Visualyzer.display_heat_map(logc.normalized_data, logc.ys)

    model = logc.train(training_set_size_percentage=0.528)
    str()

if __name__ == "__main__":
    main()