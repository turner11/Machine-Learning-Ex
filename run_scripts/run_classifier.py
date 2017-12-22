from Classifiers.DataLoaders.Utils import get_default_data_loader
from DataVisualization.Visualyzer import Visualyzer


def classifier_loaded_data_handler(classifier, X, y):
    import logging
    # Code source: Gael Varoquaux
    # License: BSD 3 clause
    print("Got Data from {0}. Plotting PCA data.".format(classifier))
    dim = 3
    Visualyzer.PlotPCA(X, y, dim)


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
