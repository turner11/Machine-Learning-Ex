from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from Classifiers.AbstractClassifier import AbstractClassifier
from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers.gda_downloaded import GaussianDiscriminantAnalysis
from Classifiers import rootLogger as logger


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
    best_tpl = results[0] if len(results) > 0 else None
    best_feature = best_tpl[0] if best_tpl is not None else None
    score = best_tpl[1].f_measure if best_tpl is not None else None
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

    logger.info("Top {0} fetures: {1}".format(len(selected_idxs),selected_idxs))
    summary = "\n".join(["{0}: {1}%".format(idx+1, sc) for ((idx,  f), sc) in zip(enumerate(selected_idxs),scores)])
    logger.info("\nFinal results for {0}:\nfeatures{1}\ngrade for number of features:\n{2}"
                .format(classifier, selected_idxs,summary ))

    plt.figure()  # new figure
    ax = plt.plot(range(1,len(scores)+1),scores)
    plt.title("results by number of features ({0})".format(classifier))
    plt.ylim([0.9, 1])
    plt.draw()
    plt.show()

    return selected_idxs