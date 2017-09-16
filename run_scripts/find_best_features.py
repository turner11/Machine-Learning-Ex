from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os

from Classifiers.AbstractClassifier import AbstractClassifier, ModelScore
from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers.gda_downloaded import GaussianDiscriminantAnalysis
from Classifiers import rootLogger as logger
from Utils.utils import get_full_plot_file_name,get_file_name


def get_next_best_feature(classifier, selected_idxs, data_loader=None):
    # type: (AbstractClassifier, list, AbstractDataLoader) -> (int,int)
    data_loader = data_loader or get_default_data_loader(set_mask=False)

    input_data = data_loader.load()
    input_data.normalize()
    orig_x = deepcopy(input_data.x_mat)

    selected_idxs = set(selected_idxs)
    col_idxs = set(range(orig_x.shape[1]))

    idxs_to_check = col_idxs - selected_idxs

    results = []
    classifier.event_data_loaded.clear()
    for col_idx in idxs_to_check:
        features_idxs = list(selected_idxs.union({col_idx}))
        input_data.x_mat = deepcopy(orig_x)
        input_data.x_mat = input_data.x_mat[:,features_idxs]  #np.array([c[features_idxs] for c in input_data.x_mat])
        classifier.set_data(input_data)
        # Visualyzer.display_heat_map(logc.normalized_data, logc.ys)
        percentage_for_300 = 0.528
        _, _, train_score = classifier.train(training_set_size_percentage=percentage_for_300)
        score = classifier.score
        results.append((col_idx, score,train_score))

    results = [tpl for tpl in results if not tpl[1].has_nans()]
    results.sort(key=lambda tpl: tpl[1].accuracy , reverse=True)
    best_tpl = results[0] if len(results) > 0 else None
    best_feature = best_tpl[0] if best_tpl is not None else None
    score, train_score = (best_tpl[1],best_tpl[2])  if best_tpl is not None else (ModelScore(),ModelScore())
    logger.info("Next best feature was: '{0}' with a score of: '{1}')".format(best_feature, score))

    return best_feature,score, train_score


def run_best_n_fitures(n=5, classifier=None, plot_block=False):
    classifier = classifier or GaussianDiscriminantAnalysis()#lc_coursera()

    selected_idxs = []
    scores = []
    training_scores = []
    while (len(selected_idxs)) < n:
        next_best, score, train_score = get_next_best_feature(classifier, selected_idxs)
        if next_best is None:
            logger.warn("Got a none value for next best index. aborting...")
            break
        selected_idxs.append(next_best)
        scores.append(score)
        training_scores.append(train_score)

    logger.info("Top {0} fetures: {1}".format(len(selected_idxs),selected_idxs))
    summary = "\n".join(["{0}: {1}%".format(idx+1, sc) for ((idx,  f), sc) in zip(enumerate(selected_idxs),scores)])

    full_msg = "\nFinal results for {0}:\nfeatures{1}\ngrade for number of features:\n{2}".format(classifier, selected_idxs,summary )
    logger.info(full_msg)


    plt.figure()  # new figure
    accuracies = [s.accuracy for s in scores]
    train_accuracies = [s.accuracy for s in training_scores]
    rng = range(1,len(accuracies)+1)
    ax = plt.plot(rng,accuracies, color="blue", label="test set")
    ax = plt.plot(rng, train_accuracies, color='red', alpha=0.5,label="training set")

    max_acc = max(accuracies)
    max_points =[(idx+1,acc) for idx,acc in enumerate(accuracies) if acc ==  max_acc]
    plt.scatter([p[0] for p in max_points],[p[1] for p in max_points], color='green')
    plt.legend(loc='upper left')


    plt.title("results by number of features ({0})".format(classifier))
    plt.ylim([0.9, 1])
    plt.xlim([1,n])
    plt.xticks(range(1,n+1))
    plt.draw()
    plt.show(block=plot_block)

    fn = get_file_name("n_features_{0}".format(classifier))
    plt.savefig(fn)

    summary_file_name = os.path.splitext(fn)[0]+'.txt'
    with open(summary_file_name,"w") as f:
        f.write(full_msg)

    return selected_idxs