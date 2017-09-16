import os
import matplotlib.pyplot as plt

from Classifiers.AbstractClassifier import AbstractClassifier
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers.DataLoaders.ClassifyingData import ClassifyingData, SlicedData
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers import rootLogger as logger
from Classifiers.Ensemble import Ensemble
from Utils.utils import get_full_plot_file_name, get_file_name, get_classifiers_folder
from cloudpickle.utils import to_cloud_pickle, from_cloud_pickle
from run_scripts.find_best_features import run_best_n_fitures




def final_project_main():
    adapt_ensemble_threshold()
    return
    #
    # classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    # # classifiers.insert(0, Ensemble())
    # n_features = 20
    # for clf in classifiers:
    #     run_best_n_fitures(n=n_features, classifier=clf)
    # return

    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    classifiers.insert(0, Ensemble())
    compare_pca(classifiers)
    # return

    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    classifiers.insert(0, Ensemble())

    data_loader = get_default_data_loader()
    data = data_loader.load()
    data.normalize()
    for clf in classifiers:
        clf.set_data(data)

    compare_results_full_data(classifiers)


def compare_results_full_data(classifiers):
    # type: ([AbstractClassifier]) -> None
    results = {}
    fails = {}
    for classifier in classifiers:
        try:
            logger.info("Starting to classify using {0}')".format(classifier))
            classifier.train(training_set_size_percentage=0.8)
            score = classifier.score
            results[classifier] = score

            model_fn = get_file_name("{0}.classifier".format(classifier),base_folder=get_classifiers_folder())

            # Save the trained model
            to_cloud_pickle(model_fn,classifier)
        except Exception as ex:
            fails[classifier] = ex
            raise
    msg_fails = "\n".join(["{0}: \t\t\t{1}".format(c, fails[c]) for c in fails.keys()])
    res = sorted([(clf, score) for clf, score in results.items()], key=lambda tpl: tpl[1].accuracy, reverse=True)
    msg_results = "\n".join(["{0} ;  ({1})".format(score, clf) for clf, score in res])
    logger.info("\n\nFails:\n{0}".format(msg_fails))
    logger.info("\n\nScors:\n{0}".format(msg_results))

    fn = get_file_name("classifier_comparison")
    summary_file_name = os.path.splitext(fn)[0] + '.txt'
    with open(summary_file_name, "w") as f:
        f.write(msg_results)


def adapt_ensemble_threshold():
    clf = Ensemble()
    data_loader = get_default_data_loader()
    data = data_loader.load()
    data.normalize()
    sliced_data = data.slice_data(training_set_size_percentage=0.6)
    train_data = ClassifyingData(sliced_data.training_y,sliced_data.training_set)


    clf.set_data(train_data )
    clf.train(training_set_size_percentage=1)

    scores = []
    rng = [v/100.0 for v in range(10,105,5)]
    # rng = [v / 100.0 for v in range(75, 95, 1)]
    for thresh in rng:
        clf.threshold = thresh
        score = clf.get_model_score(sliced_data.test_set, sliced_data.test_y)
        scores.append(score)

    accuracies = [s.accuracy for s in scores]

    plt.clf()
    ax = plt.plot(rng, accuracies, color="blue")
    plt.title("Ensemble accuracy by threshold")
    plt.ylim([0.9, 1])
    plt.xlim([rng[0],rng[-1]])
    plt.xticks(rng)
    plt.draw()
    plt.show(block=False)
    fn = get_file_name("Ensamble_by_thresh")
    plt.savefig(fn)







def compare_pca(classifiers):
    # type: ([AbstractBuiltinClassifier]) -> None
    from Classifiers.DataLoaders.final_project_data_loader_PCA import FinalProjectDataLoaderPCA
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap

    H_MESH_STEP = .4  # step size in the mesh

    pca_loader = FinalProjectDataLoaderPCA()
    data = pca_loader.load()

    X, y = data.x_mat, data.ys
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    only_train_data = ClassifyingData(ys=y_train,x_mat=X_train)
    for clf in classifiers:
        clf.set_data(only_train_data)

    figure = plt.figure(figsize=(27, 9))
    i = 1


    margin = min(H_MESH_STEP, .5)
    x_min, x_max = X[:, 0].min() - margin , X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin , X[:, 1].max() + margin
    logger.debug("Starting mesh")
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H_MESH_STEP),
                         np.arange(y_min, y_max, H_MESH_STEP))

    logger.debug("Mesh ended")
    mesh_x, mesh_y = xx.ravel(), yy.ravel()

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plot_rows = 2
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # ax = plt.subplot(plot_rows, len(classifiers) + 1, i)
    next_subplot = lambda ind: plt.subplot(plot_rows, len(classifiers)//plot_rows + 1, i)
    ax = next_subplot(i)
    # ax = plt.subplot(1, len(classifiers) + 1, i)
    if True:  # ds_cnt == 0:
        ax.set_title("Input data")

    logger.debug("scattering ended")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for clf in classifiers:
        name = str(clf)
        logger.debug("=============================  ({0}\{1}) Starting classifier: '{2}'   =============================".format(classifiers.index(clf)+1,len(classifiers),clf))
        try:
            # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            ax = next_subplot(i)
            logger.info("Training....")
            clf.train(training_set_size_percentage=1)
            logger.info("Testing....")
            score = clf.get_model_score(X_test, y_test)
            Z = clf.classify(np.c_[mesh_x, mesh_y])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k',alpha=0.5)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k')

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # if ds_cnt == 0:
            if True:
                # ax.set_title(name,rotation='vertical',x=-0.1,y=0.5)
                ax.set_title(name, size=10)
            # from Classifiers.AbstractClassifier import ModelScore
            ax.text(xx.max() - .3, yy.min() + .3, ('{0:.2f}'.format(score.accuracy)).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
            logger.info("{0}: Done".format(clf))
        except MemoryError as ex:
            logger.error("got a memory error for classifier '{0}' (mesh step was {1}): {2}".format(clf,H_MESH_STEP, ex))
            raise
        except Exception as ex:
            logger.warn("got an error for classifier '{0}': {1}".format(clf, ex))
            raise

    logger.info("Done {0} classifiers".format(len(classifiers)))
    plt.tight_layout()
    plt.show(block=False)
    fn = get_full_plot_file_name("classifiers_boundaries")
    plt.savefig(fn)
