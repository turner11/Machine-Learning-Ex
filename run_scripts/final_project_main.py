from Classifiers.AbstractClassifier import AbstractClassifier
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers.DataLoaders.ClassifyingData import ClassifyingData
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers import rootLogger as logger
from Classifiers.Ensemble import Ensemble


def final_project_main():
    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    classifiers.insert(0, Ensemble())

    compare_pca(classifiers)
    return
    data_loader = get_default_data_loader()
    data = data_loader.load()
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
        except Exception as ex:
            fails[classifier] = ex
    msg_fails = "\n".join(["{0}: \t\t\t{1}".format(c, fails[c]) for c in fails.keys()])
    res = sorted([(clf, score) for clf, score in results.items()], key=lambda tpl: tpl[1].f_measure, reverse=True)
    msg_results = "\n".join(["{0} ;  ({1})".format(score, clf) for clf, score in res])
    logger.info("\n\nFails:\n{0}".format(msg_fails))
    logger.info("\n\nScors:\n{0}".format(msg_results))


def compare_pca(classifiers):
    # type: ([AbstractBuiltinClassifier]) -> None
    from Classifiers.DataLoaders.final_project_data_loader_PCA import FinalProjectDataLoaderPCA
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap

    H_MESH_STEP = .5  # step size in the mesh

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



    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
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
            ax.text(xx.max() - .3, yy.min() + .3, ('{0:.2f}'.format(score.f_measure)).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
        except Exception as ex:
            logger.warn("got an error for classifier '{0}': {1}".format(clf, ex))
            raise

    plt.tight_layout()
    plt.show(block=True)
