import os

import itertools

import pickle
import matplotlib.pyplot as plt


from Classifiers.AbstractClassifier import AbstractClassifier
from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers.DataLoaders.ClassifyingData import ClassifyingData, SlicedData
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers import rootLogger as logger
from Classifiers.Ensemble import Ensemble
from Utils.os_utils import File
from Utils.utils import get_full_plot_file_name, get_file_name, get_classifiers_folder
from cloudpickle.utils import to_cloud_pickle, from_cloud_pickle
from run_scripts.find_best_features import run_best_n_fitures




def final_project_main():
    # view_folder_images('C:\\Users\\Avi\\PycharmProjects\\exML\\Machine-Learning-Ex\\plots\\20170926_093917')
    # return
    # find_NN_layers()
    # return

    # find_bset_classifiers_combination()

    # adapt_ensemble_threshold()
    # return

    # classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    # # classifiers.insert(0, Ensemble())
    # n_features = 20
    # for clf in classifiers:
    #     run_best_n_fitures(n=n_features, classifier=clf)
    # return

    # classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    # classifiers.insert(0, Ensemble())
    # compare_pca(classifiers)
    # return

    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    # classifiers.insert(0, Ensemble())

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
    length = len(classifiers)
    for idx, classifier in enumerate(classifiers):
        try:
            classifier_unique_name = "{0}_{1}".format(classifier, idx)
            logger.info("Starting to classify using '{0}' ({1}/{2})".format(classifier, idx+1,length ))
            classifier.train(training_set_size_percentage=0.8)
            score = classifier.score
            results[classifier_unique_name] = score

            logger.info("for '{0}' accuracy was: {1}".format(classifier_unique_name, score.accuracy))

            model_fn = get_file_name("{0}.classifier".format(classifier_unique_name), base_folder=get_classifiers_folder())

            # Save the trained model
            to_cloud_pickle(model_fn, classifier)
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

    pickle_fn = os.path.splitext(fn)[0] + '.pickle'
    with open(pickle_fn, "w") as f:
        cPickle.dump(results,f)


def adapt_ensemble_threshold():
    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    enss = [clf for clf in classifiers if isinstance(clf,Ensemble)]

    data_loader = get_default_data_loader()
    data = data_loader.load()
    data.normalize()
    sliced_data = data.slice_data(training_set_size_percentage=0.8)
    train_data = ClassifyingData(sliced_data.training_y, sliced_data.training_set)


    for i, clf in enumerate(enss):
        clf.set_data(train_data)
        clf.train(training_set_size_percentage=1)



        scores = []
        rng = [v / 100.0 for v in range(10, 105, 5)]
        # rng = [v / 100.0 for v in range(75, 95, 1)]
        for thresh in rng:
            clf.threshold = thresh
            score = clf.get_model_score(sliced_data.test_set, sliced_data.test_y)
            scores.append(score)

        accuracies = [s.accuracy for s in scores]


        plt.clf()
        fig = plt.gcf()
        ax = plt.plot(rng, accuracies, color="blue")
        max_acc = max(accuracies)
        best_thresh =rng[accuracies.index(max_acc)]
        max_points = [(rng[idx], acc) for idx, acc in enumerate(accuracies) if acc == max_acc]

        plt.scatter([p[0] for p in max_points], [p[1] for p in max_points], color='green')
        text = "{0}-{1}_{2} ".format(best_thresh,max_acc,clf.source_file)
        plt.title("Ensemble accuracy by threshold: {0}".format(text))
        plt.ylim([0.9, 1])
        # plt.xlim([0.6,rng[-1]])
        plt.xticks(rng)
        plt.draw()
        plt.show(block=False)
        fn = get_file_name("Ensamble_by_thresh_"+text)+".png"
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
        train_test_split(X, y, test_size=.3, random_state=42)

    only_train_data = ClassifyingData(ys=y_train, x_mat=X_train)
    for clf in classifiers:
        clf.set_data(only_train_data)

    figure = plt.figure(figsize=(27, 9))
    i = 1

    margin = min(H_MESH_STEP, .5)
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
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
    next_subplot = lambda ind: plt.subplot(plot_rows, len(classifiers) // plot_rows + 1, i)
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
        logger.debug(
            "=============================  ({0}\{1}) Starting classifier: '{2}'   =============================".format(
                classifiers.index(clf) + 1, len(classifiers), clf))
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
                       edgecolors='k', alpha=0.5)
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
            logger.error(
                "got a memory error for classifier '{0}' (mesh step was {1}): {2}".format(clf, H_MESH_STEP, ex))
            raise
        except Exception as ex:
            logger.warn("got an error for classifier '{0}': {1}".format(clf, ex))
            raise

    logger.info("Done {0} classifiers".format(len(classifiers)))
    plt.tight_layout()
    plt.show(block=False)
    fn = get_full_plot_file_name("classifiers_boundaries")
    plt.savefig(fn)


def find_NN_layers():
    from Classifiers.Builtins import MPL
    # layrs_structurs = [10,20,30,40,50,60,70,80,90,100,
    #                    (10,10,10), (20,20,20),(30,30,30),(10,20,30),(30,20,10),(40,40,40),(50,60,70),
    #                     (20, 40, 20),(40,20,20),
    #                    (35,20),(55,80),(70,25)]

    rng = range(5,16)
    tpl_1 = set(itertools.combinations(rng, 1))
    tpl_2 = set(itertools.combinations(rng, 2))
    tpl_3 = set(itertools.combinations(rng, 3))
    tpl_4 = set(itertools.combinations(rng, 4))
    layrs_structurs =tpl_1.union(tpl_2).union(tpl_3).union(tpl_4)

    res ={}
    l = len(layrs_structurs)
    for i,strct in enumerate(layrs_structurs):
        classifier = MPL.NNetwork(hidden_layer_sizes=strct)
        print ("testing '{0}' ({1}/{2})".format(classifier,i+1,l))
        model_fn = get_file_name("{0}.classifier".format(classifier), base_folder=get_classifiers_folder())
        if os.path.exists(model_fn):
            print("classifier exists for '{0}'. Continuing...".format(classifier))
            continue
        try:
            iindices, max = run_best_n_fitures(classifier=classifier)
            res[strct] = (iindices, max)
            print("max accuracy for {0} was {1}".format(classifier,max))

            # Save the trained model
            to_cloud_pickle(model_fn, classifier)
        except Exception as ex:
            print("an error accurred for {0}: {1}".format(classifier, ex))

    print (res)

    fn = get_file_name("nn_layres") + '.pickle'
    with open(fn, "w") as f:
        f.write(res)


def scrape_nn_results():
    nn_results = get_nn_results()
    plot_nn_results(nn_results)


def get_nn_results():
    import re
    folder = "C:\\Users\\Avi\\PycharmProjects\\exML\\Machine-Learning-Ex\\plots\\20170920_074659\\"
    files = [folder + fn for fn in os.listdir(folder) if fn.endswith('txt')]
    file_contents = {fn: File.get_text(fn) for fn in files}
    from collections import namedtuple
    def get_acc_from_content(txt):
        accuracies = [l.split('\t')[2].split(' ')[1].strip() for l in txt.split('\n')[4:]]
        accuracies = [float(a) for a in accuracies]
        max_ac =  max([float(ac) for i, ac in enumerate(accuracies)])
        i_feature = accuracies.index(max_ac) +1
        return i_feature, max_ac

    NnStats = namedtuple("NnStats", ["features", "highest_score",'feature_count', "file", "hidden_layer_sizes", "content"])

    res = [NnStats(re.search(r"\[(.*)\]", t).group(), get_acc_from_content(t)[1],get_acc_from_content(t)[0], file, re.search(r"\(\d[\d]*[_\d[\d]*]*\)",file).group(),t) for file, t in
           file_contents.items()]
    s_res = sorted(res, key=lambda stat: stat.highest_score, reverse=True)
    return s_res

plot_idx = 0
def plot_nn_results(nn_results):
    best = nn_results[0:10]
    imagess_f = [(stat, stat.file[0:-3] + "png") for stat in best]
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fn = imagess_f[0][1]
    print(fn)
    img = mpimg.imread(fn)
    fig, ax = plt.subplots()
    imgplot = plt.imshow(img)


    def onclick(args=None):
        # print str(args)
        global plot_idx
        plot_idx = ((plot_idx  + 1) if args.button == 1 else (plot_idx  - 1)) % len(imagess_f)
        stat, img_file = imagess_f[plot_idx ]
        img = mpimg.imread(img_file)
        # imgplot = plt.imshow(img)
        imgplot.set_data(img)
        plt.show()
        plt.title("Highest: {0}".format(stat.highest_score))
        fig.canvas.draw()
        # print str(dir(stat))
        print("'{0}': {1}[0:{2}]".format(stat.hidden_layer_sizes, stat.features,
                                         stat.feature_count))  # str(idx) + str(args) + "aaaaaaaaaaaaaaaaaaaaa"

    connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
    # fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()


def find_bset_classifiers_combination():
    classifiers = AbstractBuiltinClassifier.get_all_working_classifiers()
    nn_results = get_nn_results()
    # best = [s for s in nn_results if s.highest_score >= nn_results[10].highest_score]
    best = nn_results [:5]
    sub_folder = '\\classifiers\\plots\\20170920_074659\\'
    nn_classifers_path = [os.path.dirname(b.file)+sub_folder+os.path.basename(b.file.replace('n_features_',''))[:-5]+").classifier" for b in best]
    def load_clasifier(path):
        with open(path, 'r') as f:
            c = cPickle.load(f)
        return c
    nn_classifers = [load_clasifier(p) for p in nn_classifers_path]
    classifiers.extend(nn_classifers)

    set_lengths = [4]
    tpls = []
    for l in set_lengths:
        curr_tpls = set(itertools.combinations(classifiers, l))
        tpls.extend(curr_tpls)
    # scrape_nn_results()
    data_loader = get_default_data_loader()
    data = data_loader.load()
    data.normalize()
    enss = [Ensemble(classifiers =tpl) for tpl in tpls]
    for clf in enss:
        clf.set_data(data)

    compare_results_full_data(enss)


def view_folder_images(folder):
    global idx
    idx = 1
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    files = os.listdir(folder)
    paths = [os.path.join(folder,fn) for fn in files]
    fig, ax = plt.subplots()

    def load_image(fn):
        img = mpimg.imread(fn)
        # fig, ax = plt.subplots()
        imgplot = plt.imshow(img)
        imgplot.set_data(img)
        fig.canvas.draw()

    load_image(paths[0])

    def onclick(args=None):
        # print str(args)
        global idx
        idx = ((idx+ 1) if args.button == 1 else (idx  - 1)) % len(paths)
        img_file = paths[idx ]
        load_image(img_file)
        plt.show()
        plt.title("{0}".format(os.path.basename(img_file)))
        print("Index: {0}/{1}".format(idx + 1, len(paths)))
        print(img_file)

    connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
    # fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()





