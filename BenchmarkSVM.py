from DataVisualization.Visualyzer import Visualyzer
from LogisticRegression.DataLoaders.Utils import get_default_data_loader
from LogisticRegression.SvmClassifier import SvmClassifier
import numpy as np

from LogisticRegression import rootLogger as logger


def run():
    data_loader = get_default_data_loader()
    data = data_loader.load()


    # Data we will be looping over
    is_first = True
    kernels = ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
    for kernel in kernels:
        logger.info("======================================= Running SVM for kernel: {0} ======================================= ".format(kernel))
        # -------------------------------Setting classifier with kernel --------------------------------
        classifier = SvmClassifier()
        classifier.set_data(data)

        if is_first:
            dim = 3
            Visualyzer.PlotPCA(classifier.normalized_data, data.ys, dim)
            is_first = False

        _run_svm(classifier)


def _run_svm(classifier):
    # type: (SvmClassifier) -> object

    gammas = np.linspace(0.0001, 3, 30)
    Cs = [0.1 + v / 1000.0 for v in range(0, 1000, 100)]
    pairs = __get_pairs(gammas , Cs)

    points = [[], [], []]
    i = 1
    for gamma, C in pairs:
        logger.debug("############################# SVM: {0}/{1}  [Gamma: {2:.10f}; C: {3}]  #############################".format(i, len(pairs),gamma,C))
        i += 1

        # -------------------------------Going over all combinations of kernel and C --------------------------------

        classifier.set_classifier(c=C, gamma=gamma)

        # verifying...----------------------------------

        sliced_data = classifier.slice_data(training_set_size_percentage=classifier.training_set_size_percentage, normalized=False)

        # get the not yet normed data set that was used for training
        test_set = sliced_data.test_set
        ground_truth = np.array(sliced_data.test_y)
        ys = classifier.classify(test_set)
        # did we get same classification?

        score = classifier.get_model_score(test_set, ground_truth, prediction=ys)

        f_measure = score.f_measure
        points[0].append(gamma)
        points[1].append(C)
        points[2].append(f_measure * 100)

    points = [np.asarray(v) for v in points]
    x, y, z = points
    s_idx = sorted(enumerate(points[2]), key=lambda t: t[1], reverse=True)
    idxs = [t[0] for t in s_idx]
    s_values = [points[0][idxs],points[1][idxs],points[2][idxs]]
    best = [s_values[0][0:5], s_values[1][0:5], s_values[2][0:5]]
    worse = [s_values[0][-5:-1], s_values[1][-5:-1], s_values[2][-5:-1]]
    import pickle
    file_name = 'telemetry'
    with open(file_name, 'r') as f:
        telemetry = pickle.load(f)

    telemetry[classifier.clf.kernel] = s_values
    with open(file_name , 'w') as f:
        pickle.dump(telemetry, f)

    # import pickle
    # with open('points.pickle','w') as f:
    #     pickle.dunmp(points,f)
    def onclick(event):
        print( 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
        # Record the x location of the user's click in the global variable and close the figure
        # global retval
        retval = event.xdata
        curr_ker, curr_c = (event.xdata, event.ydata)
        # findong closest
        idx = np.argmin([np.linalg.norm((curr_ker, curr_c) - np.asarray([p[0], p[1]])) for p in points])
        kernel = StandardError(classifier.clf.kernel)
        gamma = points[0][idx]
        c = points[1][idx]
        f_measure = points[2][idx]
        msg = "gamma: {0};  C: {1};  f-measure: {2}".format(kernel, c, f_measure)
        fig = event.canvas.figure
        fig.set_label(msg)
        print msg

    Visualyzer.plotSufrfce(x, y, z, xlabel="gamma", ylabel="C", zlabel="F Measurew", click_callback=onclick,
                           block=True)
    str()


def __get_pairs(l1, l2):
    import itertools
    ret = list(itertools.product(l1, l2))
    return ret
