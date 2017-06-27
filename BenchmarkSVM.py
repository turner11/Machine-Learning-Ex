from DataVisualization.Visualyzer import Visualyzer
from LogisticRegression.DataLoaders.Utils import get_default_data_loader
from LogisticRegression.SvmClassifier import SvmClassifier
import numpy as np

from LogisticRegression import rootLogger as logger


def run():
    data_loader = get_default_data_loader()
    data = data_loader.load()
    training_set_percentage = 0.7

    # Data we will be looping over
    kernels = ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
    Cs = [0.1 + v / 1000.0 for v in range(0, 1000, 10)]
    pairs = __get_pairs(kernels, Cs)

    is_first = True

    points = [[], [], []]
    i = 1
    for ker, C in pairs:
        logger.debug("############################# SVM: {0}/{1}  #############################".format(i,len(pairs)))
        i+=1
        # -------------------------------Setting classifier with kernel --------------------------------
        classifier = SvmClassifier()
        classifier.set_data(data)

        if is_first:
            dim = 3
            Visualyzer.PlotPCA(classifier.normalized_data, data.ys, dim)
            is_first = False

        # -------------------------------Going over all combinations of kernel and C --------------------------------

        classifier.set_classifier(c=C, kernel=ker)  # ,kernel='Spline'
        # TODO: Add modifing the classifier to use degree and C
        classifier.train(training_set_size_percentage=training_set_percentage)

        # TODO: Modify kernel as well...
        # TODO: Add allowing mistakes (outliers)


        # verifying...----------------------------------

        sliced_data = classifier.slice_data(training_set_size_percentage=training_set_percentage, normalized=False)

        # get the not yet normed data set that was used for training
        test_set = sliced_data.test_set
        ground_truth = np.array(sliced_data.test_y)
        ys = classifier.classify(test_set)
        # did we get same classification?

        score = classifier.get_model_score(test_set, ground_truth, prediction=ys)

        f_measure = score.f_measure
        points[0].append(kernels.index(ker))
        points[1].append(C)
        points[2].append(f_measure * 100)

    points = [np.asarray(v) for v in points]
    x, y, z = points
    s_idx = sorted(enumerate(points[2]),key=lambda t: t[1],reverse=True)
    idxs = [t[0] for t in s_idx]
    best = [points[0][idxs[0:5]],points[1][idxs[0:5]],points[2][idxs[0:5]]]
    worse = [points[0][idxs[-5:-1]],points[1][idxs[-5:-1]],points[2][idxs[-5:-1]]]
    # import pickle
    # with open('points.pickle','w') as f:
    #     pickle.dunmp(points,f)

    def onclick(event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
        # Record the x location of the user's click in the global variable and close the figure
        # global retval
        retval = event.xdata
        curr_ker, curr_c = (event.xdata, event.ydata)
        # findong closest
        idx = np.argmin([np.linalg.norm((curr_ker, curr_c) - np.asarray([p[0],p[1]])) for p in points])
        kernel = kernels[points[0][idx]]
        c = points[1][idx]
        f_measure = points[2][idx]
        msg = "kernel: {0};  C: {1};  f-measure: {2}".format(kernel,c,f_measure)
        fig = event.canvas.figure
        fig.set_label(msg)
        print msg





    Visualyzer.plotSufrfce(x, y, z, xlabel="Kernel", ylabel="C", zlabel="F Measurew", click_callback=onclick, block=True)
    str()







def __get_pairs(l1, l2):
    import itertools
    ret = list(itertools.product(l1, l2))
    return ret
