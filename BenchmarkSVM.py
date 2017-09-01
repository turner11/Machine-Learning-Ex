from collections import namedtuple, defaultdict
import pickle
import os



from DataVisualization.Visualyzer import Visualyzer
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers.SvmClassifier import SvmClassifier
import numpy as np

from Classifiers import rootLogger as logger
from Utils.utils import get_full_plot_file_name, get_plots_folder

Point = namedtuple("Point","Kernal, c, f_measure")
Point_3D = namedtuple("Point_3D","Kernal, c, gamma, f_measure")


def run():
    data_loader = get_default_data_loader()
    data = data_loader.load()

    kernels = ['rbf', 'sigmoid', 'linear', 'poly']
    Cs = np.linspace(0.0001,4, 200)


    # Data we will be looping over
    is_first = True
    i =1
    count = len(Cs) * len(kernels)

    folder = get_plots_folder()
    for kernel in kernels:
        logger.info("======================================= Running SVM for kernel: {0} ======================================= ".format(kernel))
        classifier = SvmClassifier()
        classifier.set_data(data)

        _run_svm_3d(classifier,Cs,kernel)
        points = []

        if is_first:
            Visualyzer.PlotPCA(classifier.normalized_data, data.ys, dim=3)
            is_first = False

        for c in Cs:
            logger.debug("############################# SVM: {0}/{1}  [Kernel: {2}; C: {3:.5f}]  #############################".format(i, count,kernel,c ))
            # -------------------------------Setting classifier with kernel --------------------------------
            i += 1
            classifier.set_classifier(c=c, kernel=kernel, gamma='auto')
            f_measure = classifier.score.f_measure
            points.append(Point(kernel,c, f_measure))

        points = sorted(points,key=lambda p:p.f_measure,reverse=True)
        file_name = "points_"+kernel+".pickle"
        full_path = os.path.join(folder, file_name)
        with open(full_path, 'w') as f:
            pickle.dump(points, f)


    file_name = 'telemetry'
    full_path = os.path.join(folder,file_name)
    # with open(file_name, 'r') as f:
    #     telemetry = pickle.load(f)

    # telemetry[classifier.clf.kernel] = s_values
    with open(full_path, 'w') as f:
        pickle.dump(points, f)

    # import pickle
    # with open('points.pickle','w') as f:
    #     pickle.dunmp(points,f)
    def onclick(event):
        print(
        'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
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

    # Visualyzer.plotSufrfce(x, y, z, xlabel="gamma", ylabel="C", zlabel="F Measurew", click_callback=onclick,
    #                        block=True)
    str()

def _run_svm_3d(classifier, Cs,kernel):
    # type: (SvmClassifier, list) -> None

    gammas = [0.1, 0.01, 0.001]
    pairs = __get_pairs(gammas , Cs)

    points = []
    i = 1
    classifier.set_classifier(kernel=kernel)
    for gamma, C in pairs:
        logger.debug("############################# SVM: {0}/{1}  [Gamma: {2:.10f}; C: {3}]  #############################".format(i, len(pairs),gamma,C))
        i += 1

        # -------------------------------Going over all combinations of kernel and C --------------------------------
        classifier.set_classifier(c=C, gamma=gamma)
        f_measure = classifier.score.f_measure
        points.append(Point_3D(kernel, C, gamma, f_measure))

    points = sorted([p for p in points if not np.isnan(p.f_measure)], key=lambda p: p.c)
    x= [ point.c for point in points]
    y = [point.gamma for point in points]
    z = [point.f_measure for point in points]

    base_file_name = "points3D_" + kernel
    Visualyzer.plotSufrfce(x, y, z, xlabel="gamma", ylabel="C", zlabel="F Measurew",block=False,file_name=base_file_name +".plot")
    points = sorted(points, key=lambda p: p.f_measure, reverse=True)
    folder = get_plots_folder()
    file_name = os.path.join(folder,base_file_name+ ".pickle")
    with open(file_name, 'w') as f:
        pickle.dump(points, f)
    str()

def __get_pairs(l1, l2):
    import itertools
    ret = list(itertools.product(l1, l2))
    return ret


if __name__ == "__main__":
    import re
    import itertools
    import matplotlib.pyplot as plt

    test =3

    files  =os.listdir(os.getcwd())
    regex_3d = re.compile('points3D_.*pickle')
    regex_2d = re.compile('points_.*pickle')
    fn_points_3D = filter(regex_3d.match, files)
    fn_points_2D = filter(regex_2d.match, files)

    def get_points(fn):
        with open(fn) as fd:
            pnt = pickle.load(fd)
        return pnt


    points_3D =[get_points(fn) for fn in fn_points_3D]
    points_2D =[get_points(fn) for fn in fn_points_2D]
    # kernels = set([p.Kernal for p in points_2D ])
    p_2 = list(itertools.chain.from_iterable(points_2D))
    p_3 = list(itertools.chain.from_iterable(points_3D))
    dic2 = defaultdict(lambda :[])
    dic3 = defaultdict(lambda: [])
    for p in p_2:
        dic2[p.Kernal].append(p)

    for p in p_3:
        dic3[p.Kernal].append(p)
    if test <=2:
        for kernel, points in dic2.items():
            srtd = sorted(points, key=lambda p: p.c)
            cs = [p.c for p in srtd ]
            fs = [p.f_measure for p in srtd ]

            best = max(srtd, key=lambda p:p.f_measure if not np.isnan(p.f_measure) else -1)




            plt.figure(kernel)
            ax = plt.plot(cs,fs)
            plt.scatter([best.c],[best.f_measure],c='green', s=100, alpha=0.6, edgecolors='none')

            plt.xlabel('C')
            plt.ylabel('f measure')
            plt.title("{0}: Best - C:{1:.3f}; f_measure: {2:.3f}".format(kernel,best.c,best.f_measure))
            plt.grid(False)
            plt.axis('tight')
            plt.savefig("{0}_2D_plot.png".format(kernel))
            plt.show(block=False)
    else:
        for kernel, points in dic3.items():
            srtd = sorted([p for p in points if not np.isnan(p.f_measure)], key=lambda p: p.c)
            cs = [p.c for p in srtd]
            gs = [p.gamma for p in srtd]
            fs = [p.f_measure for p in srtd]

            best = max(srtd, key=lambda p: p.f_measure if not np.isnan(p.f_measure) else -1)
            file_name = get_full_plot_file_name("plots_3d_{0}".format(kernel))

            title = "{0}: Best - C:{1:.3f}; gamma:{2:.3f}; f_measure: {3:.3f}".format(kernel, best.c, best.gamma, best.f_measure)
            Visualyzer.plotSufrfce(x=gs, y=cs, z=fs, xlabel="gamma", ylabel="C", zlabel="F Measurew", block=False,title=title,
                                   file_name=file_name)
        str()

    # import numpy as np
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # import random
    # from scipy.interpolate import griddata
    # from matplotlib import cm
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.asarray(gs)
    # y = np.asarray(cs)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array(fs)
    # # Z2 = zs.reshape(X.shape)
    # # Z = np.random.random(X.shape)
    # xi = np.linspace(x.min(), x.max(), 50)
    # yi = np.linspace(y.min(), y.max(), 50)
    # Z = griddata((x, y), zs, (X, Y), method='cubic')
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.title('Meshgrid Created from 3 1D Arrays')
    # # ~~~~ MODIFICATION TO EXAMPLE ENDS HERE ~~~~ #
    #
    # plt.show()

    points = 2
