import matplotlib.pyplot as plt
import numpy as np
import os

from Utils.utils import get_plots_folder


class Visualyzer(object):
    """"""
    current_data = None
    def __init__(self, ):
        """"""
        super(self.__class__, self).__init__()

    @classmethod
    def display_heat_map(cls, dataa, tags):
        fig, ax = plt.subplots()
        # Advance color controls
        ax.pcolor(dataa, cmap=plt.cm.rainbow, edgecolors='k')#gist_heat

        feature_count = dataa.shape[1]
        data_size = dataa.shape[0]

        y_labels = np.arange(0, data_size, step=data_size //10)  # the sequential number
        x_labels = np.arange(0,feature_count)
        ax.set_xticks(x_labels)
        ax.set_yticks(y_labels)
        # Here we position the tick labels for x and y axis
        ax.xaxis.tick_bottom()
        # ax.yaxis.tick_left()
        # Values against each labels
        ax.set_xticklabels(tags, minor=True, fontsize=5)
        ax.set_yticklabels(y_labels, minor=False, fontsize=10)
        plt.show(block=False)



    @classmethod
    def plotSufrfce(cls, x, y, z,xlabel='X',ylabel='Y',zlabel='Z',click_callback=None, block=False, title=None, file_name=None):

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from matplotlib import cm
        from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

        Xs = np.array(x)
        Ys = np.array(y)
        Zs = np.array(z)

        # ======
        ## plot:

        fig = plt.figure(title or file_name or  "NA")
        ax = Axes3D(fig)
        if title:
            plt.title(title)




        surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0
                                ,antialiased=False)

        fig.colorbar(surf)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)



        if click_callback is not None:
            cid = fig.canvas.mpl_connect('button_press_event', click_callback )
            # cid = fig.canvas.mpl_connect('pick_event', click_callback )

        # fig.tight_layout()
        if file_name is not None:
            folder = get_plots_folder()
            full_path = os.path.join(folder, "{0}.jpg".format(file_name))
            plt.show(block=block)  # or:
            plt.savefig(full_path)

    @classmethod
    def PlotPCA(cls, X, y, dim):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn import decomposition

        X = np.asarray(X)
        y = np.asarray(y)
        try:
            c_x, c_y = cls.current_data or ((np.asarray([np.inf]*X.shape[0])),np.asarray([np.inf]*X.shape[1]))
            same_x = X.shape == c_x.shape and (X == c_x).all()
            same_y =  y.shape == c_y.shape and  (y == c_y).all()
            if same_x and same_y:
                # dont plot data again...
                return

            cls.current_data = (X,y)
        except Exception as ex:
            raise


        feature_count = X.shape[1]
        if (dim > feature_count):
            print('cannot perform PCA')
            return

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        pca = decomposition.PCA(n_components=dim)
        pca.fit(X)
        X = pca.transform(X)

        unique_labels = np.unique(y)
        labels = [("Label: " + str(lbl), lbl) for lbl in unique_labels]

        for name, label in labels:
            ax.text3D(X[y == label, 0].mean(),
                      X[y == label, 1].mean() + 1.5,
                      X[y == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)

        cm = plt.cm.get_cmap('RdYlBu')  # plt.cm.spectral
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cm)

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])


        plt.show(block=False)



