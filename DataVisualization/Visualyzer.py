import matplotlib.pyplot as plt
import numpy as np

class Visualyzer:
    """"""

    def __init__(self, ):
        """"""
        super(self.__class__, self).init()

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
        plt.show()




