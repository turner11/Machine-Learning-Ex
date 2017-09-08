class ClassifyingData(object):
    """"""

    def __init__(self, ys, x_mat, val_0_str=None, val_1_str=None):
        """"""
        super(self.__class__, self).__init__()
        self.ys = ys
        self.x_mat = x_mat


        self.val_0_str = val_0_str or ys[0]
        self.val_1_str = val_1_str or next(x for x in ys if x != val_0_str)
