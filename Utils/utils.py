import os
def get_plots_folder(base_folder=None):
    base_folder = base_folder or os.getcwd()
    folder =  os.path.join(base_folder, "plots"+os.sep)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


def get_full_plot_file_name(file_name, base_folder=None):
    return os.path.join(get_plots_folder(base_folder),file_name+ ".plot")