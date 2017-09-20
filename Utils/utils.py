import os
import string
validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
sub_folder_name=None

def get_classifiers_folder():
    return get_plots_folder()+os.sep + "classifiers"+os.sep

def get_plots_folder(base_folder=None):
    base_folder = base_folder or os.getcwd()
    folder = base_folder+os.sep+ "plots"+os.sep

    global sub_folder_name
    sub_folder_name = "20170920_074659"
    if sub_folder_name is None:
        import time
        timestr = time.strftime("%Y%m%d_%H%M%S")
        sub_folder_name = timestr

    folder = os.path.join(folder, sub_folder_name+os.sep)
    # return "C:\\Users\\Avi\\PycharmProjects\\exML\\Machine-Learning-Ex\\plots\\20170920_074659"

    if not os.path.exists(folder):
        # import shutil
        os.makedirs(folder)
    return folder


def get_full_plot_file_name(file_name, base_folder=None):
    return get_file_name(file_name, base_folder,suffix="_plot.png" )


def get_file_name(file_name, base_folder=None,suffix=""):
    sanitized = removeDisallowedFilenameChars(file_name)
    fn = os.path.join(get_plots_folder(base_folder),sanitized+ suffix)

    return fn




def removeDisallowedFilenameChars(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    fn = ''.join(c if c in valid_chars else '_' for c in filename)
    fn = fn.replace(' ', '_')  # I don't like spaces in filenames.
    return fn