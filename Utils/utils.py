import os
import string
import unicodedata
validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)


def get_plots_folder(base_folder=None):
    base_folder = base_folder or os.getcwd()
    folder =  os.path.join(base_folder, "plots"+os.sep)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


def get_full_plot_file_name(file_name, base_folder=None, add_time_stamp=False):
    if add_time_stamp:
        import time
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file_name = timestr+"_"+file_name

    sanitized = removeDisallowedFilenameChars(file_name)
    fn = os.path.join(get_plots_folder(base_folder),sanitized+ "_plot.png")

    return fn




def removeDisallowedFilenameChars(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    fn = ''.join(c if c in valid_chars else '_' for c in filename)
    fn = fn.replace(' ', '_')  # I don't like spaces in filenames.
    return fn