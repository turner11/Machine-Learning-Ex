import sys

from Classifiers import rootLogger as logger
from Classifiers.Builtins import Logistic_Regression as bi_lr, Random_Forest as bi_rf
from Classifiers.LogisticClassifier import LogisticClassifier as lc
from Classifiers.LogisticClassifier_coursera import LogisticClassifier_coursera as lc_coursera
from Classifiers.gda_downloaded import GaussianDiscriminantAnalysis
from run_scripts.find_best_features import run_best_n_fitures


def main():
    if len(sys.argv) > 1:
        # C:\Users\Avi\Anaconda2\python.exe  C:\Users\Avi\PycharmProjects\exML\Machine-Learning-Ex\Main.py c:\a.xls
        from Classifiers.DataLoaders.final_project_data_loader import FinalProjectDataLoader
        from Utils.os_utils import File
        import logging
        logger.level = logging.INFO
        xls_file = sys.argv[1]
        loader = FinalProjectDataLoader()
        loader.data_path = xls_file
        data = loader.load()

        # classifier_fn =  'C:\\Users\\Avi\\PycharmProjects\\exML\\Machine-Learning-Ex\\plots\\20170926_082631\\classifiers\\plots\\20170926_082631\\best\\best_Ensemble_20.classifier'
        classifier_fn = 'C:\\Users\\Avi\\PycharmProjects\\exML\\Machine-Learning-Ex\\plots\\looks_like_good_classifier.pickle'
        classifier = File.get_pickle(classifier_fn )
        classification  = classifier.classify(data)
        correct_answers = (classification == data.ys) * 1
        accuracy = sum(correct_answers)*1.0/ len(correct_answers)
        bad_indices = [str(i) for i,v in enumerate(correct_answers) if v == 0]
        logger.info("\n\n{0}%\n\nGot wrong classification for indices (zero seed):\n{1}".format(accuracy,"\n".join(bad_indices)))


        pass
    else:
        return
        from run_scripts.final_project_main import final_project_main
        final_project_main()

    return
    n_features = 4
    run_best_n_fitures(n=n_features, classifier=bi_rf())

    run_best_n_fitures(n=n_features, classifier=bi_lr())
    return
    run_best_n_fitures(n=n_features , classifier=lc_coursera())
    return
    run_best_n_fitures(n=n_features ,classifier=GaussianDiscriminantAnalysis())
    return
    # LogisticClassifier           -----------------------------------------------
    logc = lc()
    run_classifier(logc)

    # GaussianGenerativeClassifier -----------------------------------------------
    downloaded_gc = GaussianDiscriminantAnalysis()
    run_classifier(downloaded_gc)

    # GaussianGenerativeClassifier -----------------------------------------------
    # gc = GaussianGenerativeClassifier()
    # run_classifier(gc)

    # Coursera -----------------------------------------------
    logc_coursera = lc_coursera()
    run_classifier(logc_coursera)

    str()
    str()


if __name__ == "__main__":
    # BenchmarkSVM.run()
    main()
