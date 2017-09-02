from Classifiers.Builtins.abstract_builtin_classifier import AbstractBuiltinClassifier
from Classifiers.DataLoaders.Utils import get_default_data_loader
from Classifiers import rootLogger as logger


def final_project_main():
    clfs = AbstractBuiltinClassifier.get_all_classifiers()
    data_loader = get_default_data_loader()
    data = data_loader.load()
    for clf in clfs:
        clf .set_data(data)

    results = {}
    fails = {}
    for classifier in clfs:
        try:
            logger.info("Starting to classify using {0}')".format(classifier))
            classifier.train(training_set_size_percentage = 0.8)
            score = classifier.score
            results[classifier] = score
        except Exception as ex:
            fails[classifier] = ex

    msg_fails = "\n".join(["{0}: \t\t\t{1}".format(c, fails[c]) for c in fails.keys()])
    res = sorted([(clf,score) for clf,score in results.items()], key=lambda tpl: tpl[1].f_measure, reverse=True)
    msg_results = "\n".join(["{0} ;  ({1})".format(score, clf) for clf,score in res])
    logger.info("\n\nFails:\n{0}".format(msg_fails))
    logger.info("\n\nScors:\n{0}".format(msg_results))
    7


