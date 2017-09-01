from Classifiers.DataLoaders.CancerDataLoader import CancerDataLoader
from Classifiers.DataLoaders.CreditDataLoader import CreditDataLoader
from Classifiers.DataLoaders.final_project_data_loader import FinalProjectDataLoader


def get_default_data_loader():
    return FinalProjectDataLoader()
    return CreditDataLoader()
    return CancerDataLoader()
