from Classifiers.DataLoaders.AbstractDataLoader import AbstractDataLoader
from Classifiers.DataLoaders.CancerDataLoader import CancerDataLoader
from Classifiers.DataLoaders.CreditDataLoader import CreditDataLoader
from Classifiers.DataLoaders.final_project_data_loader import FinalProjectDataLoader


def get_default_data_loader(set_mask=True):
    # type: () -> AbstractDataLoader
    return FinalProjectDataLoader(set_mask=True)
    return CreditDataLoader()
    return CancerDataLoader()
