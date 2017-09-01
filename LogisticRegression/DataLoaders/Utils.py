from LogisticRegression.DataLoaders.CancerDataLoader import CancerDataLoader
from LogisticRegression.DataLoaders.CreditDataLoader import CreditDataLoader
from LogisticRegression.DataLoaders.final_project_data_loader import FinalProjectDataLoader


def get_default_data_loader():
    return FinalProjectDataLoader()
    return CreditDataLoader()
    return CancerDataLoader()
