from LogisticRegression.DataLoaders.CancerDataLoader import CancerDataLoader
from LogisticRegression.DataLoaders.CreditDataLoader import CreditDataLoader


def get_default_data_loader():
    return CreditDataLoader()
    # return CancerDataLoader()