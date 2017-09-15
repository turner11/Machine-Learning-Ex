import pickle

from cloudpickle import CloudPickler


def to_cloud_pickle(path,obj):
    with open(path, 'w') as f:
        pickler = CloudPickler(f)
        f.flush()
        return pickler.dump(obj)



def from_cloud_pickle(path):
    with open(path, 'r') as f:
        return pickle.load(f)