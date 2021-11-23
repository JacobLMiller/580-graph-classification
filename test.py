import pickle
from collections import namedtuple

feature = namedtuple("feature", ['label',
                                'type',
                                'stress',
                                'neighbor',
                                'edge_crossings',
                                'angular_resolution',
                                'avg_edge_length',
                                'V',
                                'E'])

with open('data/test_collection.pkl', 'rb') as myfile:
    myobj = pickle.load(myfile)

print(myobj)
