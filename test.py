import pickle
from collections import namedtuple
from generation_script import feature


with open('data/test1.pkl', 'rb') as myfile:
    myobj = pickle.load(myfile)

print(myobj[38])
