import pickle

with open('data/test_collection.pkl', 'rb') as myfile:
    myobj = pickle.load(myfile)

print(myobj)
