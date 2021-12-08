from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import neighbors, datasets
import graph_tool.all as gt
import numpy as np
import pickle
import sys
import os
from sklearn import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation_script import feature

def load_paths():
    path = 'data/noise'
    noisy_tests = []
    for filename in os.listdir(path):
        noisy_tests.append((int(filename.split("t")[2].split('.')[0]),filename))

    noisy_tests.sort()
    return [noisy_tests[i][1] for i in range(len(noisy_tests))]

def getXandY(fname):
    with open(fname, 'rb') as myfile:
        Training = pickle.load(myfile)

    num_features = [2,3,4,5,6]
    features = []
    for i in range(len(Training)):
        if isinstance(Training[i],int):
            break
        features.append(Training[i][2])

    arr_features = [list(features[i]) for i in range(len(features))]

    arr_features = np.asarray(arr_features)

    X = arr_features[:,num_features]
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    y = arr_features[:,0]
    return X,y

if __name__ == "__main__":
    X,y = getXandY('data/training2.pkl')
    with open('data/tuned_parameters_MLPClassifier.pkl', 'rb') as f:
        params = pickle.load(f)
        clf = MLPClassifier(**params).fit(X,y)

    testpath = load_paths()
    path = 'data/noise/'
    scores = []
    for test in testpath:
        with open(path+test,'rb') as f:
            test_data = pickle.load(f)
        testx,testy = getXandY(path+test)
        scores.append(clf.score(testx,testy))
