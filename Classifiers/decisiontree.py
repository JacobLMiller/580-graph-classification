from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors, datasets
import graph_tool.all as gt
import numpy as np
import pickle
import sys
import os
from sklearn import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation_script import feature

with open('data/training1.pkl', 'rb') as myfile:
    Training = pickle.load(myfile)


features = []
for i in range(len(Training)):
    features.append(Training[i][2])

arr_features = [list(features[i]) for i in range(len(features))]

arr_features = np.asarray(arr_features)

cats = arr_features[:,1]
onehot = OneHotEncoder(sparse=False)
cats = onehot.fit_transform(cats.reshape(-1,1))


X = arr_features[:,2:7]
#X = np.delete(X,2,axis=1)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X = np.hstack((cats, X))



print("----------------------")

y = arr_features[:,0]


#clf = DecisionTreeClassifier(ccp_alpha=0.02).fit(X, y)
#clf = neighbors.KNeighborsClassifier(5,weights='uniform').fit(X,y)
"""
MLPClassifier(hidden_layer_sizes=(100),
              activation='relu',
               *,
               solver='adam',
               alpha=0.0001,
               batch_size='auto',
               learning_rate='constant',
               learning_rate_init=0.001,
               power_t=0.5,
               max_iter=200,
               shuffle=True,
               random_state=None,
               tol=0.0001,
               verbose=False,
               warm_start=False,
               momentum=0.9,
               nesterovs_momentum=True,
               early_stopping=False,
               validation_fraction=0.1,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-08,
               n_iter_no_change=10,
               max_fun=15000)
"""

clf = MLPClassifier(hidden_layer_sizes=(10),verbose=True).fit(X,y)
#print(clf)
##########################################################################
with open('data/test1.pkl', 'rb') as myfile:
    Test = pickle.load(myfile)

#

features = []
for i in range(len(Test)):
    features.append(Test[i][2])


arr_features = [list(features[i]) for i in range(len(features))]


arr_features = np.asarray(arr_features)

cats = arr_features[:,1]
onehot = OneHotEncoder(sparse=False)
cats = onehot.fit_transform(cats.reshape(-1,1))


X = arr_features[:,2:7]
#X = np.delete(X,2,axis=1)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X = np.hstack((cats, X))


y = arr_features[:,0]

print(clf.score(X,y))
D = clf.predict(X).reshape(-1,1) != y.reshape(-1,1)
