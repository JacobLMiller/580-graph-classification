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

from noises import getXandY
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation_script import feature

"""
feature = namedtuple("feature", ['label',
                                'type',
                                'stress',
                                'neighbor',
                                'edge_crossings',
                                'angular_resolution',
                                'avg_edge_length',
                                'V',
                                'E',
                                'degree',
                                'degree_std',
                                'clusters',
                                'cluster_std'
                                ])
"""

X,y = getXandY('data/training2.pkl')
num_features = [9,10,11,12]

#X = np.delete(X,2,axis=1)
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

#X = np.hstack((cats, X))



print("----------------------")



# cats = arr_features[:,1]
# onehot = OneHotEncoder(sparse=False)
# y = onehot.fit_transform(cats.reshape(-1,1))

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

with open('data/test2.pkl', 'rb') as myfile:
        Test = pickle.load(myfile)

features = []
for i in range(len(Test)):
    features.append(Test[i][2])


arr_features = [list(features[i]) for i in range(len(features))]


arr_features = np.asarray(arr_features)

cats = arr_features[:,1]
onehot = OneHotEncoder(sparse=False)
cats = onehot.fit_transform(cats.reshape(-1,1))


testX = arr_features[:,num_features]
#X = np.delete(X,2,axis=1)
# scaler = preprocessing.StandardScaler().fit(testX)
#
#
# testX = scaler.transform(testX)
testy = arr_features[:,1]

#Checks if best_model has been written or not
if not os.path.exists('data/tuned_parameters_MLPClassifier.pkl'):
    alpha = np.linspace(0.000001,0.001,100)
    learning_rate_init = np.linspace(0.0001,0.01,100)
    power_t = np.linspace(0,1,100)
    momentum = np.linspace(0,1,100)
    validation_fraction = np.linspace(0,0.5,100)
    beta_1 = np.linspace(0,0.9999999,100)
    beta_2 = np.linspace(0,0.9999999,100)
    epsilon = np.linspace(1e-10,1e-6,100)

    choose = np.random.choice

    best_model, best_score = None,0

    for i in range(5000):
        params = {
                    'alpha': choose(alpha),
                    'learning_rate_init': choose(learning_rate_init),
                    'power_t': choose(power_t),
                    'momentum': choose(momentum),
                    'validation_fraction': choose(validation_fraction),
                    'beta_1': choose(beta_1),
                    'beta_2': choose(beta_2),
                    'epsilon': choose(epsilon)
                }

        clf = MLPClassifier(hidden_layer_sizes=(100,100),
                    activation='relu',
                    solver='adam',
                    alpha=params['alpha'],
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=params['learning_rate_init'],
                    power_t=params['power_t'],
                    max_iter=500,
                    shuffle=True,
                    random_state=None,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=params['momentum'],
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=params['validation_fraction'],
                    beta_1=params['beta_1'],
                    beta_2=params['beta_2'],
                    epsilon=params['epsilon'],
                    n_iter_no_change=10,
                    max_fun=15000).fit(X,y)

        if clf.score(testX,testy) > best_score :
            best_score = clf.score(testX,testy)
            best_model = params
            print("Updated best_score to ", best_score)

    #Saves the best model's parameters
    with open('data/tuned_parameters_'+ clf.__class__.__name__ +'.pkl', 'wb') as f:
        pickle.dump(clf.get_params(), f)
else:
    #Load best parameters
    with open('data/tuned_parameters_MLPClassifier.pkl', 'rb') as f:
        params = pickle.load(f)
        clf = MLPClassifier(**params).fit(X,y)

#print(clf)
##########################################################################
# import matplotlib.pyplot as plt
# mytree = DecisionTreeClassifier(ccp_alpha=0.02).fit(X, y)
# from sklearn import tree

# plt.plot(np.arange(len(clf.loss_curve_)),clf.loss_curve_)
# plt.ylabel('Loss function value')
# plt.xlabel("Iteration")
# plt.show()
# plt.clf()
# tree.plot_tree(mytree)
# plt.show()



# cats = arr_features[:,1]
#
# onehot = OneHotEncoder(sparse=False)
# y = onehot.fit_transform(cats.reshape(-1,1))



print("Classification score",clf.score(testX,testy))
#D = clf.predict(X).reshape(-1,1) == y.reshape(-1,1)
D = clf.predict(X).reshape(-1,1)
count = 0
for i in range(D.shape[0]):
    if D[i][0] == '0':
        count += 1
print("Classified prob of 0",count/D.shape[0])




y = y.reshape(-1,1)
count = 0
for i in range(y.shape[0]):
    if y[i][0] == '0':
        count += 1



def k_fold_cross_evaluation(clf, k, X, y):
    #Grabs the f1 score for each class
    y = y.flatten()
    kf = KFold(n_splits=k,shuffle=True)

    kf.get_n_splits(X)

    fscore = 0
    for train_i, test_i in kf.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        # if len(fscore) == 0:
        #     fscore = f1_score(y_test, pred, average='weighted')
        # else:
        fscore += f1_score(y_test, pred, average='weighted')
    fscore /= k

    return fscore
print("Neural Network f1-scores per class")
print(k_fold_cross_evaluation(clf, 10, X,y))
#print("True prob of 0",count/y.shape[0])
#print(y)


# def test_noise():
#      with open('data/tuned_parameters_MLPClassifier.pkl', 'rb') as f:
#         params = pickle.load(f)
#         clf = MLPClassifier(**params)
