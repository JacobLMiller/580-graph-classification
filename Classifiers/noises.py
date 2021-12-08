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

from decisiontree import k_fold_cross_evaluation

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

    num_features = [2,3,4,5,6,7,8,9,10,11,12]
    features = []
    for i in range(len(Training)):
        if isinstance(Training[i],int):
            break
        features.append(Training[i][2])

    arr_features = [list(features[i]) for i in range(len(features))]

    arr_features = np.asarray(arr_features)

    X = arr_features[:,num_features]

    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    scaler = preprocessing.Normalizer().fit(X)
    X = scaler.transform(X)


    cats = arr_features[:,1]
    onehot = OneHotEncoder(sparse=False)
    cats = onehot.fit_transform(cats.reshape(-1,1))
    X = np.hstack((cats, X))


    y = arr_features[:,1]
    return X,y

if __name__ == "__main__":
    X,y = getXandY('data/training2.pkl')
    with open('data/tuned_parameters_MLPClassifier.pkl', 'rb') as f:
        params = pickle.load(f)
        clf = MLPClassifier(**params).fit(X,y)
        dTree = DecisionTreeClassifier().fit(X,y)
        knn = neighbors.KNeighborsClassifier(5,weights='uniform').fit(X,y)
    classifiers = (clf,dTree,knn)

    testpath = load_paths()
    path = 'data/noise/'
    scores = {}
    f1scores = {}
    clf = classifiers[2]

    testx,testy = getXandY('data/test2.pkl')

    print(clf.score(testx,testy))

    # for clf in classifiers:
    #     name = clf.__class__.__name__
    #     scores[name] = []
    #     f1scores[name] = []
    #     for test in testpath:
    #         with open(path+test,'rb') as f:
    #             test_data = pickle.load(f)
    #         testx,testy = getXandY(path+test)
    #         scores[name].append(clf.score(testx,testy))
    #         f1scores[name].append(k_fold_cross_evaluation(clf,5,testx,testy))
    #     print(name,scores[name])
    #     print(name,f1scores[name])
    #
    # import matplotlib.pyplot as plt
    #
    # plt.suptitle("Accuracy scores")
    # for score in scores.keys():
    #     plt.plot(np.linspace(0,5,20), scores[score],label=score)
    # plt.xlabel('Standard deviation of gaussian noise')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.savefig('normalAccuracyscores.png')
    # plt.clf()
    #
    # plt.suptitle("F scores")
    # for score in f1scores.keys():
    #     plt.plot(np.linspace(0,5,20),f1scores[score],label=score)
    # plt.xlabel('Standard deviation of gaussian noise')
    # plt.ylabel('F-Score')
    # plt.legend()
    # plt.savefig('normalf1scores.png')
    # plt.clf()
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D
    # pca = PCA(n_components=2,svd_solver='full')
    # pos = pca.fit_transform(X)
    # print(y)
    # color = [None for i in range(len(y))]
    # for i in range(len(color)):
    #     color[i] = 'red' if y[i] == '0' else 'blue' if y[i] == '1' else 'green'
    #
    # plt.scatter(pos[:,0],pos[:,1],c=color)
    # plt.show()
    # plt.clf()
    pca = PCA(n_components=2,svd_solver='full')
    pos = pca.fit_transform(X)

    color = [None for i in range(len(y))]

    for i in range(len(color)):
        color[i] = 'red' if y[i] == 'lattice' else 'blue' if y[i] == 'triangulation' else 'green' if y[i] == 'tree' else 'orange' if y[i] == 'block' else 'purple'
    # for i in range(len(color)):
    #     color[i] = 'red' if y[i] == '0' else 'blue' if y[i] == '1' else 'green'


    plt.scatter(pos[:,0],pos[:,1],c=color)

    legend_elements = [Line2D([0], [0], color='b',linestyle="none",marker='o', label='Triangulation'),
                   Line2D([0], [0], marker='o',linestyle="none", color='red', label='Lattice'),
                   Line2D([0], [0], marker='o',linestyle="none", color='green', label='Tree'),
                   Line2D([0], [0], marker='o',linestyle="none", color='orange', label='Block'),
                   Line2D([0], [0], marker='o',linestyle="none", color='purple', label='Ring')]

    # legend_elements = [Line2D([0], [0], color='b',linestyle="none",marker='o', label='MDS'),
    #                Line2D([0], [0], marker='o',linestyle="none", color='red', label='Random'),
    #                Line2D([0], [0], marker='o',linestyle="none", color='green', label='t-sNET')]
    plt.suptitle("PCA of drawing features")
    plt.legend(handles=legend_elements)
    #plt.show()
    plt.savefig('onlydrawing.png')
