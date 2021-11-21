import networkx as nx
import math
import random
import numpy as np
import pickle
import graph_tool.all as gt
from SGD_MDS import all_pairs_shortest_path

def euclid_dist(x1,x2):
    x = x2[0]-x1[0]
    y = x2[1]-x1[1]
    return pow(x*x+y*y,0.5)

def get_k_nearest(d_row,k):
    return np.argpartition(d_row,k)[:k]


def k_nearest_embedded(X,k_theory):
    sum = 0
    dist_mat = np.zeros([len(X),len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                dist_mat[i][j] = euclid_dist(X[i],X[j])
            else:
                dist_mat[i][j] = 100000
    k_embedded = [np.zeros(k_theory[i].shape) for i in range(len(k_theory))]

    for i in range(len(dist_mat)):
        k = len(k_theory[i])
        k_embedded[i] = np.argpartition(dist_mat[i],k)[:k]

    for i in range(len(X)):
        count_intersect = 0
        count_union = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)
    return sum/len(G.nodes())

def calc_stress(X,d):
    stress = 0
    for i in range(len(d)):
        for j in range(i):
            stress += pow((euclid_dist(X[i],X[j])-d[i][j])/d[i][j],2)
    return stress

def calc_distortion(X,d):
    distortion = 0
    for i in range(len(d)):
        for j in range(i):
            distortion += abs((euclid_dist(X[i],X[j])-d[i][j]))/d[i][j]
    return (1/choose(len(d),2))*distortion

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

with open('tsnet_block_grid.pkl', 'rb') as myfile:
    myObj = pickle.load(myfile)

X = myObj[1]

G = nx.drawing.nx_agraph.read_dot('tsnet_block_grid.dot')
d = np.asarray(all_pairs_shortest_path(G))


for i in range(len(d)):
    for j in range(len(d)):
        if i == j:
            d[i][j] = 100000



rg = 2

k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]
print(k_theory)
#for i in range(len(k_theory)):
#    k_theory[i] = k_theory[i][0]
print(k_theory)
neighborhood_pres = k_nearest_embedded(X,k_theory)

print(calc_distortion(X,d))
print(neighborhood_pres)
