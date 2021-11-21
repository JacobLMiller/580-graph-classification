import graph_tool.all as gt
import numpy as np

norm = lambda x: np.linalg.norm(x,ord=2)

def calc_stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(norm(X[i]-X[j])-d[i][j],2)
    return stress

def calc_neighborhood(X,d,rg = 2):
    def get_k_embedded(X,rg):
        dist_mat = [[norm(X[i]-X[j]) if for j in range(len(X))] for i in range(len(X))]
        return np.array([np.argpartition(dist_mat[i],rg)[:rg] for i in range(len(dist_mat))])

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]
    k_embedded = get_k_embedded(X,rg)

    sum = 0
    for i in range(len(X)):
        count_intersect = 0
        for j in range(len(k_theory[i])):
            if k_theory[i][j] in k_embedded[i]:
                count_intersect += 1
        sum += count_intersect/(len(k_theory[i])+len(k_embedded[i])-count_intersect)

    return sum/len(X)
