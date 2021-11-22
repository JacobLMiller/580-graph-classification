from graph_tool import VertexPropertyMap
import graph_tool.all as gt
import numpy as np
from pprint import pprint
from collections import namedtuple
from shapely.geometry import LineString
from shapely.geometry import Point
norm = lambda x: np.linalg.norm(x,ord=2)

def calc_stress(X,d):
    """
    How well do the pairwise distances in the embedding match the theoretic distances?
    Closer to 0 is better.
    Stress = \sqrt (\sum (||X_i - X_j|| - d_ij)^2)
    """
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(norm(X[i]-X[j])-d[i][j],2)
    return pow(stress,0.5)

def calc_neighborhood(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    def get_k_embedded(X,rg):
        dist_mat = [[norm(X[i]-X[j]) if i != j else 10000 for j in range(len(X))] for i in range(len(X))]
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

def calc_edge_crossings(edges, node_poses):
    print("poses",node_poses[0])
    Node = namedtuple("vertex", "x y")

    node_poses = [Node(n[0],n[1]) for n in node_poses]
    edges = [(int(n1),int(n2)) for (n1,n2) in edges]
   

    lines = list()

    #
    for edge in edges:
        n1, n2 = edge
        lines.append(LineString([(node_poses[n1].x,node_poses[n1].y), (node_poses[n2].x,node_poses[n2].y)]))
    print("lines", len(lines))
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            point = line1.intersection(line2)
            #Make sure that intersection is not on boundary (happens if a node has out-degree >1)
            if (not point.is_empty) and (line1.contains(point) and line2.contains(point)):
                assert(isinstance(point, Point))
                intersections.append(point)
    

    return len(intersections)

def calc_angular_resolution(G,X):
    """
    Smallest angle formed by any two edges....
    TODO: is this correct?
    """
    atan2 = np.arctan2
    mymin = 10000
    for (i,j) in G.edges():
        x1 = X[int(i)]
        x2 = X[int(j)]
        mymin = min(abs(atan2(x2[1]-x1[1],x2[0]-x1[0])),mymin)
    return min

def calc_edge_lengths(E,X):
    """
    Returns the average edge length of edges in the drawing (should be close to 1).
    """
    return np.mean([norm(X[int(i)]-X[int(j)]) for (i,j) in E])
