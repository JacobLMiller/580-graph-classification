import networkx as nx
import graph_tool.all as gt
import pickle
import numpy as np

from shapely.geometry import LineString

line = LineString([(0, 0), (1, 1)])
other = LineString([(0, 1), (1, 0)])
print(line.intersects(other))
