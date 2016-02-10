import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()

spanTree = [[ 4.,  5.], [ 3.,  4.], [ 0.,  7.], [ 1.,  7.], [ 1.,  6.], [ 1.,  4.], [ 7.,  8.], [ 0.,  2.]]

for edge in spanTree:
    G.add_edge(edge[0], edge[1])    
    

nx.draw(G)
