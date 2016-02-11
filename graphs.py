import networkx as nx
import matplotlib.pyplot as plt
import pylab

G = nx.DiGraph()

spanTree = [[0,  7], [ 0,  2], [ 1,  7], [ 7,  8], [ 1,  6], [ 1,  4], [4,  5], [4,  3]]
spanTree = [(edge[0], edge[1]) for edge in spanTree]
print(spanTree)
nodes = list(range(9))

G.add_nodes_from(nodes)
G.add_edges_from(spanTree)

pos = nx.spectral_layout(G)
#pos = nx.random_layout(G)

nx.draw(G, pos, node_size=1500, node_color = ['r','r','b','b','b','b','b','b','b'], with_labels=True)#,)#, with_labels=True)
#nx.draw_networkx_labels(G,pos, nodes)
# plt.show()
plt.show()