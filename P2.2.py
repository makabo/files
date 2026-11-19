# %%
import pandas as pd
import numpy as np
import networkx as nx
from IPython.display import display
nodes = ['A', 'B', 'C', 'D', 'E']
matrix = np.array([
 [0, 1, 1, 0, 0],
 [1, 0, 1, 0, 0],
 [1, 1, 0, 1, 0],
 [0, 0, 1, 0, 1],
 [0, 0, 0, 1, 0]
])
df = pd.DataFrame(matrix, index=nodes, columns=nodes)
print("Adjacency Matrix:")

G = nx.from_numpy_array(matrix)
G = nx.relabel_nodes(G, dict(enumerate(nodes)))
print("Edge List:")
print(list(G.edges()))

edge_list = [
 ('A', 'B'),
 ('A', 'C'),
 ('B', 'C'),
 ('C', 'D'),
 ('D', 'E')
]
G2 = nx.Graph()
G2.add_nodes_from(nodes)
G2.add_edges_from(edge_list)
new_matrix = nx.to_numpy_array(G2, nodelist=nodes, dtype=int)
print("Adjacency Matrix from Edge List:")
display(pd.DataFrame(new_matrix, index=nodes, columns=nodes))

transpose_matrix = matrix.T
print("Transpose Matrix:")
display(pd.DataFrame(transpose_matrix, index=nodes, columns=nodes))



degree = matrix.sum(axis=1)
print("Degree of Each Node:")
for i in range(len(nodes)):
 print(nodes[i], ":", degree[i])
valued_matrix = np.array([
 [0, 5, 3, 0, 0],
 [5, 0, 4, 0, 0],
 [3, 4, 0, 6, 0],
 [0, 0, 6, 0, 2],
 [0, 0, 0, 2, 0]
])
print("Valued Adjacency Matrix:")
display(pd.DataFrame(valued_matrix, index=nodes,columns=nodes))


square_matrix = matrix @ matrix
print("Square of Adjacency Matrix:")
display(pd.DataFrame(square_matrix, index=nodes, columns=nodes))



