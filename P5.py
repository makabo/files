# %%
#Network 1 : Friendship Network
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# %%
#Network as a Matrix
# Network 1: Friendship network
nodes1 = ['A', 'B', 'C', 'D', 'E']

matrix1 = np.array([
[0, 1, 1, 0, 0],
[1, 0, 1, 1, 0],
[1, 1, 0, 0, 1],
[0, 1, 0, 0, 1],
[0, 0, 1, 1, 0]
])

matrix1_df = pd.DataFrame(matrix1,index=nodes1,columns=nodes1)

print("Network 1 - Adjacency Matrix")
print(matrix1_df)

# %%
#Network as an Edge List
edges1 = [
('A', 'B'),
('A', 'C'),
('B', 'C'),
('B', 'D'),
('C', 'E'),
('D', 'E')
]

edge_list1 = pd.DataFrame(
edges1,

columns=['Source', 'Target']
)

print("Network 1 - Edge List")
print(edge_list1)

# %%
# Network as a Sociogram
G1 = nx.Graph()
G1.add_nodes_from(nodes1)
G1.add_edges_from(edges1)

plt.figure(figsize=(7, 5))

nx.draw(G1,with_labels=True,node_color='skyblue',node_size=1800,edge_color='gray',font_size=12,font_weight='bold')

plt.title("Network 1: Friendship Sociogram")
plt.show()



