import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#Create the two-mode network
people = ['P1','P2','P3','P4','P5','P6']
groups = ['G1','G2','G3']

edges = [
    ('P1','G1'), ('P1','G2'),
    ('P2','G1'), ('P2','G2'),
    ('P3','G2'), ('P3','G3'),
    ('P4','G3'),
    ('P5','G3'),
    ('P6','G1')
]

B = nx.Graph()

B.add_nodes_from(people, bipartite=0)
B.add_nodes_from(groups, bipartite=1)
B.add_edges_from(edges)

#Calculate number of ties
for node in B.nodes():
    print(node, ":", B.degree(node), "ties")

#Identify core and peripheral nodes
core = []
periphery = []

for node in B.nodes():
    if B.degree(node) >= 2:
        core.append(node)
    else:
        periphery.append(node)

print("Core:", core)
print("Periphery:", periphery)

#Visualize the network
pos = {}

for i, p in enumerate(people):
    pos[p] = (0, len(people)-i)

for i, g in enumerate(groups):
    pos[g] = (2, len(groups)-i)

colors = []

for node in B.nodes():
    if node in core:
        colors.append("red")
    else:
        colors.append("lightblue")

nx.draw(
    B,
    pos,
    with_labels=True,
    node_color=colors,
    node_size=1000,
    edge_color="gray"
)

plt.title("Two-Mode Core-Periphery Network")
plt.show()
G = nx.Graph()
G.add_edges_from([
 ('A', 'B'),
 ('A', 'C'),
 ('B', 'C'),
 ('C', 'D'),
 ('D', 'E')
])
nx.draw(G, with_labels=True, node_color='lightblue',  node_size=1500)
plt.show()

#Create the adjacency matrix
A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
print("Adjacency Matrix:")
print(A)

#Perform SVD analysis
U, S, VT = np.linalg.svd(A)
print("Matrix U:")
print(np.round(U, 3))
print("\nSingular Values:")
print(np.round(S, 3))
print("\nMatrix VT:")
print(np.round(VT, 3))

#Identify the important patterns
print("Singular Values in Descending Order:")
for i, value in enumerate(S):
 print("Pattern", i + 1, ":", round(value, 3))

#Reduce the network to two dimensions
# Select the first two singular vectors
reduced = U[:, :2] * S[:2]
for i, node in enumerate(sorted(G.nodes())):
  print(node, np.round(reduced[i], 3))

#Visualize the SVD representation
nodes = sorted(G.nodes())
plt.figure(figsize=(7, 5))
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, node in enumerate(nodes):
 plt.text(reduced[i, 0], reduced[i, 1], node)
plt.xlabel("SVD Dimension 1")
plt.ylabel("SVD Dimension 2")
plt.title("SVD Representation of the Network")
plt.grid(True)
plt.show()


