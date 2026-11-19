# %%
import networkx as nx
import matplotlib.pyplot as plt

# %%
#Create the Network
G = nx.Graph()
edges = [
('A','B'),
('A','C'),
('B','D'),
('C','D'),
('C','E'),
('D','F'),
('E','G'),
('F','G'),
('G','H')
]

G.add_edges_from(edges)

print("Nodes:", G.nodes())
print("Edges:", G.edges())

# %%
#Visualize the Network
plt.figure(figsize=(8,6))

pos = nx.spring_layout(G, seed=42)

nx.draw(G,pos,with_labels=True,node_size=2500,node_color='lightblue',
font_size=12,
font_weight='bold'
)

plt.title("Original Network")
plt.show()

# %%
#(i) Length of the Shortest Path
#Find the shortest path from node A to node G.
source_node = 'A'
target_node = 'G'

shortest_path = nx.shortest_path(G,source=source_node,target=target_node)
path_length = nx.shortest_path_length(G,source=source_node,target=target_node)

print("Shortest Path:", shortest_path)
print("Shortest Path Length:", path_length)

# %%
#(ii) Density of the Graph
density = nx.density(G)
print("Graph Density =", density)

N = G.number_of_nodes()
E = G.number_of_edges()
density_manual = (2 * E) / (N * (N - 1))
print("Number of Nodes =", N)
print("Number of Edges =", E)
print("Density =", density_manual)

# %%
#(iii) Draw Egocentric Network of Node G
#An Egocentric Network consists of: Ego node (G), All nodes directly connected to G , All links among these nodes

ego_G = nx.ego_graph(G, 'G')
print("Nodes in Ego Network:")
print(list(ego_G.nodes()))
print("\nEdges in Ego Network:")
print(list(ego_G.edges()))

#Draw Egocentric Network
plt.figure(figsize=(6,6))
pos = nx.circular_layout(ego_G)
nx.draw(
ego_G,
pos,
with_labels=True,
node_size=3000,
node_color='orange',
edge_color='black',
width=2,
font_size=12,
font_weight='bold'
)

plt.title("Egocentric Network of Node G")
plt.axis('off')
plt.show()

# %%



