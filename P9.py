# %%
import networkx as nx
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
print("Periphery", periphery)

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

nx.draw(B,pos,with_labels=True,node_color=colors,node_size=1000,edge_color="gray")

plt.title("Two-Mode Core-Periphery Network")
plt.show()



