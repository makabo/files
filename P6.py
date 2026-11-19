# %%
#Create and display the network
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([('A', 'C'), ('B', 'C'),('C', 'D'), ('C', 'E')])

nx.draw(G, with_labels=True, node_color='lightblue',
node_size=1500, arrows=True)
plt.show()

# %%
for u in G.nodes():
    for v in G.nodes():
        if u < v:
            if (set(G.predecessors(u)) == set(G.predecessors(v))
                and set(G.successors(u)) == set(G.successors(v))):
                print(u, "and", v, "are structurally equivalent")

# %%
from networkx.algorithms.isomorphism import DiGraphMatcher

matcher = DiGraphMatcher(G, G)

for mapping in matcher.isomorphisms_iter():
    for u, v in mapping.items():
        if u < v:
            print(u, "and", v, "are automorphically equivalent")



