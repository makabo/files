# %%
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# %%
# Add edges
edges = [
('A', 'B'),
('B', 'A'),
('A', 'C'),
('B', 'C'),
('C', 'A'),
('C', 'D'),
('D', 'C')
]
G.add_edges_from(edges)

# %%
# 1. Density
density = nx.density(G)
print("Density:", round(density, 3))

# %%
# 2. Degree Measures
print("\nDegree Measures:")
for node in G.nodes():
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    total_deg = in_deg + out_deg
    
    print(f"{node}: In-Degree={in_deg}, "
          f"Out-Degree={out_deg}, "
          f"Total Degree={total_deg}")

# %%
# 3. Reciprocity
reciprocity = nx.reciprocity(G)
print("\nReciprocity:", round(reciprocity, 3))

# %%
# 4. Transitivity
# Convert to undirected graph for transitivity
transitivity = nx.transitivity(G.to_undirected())
print("Transitivity:", round(transitivity, 3))

# %%
# 5. Degree Centralization (undirected, simple graph — reciprocal edges collapsed)
G_simple = nx.Graph(G)                                    # merges reciprocal directed edges into one
G_simple.remove_edges_from(nx.selfloop_edges(G_simple))    # safety: strip any self-loops

degrees = dict(G_simple.degree())
max_degree = max(degrees.values())
numerator = sum(max_degree - d for d in degrees.values())
n = len(G_simple.nodes())

if n > 2:
    centralization = numerator / ((n - 1) * (n - 2))
else:
    centralization = 0

print("Degree Centralization:", round(centralization, 3))

# %%
# 6. Clustering Coefficient
clustering = nx.clustering(G.to_undirected())

print("\nNode Clustering Coefficients:")
for node, coeff in clustering.items():
    print(f"{node}: {round(coeff, 3)}")

avg_clustering = nx.average_clustering(G.to_undirected())
print("\nAverage Clustering Coefficient:",
      round(avg_clustering, 3))

# %%



