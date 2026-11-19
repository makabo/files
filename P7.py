# %%
"""
Sociograms for an affiliation (persons x committees) dataset.

Produces:
  1. One-mode persons-by-persons network   (linked by shared committees)
  2. One-mode committee-by-committee network (linked by shared members)
  3. Two-mode (bipartite) persons-and-committees network
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite


# SAMPLE DATA (persons x committees incidence matrix)
memberships = {
    "Alice":  ["Finance", "Budget", "Audit"],
    "Bob":    ["Finance", "Hiring"],
    "Carla":  ["Budget", "Audit", "Ethics"],
    "David":  ["Hiring", "Ethics"],
    "Elena":  ["Finance", "Budget"],
    "Farid":  ["Audit", "Ethics", "Safety"],
    "Grace":  ["Safety", "Hiring"],
    "Hiro":   ["Safety"],
}
persons = list(memberships)
committees = sorted({c for cs in memberships.values() for c in cs})
incidence = pd.DataFrame(0, index=persons, columns=committees)
for p, cs in memberships.items():
    incidence.loc[p, cs] = 1

# BUILD NETWORKS
M = incidence.values

# One-mode projections: M M^T (persons) and M^T M (committees)
P_arr, C_arr = M @ M.T, M.T @ M          # shared committees / shared members
np.fill_diagonal(P_arr, 0)                # drop self-ties
np.fill_diagonal(C_arr, 0)
P = pd.DataFrame(P_arr, index=persons, columns=persons)
C = pd.DataFrame(C_arr, index=committees, columns=committees)

def one_mode_graph(adj: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_adjacency(adj)   # edge weight = matrix value
    G.remove_edges_from([(u, v) for u, v, w in G.edges(data="weight") if w == 0])
    return G

G_persons = one_mode_graph(P)
G_committees = one_mode_graph(C)

# Two-mode (bipartite) graph
G_two = nx.Graph()
G_two.add_nodes_from(persons, bipartite=0)
G_two.add_nodes_from(committees, bipartite=1)
for p in persons:
    for c in committees:
        if incidence.loc[p, c]:
            G_two.add_edge(p, c)


# DRAWING HELPERS
PERSON_COLOR, COMMITTEE_COLOR = "#4C78A8", "#F58518"

def draw_one_mode(G, title, color, ax, seed=42):
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=0.9)
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    deg = dict(G.degree(weight="weight"))
    sizes = [300 + 120 * deg[n] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=1 + 2.5 * weights,
                           alpha=0.55, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=color,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, font_size=8,
        edge_labels={(u, v): int(G[u][v]["weight"]) for u, v in G.edges()},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

def draw_two_mode(G, ax):
    top = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    bottom = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
    pos = nx.bipartite_layout(G, top, align="horizontal", aspect_ratio=2.2)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, nodelist=top, ax=ax, node_shape="o",
                           node_size=900, node_color=PERSON_COLOR,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=bottom, ax=ax, node_shape="s",
                           node_size=1100, node_color=COMMITTEE_COLOR,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    ax.set_title("Two-mode network: persons (circles) x committees (squares)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

# PLOTS
if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    draw_one_mode(G_persons, "One-mode: persons x persons\n(edge = # shared committees)",
                  PERSON_COLOR, axes[0])
    draw_one_mode(G_committees, "One-mode: committees x committees\n(edge = # shared members)",
                  COMMITTEE_COLOR, axes[1])
    plt.tight_layout()
    plt.savefig("sociogram_one_mode.png", dpi=200)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    draw_two_mode(G_two, ax)
    plt.tight_layout()
    plt.savefig("sociogram_two_mode.png", dpi=200)

    # Quick descriptive stats
    print("Incidence matrix:\n", incidence, "\n")
    print("Persons x persons matrix:\n", P, "\n")
    print("Committees x committees matrix:\n", C, "\n")
    print("Degree centrality (persons network):")
    print(pd.Series(nx.degree_centrality(G_persons)).round(2).sort_values(ascending=False), "\n")
    print("Degree centrality (committee network):")
    print(pd.Series(nx.degree_centrality(G_committees)).round(2).sort_values(ascending=False))
    plt.show()




