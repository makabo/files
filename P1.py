# %%
# Graph represented as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D']
}
# Displaying the number of nodes
num_nodes = len(graph)
print(f"Number of nodes in the graph: {num_nodes}")

# For an undirected graph, each edge is counted twice in an adjacency list.
# Dividing the sum of list lengths by 2 gives the actual number of unique edges.
num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
print(f"Number of edges in the graph: {num_edges}")

#Degree of each node
degrees = {node: len(neighbors) for node, neighbors in graph.items()}
print("Degree of each node:")
for node, degree in degrees.items():
    print(f"  Node {node}: {degree}")

#Node(s) with the lowest degree
min_degree = min(degrees.values())
min_nodes = [node for node, degree in degrees.items() if degree == min_degree]
print(f"\nNode(s) with the lowest degree ({min_degree}): {', '.join(min_nodes)}")

#Adjacency List Display
print("\nAdjacency List:")
for node, neighbors in graph.items():
    print(f"  {node}: {neighbors}")

#Adjacency Matrix Creation and Display
nodes = sorted(graph.keys())
matrix = []
for u in nodes:
    row = []
    for v in nodes:
        row.append(1 if v in graph[u] else 0)
    matrix.append(row)
print("\nAdjacency Matrix:")
print("   " + " ".join(nodes))
for i, row in enumerate(matrix):
    print(f"{nodes[i]}  " + " ".join(map(str, row)))

# Example: Displaying the graph structure
for node, neighbors in graph.items():
    print(f"Node {node} is connected to: {', '.join(neighbors)}")

# %%



