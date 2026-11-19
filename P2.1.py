# %%
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# Create one-mode dataset
data = {
 "Source": ["A", "A", "B", "C", "D"],
 "Target": ["B", "C", "C", "D", "E"]
}
df = pd.DataFrame(data)
print("One-mode network dataset:")
display(df)

G = nx.from_pandas_edgelist(
 df, source="Source", target="Target"
)
nx.draw(
 G, with_labels=True,
 node_color="lightblue",
 node_size=1500
)
plt.title("One-Mode Friendship Network")
plt.show()
data = {
 "Student": ["A", "A", "B", "B", "C", "C"],
 "Course": [
 "Python", "AI",
 "Python", "DBMS",
 "AI", "DBMS"
 ]
}
df2 = pd.DataFrame(data)
print("Two-mode dataset:")
display(df2)


B = nx.Graph()
students = df2["Student"].unique()
courses = df2["Course"].unique()
B.add_nodes_from(students, bipartite=0)
B.add_nodes_from(courses, bipartite=1)
B.add_edges_from(
 zip(df2["Student"], df2["Course"]) )
pos = nx.bipartite_layout(B, students)
nx.draw(
 B, pos, with_labels=True,
 node_color=[
 "lightblue" if n in students else "lightgreen"  for n in B.nodes()
 ],
 node_size=1800
)
plt.title("Two-Mode Student-Course Network")
plt.show()



