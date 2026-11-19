# %%
#A faction is a group of nodes that have relatively similar patterns of connections.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.cluster import SpectralClustering

# %%
#Create a two-mode network: 8 people and 6 events.
#This is called two-mode because there are two node types:
people = [
    "Alice", "Bob", "Charlie", "David",
    "Eve", "Frank", "Grace", "Henry"
]

events = [
    "Event_A", "Event_B", "Event_C",
    "Event_D", "Event_E", "Event_F"
]

data = np.array([
    [1, 1, 1, 0, 0, 0], # Alice
    [1, 1, 1, 0, 0, 0], # Bob
    [1, 1, 0, 0, 0, 0], # Charlie
    [0, 0, 1, 1, 1, 0], # David
    [0, 0, 1, 1, 1, 0], # Eve
    [0, 0, 0, 1, 1, 1], # Frank
    [0, 0, 0, 0, 1, 1], # Grace
    [0, 0, 0, 0, 1, 1] # Henry
])

matrix = pd.DataFrame(
    data,
    index=people,
    columns=events
) 
print(matrix)

#Visualize the two-mode network
G = nx.Graph()

# Add people
G.add_nodes_from(people, bipartite=0)

# Add events
G.add_nodes_from(events, bipartite=1)

# Add edges
for person in people:
    for event in events:
        if matrix.loc[person, event] == 1:
            G.add_edge(person, event)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

plt.figure(figsize=(12, 7))

person_pos = {
    person: (i, 1)
    for i, person in enumerate(people)
}

event_pos = {
    event: (i, 0)
    for i, event in enumerate(events)
}

pos = {**person_pos, **event_pos}

nx.draw_networkx_nodes(G,pos,nodelist=people,node_color="skyblue",node_size=1500,label="People"
)

nx.draw_networkx_nodes(G,pos,nodelist=events,node_color="orange",node_size=1500,label="Events"
)

nx.draw_networkx_edges(G,pos,width=2,alpha=0.6
)

nx.draw_networkx_labels(G,pos,font_size=9
)

plt.title("Two-Mode Bipartite Network")
plt.legend(markerscale=0.4, labelspacing=1.5)
plt.axis("off")
plt.show()




