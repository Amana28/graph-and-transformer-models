import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
graphml_file = "/content/drive/MyDrive/Research/Graphs-and-Transformer-Models/ALPINE-Experiments/graph-and-transformer-models/data/simple_graph/10/path_graph.graphml"
save_dir = "/content/drive/MyDrive/Research/Graphs-and-Transformer-Models/ALPINE-Experiments/graph-and-transformer-models/data/simple_graph/10/matrices"

# Ensure the matrices directory exists
os.makedirs(save_dir, exist_ok=True)

# Load the GraphML file
G = nx.read_graphml(graphml_file)

# Convert to adjacency matrix
adj_matrix = nx.to_numpy_array(G, dtype=int)

# Plot the adjacency matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label="Edge Presence")
plt.title("Adjacency Matrix Heatmap")
plt.xlabel("Target Nodes")
plt.ylabel("Source Nodes")

# Add grid lines for better visualization
plt.xticks(np.arange(len(G.nodes)), G.nodes(), rotation=90)
plt.yticks(np.arange(len(G.nodes)), G.nodes())
plt.grid(False)  # Turn off default grid lines

# Save the heatmap in the matrices folder
save_path = os.path.join(save_dir, "adjacency_matrix_heatmap.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Heatmap saved at: {save_path}")
