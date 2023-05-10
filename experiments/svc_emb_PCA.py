import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# with open('node2vec/emb/ppd_eICU.emd', 'r') as f:
#     # The first line contains the number of nodes and the dimensionality of the embeddings
#     first_line = f.readline()
#     num_nodes = int(first_line.split()[0])

# print('Number of nodes:', num_nodes)

# Load the .emd file and extract the embeddings and node IDs
data = np.loadtxt('node2vec/emb/ppd_eICU.emd', skiprows=1, dtype=str)
# ids = data[:, 0]
embeddings = data[:, 1:].astype(float)

# Perform PCA to reduce the embeddings to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Create a scatter plot of the reduced-dimensional embeddings with node IDs as labels
fig, ax = plt.subplots()
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
# for i, id in enumerate(ids):
#     ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], id, fontsize=8)
plt.show()

