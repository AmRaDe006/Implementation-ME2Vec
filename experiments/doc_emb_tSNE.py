import pickle

with open('saved_data/doc_emb_prime.pkl', 'rb') as f:
    embeddings = pickle.load(f)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

import matplotlib.pyplot as plt
import numpy as np

labels = ["label{}".format(n) for n in range(1, 50)]
label_map = {label: i for i, label in enumerate(set(labels))}
c = np.array([label_map[label] for label in labels])

# Define a colormap
cmap = plt.cm.get_cmap('viridis', len(label_map)) # type: ignore

# Get the colors from the colormap
colors = cmap(c)

# Visualize the embeddings with colors based on labels
# plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors, s=10)
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=10)

# Show the plot
plt.show()
