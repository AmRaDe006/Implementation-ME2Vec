import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the embeddings from the first file
emb_file1 = 'BS/ppd_eICU_before_weighted.emd'
with open(emb_file1, 'r') as f:
    num_nodes, emb_size = map(int, f.readline().split())
    emb_dict1 = {}
    for line in f:
        node, emb = line.rstrip().split(' ', 1)
        emb_dict1[node] = np.fromstring(emb, sep=' ')

# Load the embeddings from the second file
emb_file2 = 'BS/ppd_eICU_after_weighted.emd'
with open(emb_file2, 'r') as f:
    num_nodes, emb_size = map(int, f.readline().split())
    emb_dict2 = {}
    for line in f:
        node, emb = line.rstrip().split(' ', 1)
        emb_dict2[node] = np.fromstring(emb, sep=' ')

# Get the set of nodes that are present in both files
nodes = set(emb_dict1.keys()) & set(emb_dict2.keys())

# Calculate the cosine similarity between the embeddings for each node
sims = []
for node in nodes:
    emb1 = emb_dict1[node]
    emb2 = emb_dict2[node]
    sim = cosine_similarity([emb1], [emb2])[0][0] # type: ignore
    sims.append(sim)

# Calculate the average cosine similarity across all nodes
avg_sim = np.mean(sims)

print('Average cosine similarity between embeddings: {:.4f}'.format(avg_sim))
