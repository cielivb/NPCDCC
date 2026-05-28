""" Exploratory Data Analysis

Get number of unique nodes (neurons) in (proofread) Drosophila connectome.

Findings: 139,255 neurons, no duplicates.

"""

import numpy as np

root_ids = np.load("../data/proofread_root_ids_783.npy")
print(f"Total number of nodes in Drosophila connectome: {len(root_ids)}")
num_unique_root_ids = len(np.unique(root_ids))
print(f"Number of unique nodes in Drosophila connectome: {num_unique_root_ids}")