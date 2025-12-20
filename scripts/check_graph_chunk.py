
import torch
import os

X_PATH = "../graph_emb/task_3_x.pt"
x = torch.load(X_PATH, map_location="cpu")

num_total = x.shape[0]
num_nonzero = int((x.abs().sum(dim=1) > 0).sum())

print(f"Total nodes: {num_total}")
print(f"Nodes with embeddings: {num_nonzero}")
print(f"Percent complete: {num_nonzero/num_total:.2%}")
