
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv

class GNNEncoder(nn.Module):
    def __init__(self, num_nodes, emb_dim, model_type="sage"):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        if model_type == "sage":
            self.conv1 = SAGEConv(emb_dim, emb_dim)
            self.conv2 = SAGEConv(emb_dim, emb_dim)
        elif model_type == "gat":
            self.conv1 = GATConv(emb_dim, emb_dim // 2, heads=2)
            self.conv2 = GATConv(emb_dim, emb_dim // 2, heads=2)

    def forward(self, node_ids, edge_index):
        x = self.embedding(node_ids)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
