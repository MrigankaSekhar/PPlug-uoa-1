
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch.nn.functional as F

class GraphEncoder(nn.Module):
    """
    Trainable GraphSAGE Encoder.
    Can be used either standalone for pretraining/inference or embedded inside a bigger model (e.g. PersonalLLM_Slim).
    """
    def __init__(self, num_nodes, emb_dim, hidden_dim=256, alpha=0.2, dropout=0.3, use_norm=True):
        super().__init__()
        self.alpha = alpha
        self.use_norm = use_norm
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_ids_or_x, edge_index, edge_weight=None):
        """
        If tensor is long dtype → treat as node IDs.
        If float tensor → treat as precomputed x features.
        """
        x = self.node_emb(node_ids_or_x) if node_ids_or_x.dtype == torch.long else node_ids_or_x
        try:
            h = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        except TypeError:
            h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        try:
            h = self.conv2(h, edge_index, edge_weight=edge_weight)
        except TypeError:
            h = self.conv2(h, edge_index)
        out = self.alpha * h + (1 - self.alpha) * x
        return F.normalize(out, p=2, dim=1) if self.use_norm else out


# ---------- Utility functions for training/inference ----------
def train_graph_encoder(graph_data: Data, epochs=15, lr=1e-3, weight_decay=1e-4, hidden_dim=256, alpha=0.2):
    """
    Train a GraphSAGE encoder on given graph_data (with precomputed node features in graph_data.x).
    Returns trained model and final embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphEncoder(graph_data.num_nodes, graph_data.x.size(1), hidden_dim=hidden_dim, alpha=alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(graph_data.x.to(device), graph_data.edge_index.to(device),
                     edge_weight=getattr(graph_data, 'edge_attr', None))
        target = F.normalize(graph_data.x.to(device), p=2, dim=1)
        mask = (graph_data.x.norm(dim=1) > 1e-6)
        cos_loss = 1 - F.cosine_similarity(pred[mask], target[mask]).mean()

        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            w = graph_data.edge_attr.to(device)
            nb_cos = F.cosine_similarity(pred[graph_data.edge_index[0]], pred[graph_data.edge_index[1]])
            nb_loss = 1 - ( (w * nb_cos).sum() / w.sum().clamp(min=1e-9) )
        else:
            nb_cos = F.cosine_similarity(pred[graph_data.edge_index[0]], pred[graph_data.edge_index[1]])
            nb_loss = 1 - nb_cos.mean()

        loss = 0.7 * cos_loss + 0.3 * nb_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"[GNN] Epoch {epoch+1}/{epochs} | Loss={loss.item():.4f}")

    final_embs = model(torch.arange(graph_data.num_nodes).to(device), 
                       graph_data.edge_index.to(device), 
                       edge_weight=getattr(graph_data, 'edge_attr', None))
    return model, final_embs.cpu()


def infer_graph_encoder(model: GraphEncoder, graph_data: Data):
    """Run inference on a trained GraphEncoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.inference_mode():
        out = model(torch.arange(graph_data.num_nodes).to(device),
                    graph_data.edge_index.to(device),
                    edge_weight=getattr(graph_data, 'edge_attr', None))
    return out.cpu()
