# PPlug Extensions

This folder contains three architectural improvements for the PPlug model on LaMP‑3:

## 1. Gated Cross-Attention Layer
- **Purpose**: Fuse task-specific embeddings (from LLM input encoder) with personalization embeddings (profile, session, graph).
- **Key Benefit**: Learns dynamic weighting between task and personalization signals via a gate mechanism.
- **Integration**: Replace simple concatenation in `ModelForPer.py` with this module.

## 2. Session-Aware Transformer
- **Purpose**: Encode recent clickstream or last N user interactions for short-term personalization.
- **Key Benefit**: Captures temporal and contextual patterns in recent actions.
- **Integration**: Pass session embeddings alongside profile embeddings into Gated Cross-Attention.

## 3. Graph Neural Network Encoder
- **Purpose**: Encode relational context between users, items, and reviews from a Neo4j/PyTorch Geometric graph.
- **Key Benefit**: Adds collaborative and relational filtering signals to personalization.
- **Integration**: Precompute node embeddings offline; plug them in as part of user_embs in Gated Cross-Attention.

---

## How to Use in Code
In `ModelForPer.py`:
```python
from extention.gated_cross_attention import GatedCrossAttention
from extention.session_transformer import SessionTransformer
from extention.gnn_encoder import GNNEncoder


Initialize in __init__:
self.cross_attn = GatedCrossAttention(embed_dim=self.llm_emb_size)
self.session_encoder = SessionTransformer(input_dim=self.emb_emb_size)
self.gnn_encoder = GNNEncoder(num_nodes, self.emb_emb_size)

Fuse in forward():
session_embs = self.session_encoder(session_input)
graph_embs = self.gnn_encoder(graph_nodes, edge_index)
combined_user_embs = torch.cat([profile_embs, session_embs.unsqueeze(1), graph_embs.unsqueeze(1)], dim=1)
fused_task_embs = self.cross_attn(task_embs, combined_user_embs)

---

Next Step:  
We should modify `PersonalDataset_profile.py` so that in `__getitem__` it also emits:
- `session_ids`
- `graph_node_ids`  
from your existing LaMP‑3 profile data, so these extension modules can get their inputs.

Do you want me to go ahead and update `PersonalDataset_profile.py` for that? That will make the extension modules immediately usable with your dataset.

---
