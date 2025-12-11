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
----------------------------------





1️⃣ History Embedding Tables — `his_train_emb_table` & `his_dev_emb_table`
What it is:  
Two nn.Embedding tables storing precomputed BGE-style embeddings for each historical user item (train and dev split).
Purpose here:  
Freeze these embeddings so they provide a stable long-term personalization signal.  
his_train_emb_table → used when self.training == True.  
his_dev_emb_table → used in evaluation mode.
Effect:  
Avoids re-calculating historical vectors each training step; preserves the user's historical "profile" as a fixed reference in model fusion.

2️⃣ Graph Embedding Table — `graph_emb_table` (using compute_graph_emb_generic.py)
What it is:  
An nn.Embedding table for precomputed graph embeddings (from a Neo4j graph / PyTorch Geometric pipeline).
Purpose here:  
Encodes user-item-review relationships from graph data. Operates in parallel to behavioral & linguistic embeddings.
Effect:  
Adds a collaborative-filter-style personalization signal — captures relational patterns between entities in the dataset.

3️⃣ Learnable Instruction Token — `inst_token`
What it is:  
A single learnable vector (nn.Parameter) the same size as the embedding model output.
Purpose here:  
Serves as a special marker token inside the LLM embedding space, representing an "instruction" or control hint.
Effect:  
Lets the model inject a consistent non-task signal into the LLM sequence (usually for prompting or guiding behavior).

4️⃣ Alignment MLP — `align_mlp_inst` & `align_mlp`
What it is:  
Two identical nn.Sequential MLP pipelines:
nn.Linear(emb_emb_size, llm_emb_size * mult_k)
nn.GELU()
nn.Linear(llm_emb_size * mult_k, llm_emb_size * mult_k)
Purpose here:  
Project embeddings from the BGE/embedding space (emb_emb_size) into the LLM hidden space (llm_emb_size), so personalization vectors match the dimensionality expected by the LLM.
align_mlp_inst → aligns the instruction token vector.
align_mlp → aligns history/session/graph embeddings.
Effect:  
Ensures vectors from the personalization encoders can be directly added or fused with the LLM’s token embeddings without dimension mismatch.

5️⃣ Session-Aware Transformer — `session_encoder` ( # === NEW: Session IDs (short-term context) === has been added and derived from recent 3 of his_id , see personaldataset_profile.py)
What it is:  
A small stacked Transformer encoder (nn.TransformerEncoder) with:
d_model = emb_emb_size
nhead = 4  
2 layers
Purpose here:  
Encodes recent user interactions (clickstream, short-term actions), preserving temporal relationships via self-attention.
Effect:  
Produces a "session embedding" summarizing short-term intent, complementing long-term profile and graph signals.

6️⃣ Gated Cross-Attention — `cross_attn` + `gate`
What it is:  
A standard nn.MultiheadAttention with 8 heads.
A gating MLP nn.Linear(llm_emb_size * 2, 1).
Purpose here:  
Fuse task embeddings (query meaning) with user personalization embeddings:
Queries = task embeddings
Keys/Values = personalization embeddings
Gate decides how much personalization to inject per token.
Effect:  
Dynamic per-example, per-token blending of personalization with current task context — prevents over-personalization or irrelevant bias.

7️⃣ `obtain_task_emb` Method
What it is:  
A method forwarding inputs into the embedding model (self.emb_model) and normalizing the [CLS]-like representation.
Purpose here:  
Extract task semantic meaning into a dense vector representation ready for fusion with personalization signals.
Effect:  
Represents the current query’s meaning in embeddings before fusion.

8️⃣ `obtain_profile_emb` Method
What it is:  
Uses dot products between historical embeddings and the task embedding to compute an attention-weighted profile vector.
Purpose here:  
Weight long-term history entries by relevance to current query. Produces a profile embedding summarizing relevant history.
Effect:  
Focuses on the most relevant historical interactions per task — more contextually-aware personalization.

9️⃣ Fusion & Injection into LLM Embeddings (`forward`)
What it is:  
The main forward pass logic:
Get task embeddings.
Get profile, session, and graph embeddings.
Concatenate personalization sources → combined_user_embs.
Fuse with gated cross-attention → fused_task_embs.
Inject fused embeddings into the LLM's input token embedding matrix at special indices (like vocab_size-1).
Purpose here:  
Augment the LLM's input tokens with additional personalization context vectors.
Effect:  
The frozen LLM receives input embeddings already modified with personalized signals — enabling personalization without fine-tuning the LLM weights.
📌 Final Takeaway
The architecture is a plug-in personalization module:
Precomputed, frozen long-term signals (history/graph)
On-the-fly short-term signals (session)
Smart fusion via gated cross-attention
Dimension alignment via MLPs
Injection into the frozen LLM embedding matrix
This lets you condition a frozen LLM on multiple personalization sources without retraining the LLM itself.
