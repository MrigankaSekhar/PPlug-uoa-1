
import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, 1)

    def forward(self, task_embs, user_embs):
        """
        task_embs: (B, T, E) -> LLM task encoder outputs
        user_embs: (B, U, E) -> concatenated personalization embeddings
        """
        attn_out, _ = self.cross_attn(task_embs, user_embs, user_embs)
        gate_val = torch.sigmoid(self.gate(torch.cat([task_embs, attn_out], dim=-1)))
        return gate_val * attn_out + (1 - gate_val) * task_embs
