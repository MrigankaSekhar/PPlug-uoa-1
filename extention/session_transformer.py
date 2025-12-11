
import torch
import torch.nn as nn

class SessionTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, session_embs):
        """
        session_embs: (B, S, E) sequence of recent interactions
        returns: (B, E) session summary (last step representation)
        """
        out = self.transformer(session_embs)
        return out[:, -1, :]
