import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import numpy as np

class PersonalLLM_Slim(nn.Module):
    """
    Slim variant with:
    - Frozen Flan-T5 backbone
    - Memmap-based history and graph embeddings
    - Session-aware Transformer
    - Gated cross-attention fusion applied to ALL encoder tokens
    """

    def __init__(self, llm_model, emb_model, max_input_len, max_new_len, task_id):
        super().__init__()
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len

        self.llm_config = self.llm_model.config
        self.llm_emb_size = self.llm_config.hidden_size
        self.emb_config = self.emb_model.config
        self.emb_emb_size = self.emb_config.hidden_size

        # === Load memmap history ===
        train_npy_path = f"../bge_emb/task_{task_id}_train_bge.npy"
        dev_npy_path   = f"../bge_emb/task_{task_id}_dev_bge.npy"
        if not (os.path.exists(train_npy_path) and os.path.exists(dev_npy_path)):
            raise FileNotFoundError("Memmap files missing — run conversion first.")
        self.his_train_memmap = np.load(train_npy_path, mmap_mode='r')
        self.his_dev_memmap   = np.load(dev_npy_path,   mmap_mode='r')

        # === Load memmap graph ===
        graph_path = f"../graph_emb/task_{task_id}_graph.npy"
        self.graph_memmap = np.load(graph_path, mmap_mode='r') if os.path.exists(graph_path) else None
        if self.graph_memmap is None:
            print("⚠ No graph embeddings found — skipping")

        # Trainable personalization modules with (Multi‑Layer Perceptron aligners)
        # ------------------------------------------------------------------
        # Layer: Instruction Token Embedding
        # Purpose: A single learnable vector appended to the sequence to inject
        #          a "prompt-like" control signal for the LLM during generation.
        self.inst_token = nn.Parameter(torch.rand(self.emb_emb_size), requires_grad=True)

        # For alignment MLPs, we use mult_k=1 for direct dim-matching
        self.mult_k = 1

        # Layer: Align MLP (for instruction token) 
        # Purpose: Projects instruction embedding from emb_model space (BGE dims)
        #          to LLM embedding space (Flan-T5 hidden size), allowing 
        #          fusion with the original token embedding table.
        self.align_mlp_inst = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size),
            nn.GELU(),
            nn.Linear(self.llm_emb_size, self.llm_emb_size)
        )

        # Layer: Align MLP (for general profile embeddings)
        # Purpose: Projects static profile/history embeddings to LLM space.
        self.align_mlp = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size),
            nn.GELU(),
            nn.Linear(self.llm_emb_size, self.llm_emb_size)
        )

        # Layer: Align MLP Session
        # Purpose: Projects short-term session embeddings from session encoder
        #          into the LLM token embedding space.
        self.align_mlp_session = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size),
            nn.GELU(),
            nn.Linear(self.llm_emb_size, self.llm_emb_size)
        )

        # Layer: Align MLP Graph
        # Purpose: Projects collaborative graph embeddings (user-item-relations)
        #          into the LLM token embedding space so they can be attended 
        #          alongside profile and session signals.
        self.align_mlp_graph = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size),
            nn.GELU(),
            nn.Linear(self.llm_emb_size, self.llm_emb_size)
        )

        # Layer: Session-aware Transformer Encoder
        # Purpose: Learns temporal context from recent clickstream or short-term 
        #          history (session_ids), output is zone that captures recency.
        self.session_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_emb_size, nhead=4, batch_first=True),
            num_layers=2
        )

        # Layer: Gated Multi-Head Cross-Attention
        # Purpose: Allows the LLM to attend to personalized vectors (profile, 
        #          session, graph) with a learned gate to control influence.
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.llm_emb_size, num_heads=8, batch_first=True)
        
        # Layer: Attention Gate
        # Purpose: Scalar gate computed per token by concatenating original 
        #          task embeddings and cross-attended outputs; balances 
        #          personalization vs. pure task relevance.
        self.gate = nn.Linear(self.llm_emb_size * 2, 1)

        # for layer in [self.align_mlp, self.align_mlp_inst,
        #               self.align_mlp_session, self.align_mlp_graph,
        #               self.session_encoder, self.cross_attn, self.gate]:
        #     for _, p in layer.named_parameters():
        #         p.requires_grad = True

        # # Freeze LLM backbone
        # for _, p in self.llm_model.named_parameters():
        #     p.requires_grad = False


        # ✅ Freeze full LLM backbone
        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False

        # ✅ Freeze full emb_model again (BGE encoder) — keep it as a fixed semantic space
        for _, p in self.emb_model.named_parameters():
            p.requires_grad = False

        # ✅ Ensure all fusion/projection layers remain trainable
        for layer in [self.align_mlp, self.align_mlp_inst,
                      self.align_mlp_session, self.align_mlp_graph,
                      self.session_encoder, self.cross_attn, self.gate]:
            for _, p in layer.named_parameters():
                p.requires_grad = True
                
        self._printed_debug_shapes = False

        # === PATCH: Print trainable parameter summary ===
        print("\n-------- TRAINABLE PARAMS --------")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"TRAINABLE: {name} | shape={tuple(p.shape)} | numel={p.numel()}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print("\n-------- SUMMARY --------")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print("-------------------------\n")

    # --- NEW helper ---
    def memmap_to_tensor(self, np_array, device, requires_grad=False):
        """
        Convert a NumPy memmap slice to float32 torch tensor on target device.
        """
        tensor = torch.from_numpy(np_array).to(device).float()
        tensor.requires_grad = requires_grad
        return tensor

    def obtain_task_emb(self, emb_input_ids, emb_attention_mask, emb_token_type_ids):
        task_outputs = self.emb_model(
            input_ids=emb_input_ids,
            attention_mask=emb_attention_mask,
            token_type_ids=emb_token_type_ids
        )
        task_vecs_raw = torch.nn.functional.normalize(task_outputs[0][:, 0], p=2, dim=1)
        self._last_task_vecs_raw = task_vecs_raw
        return self.align_mlp(task_vecs_raw)

    def obtain_profile_emb(self, his_id, task_embs_aligned):
        his_mask = torch.eq(his_id, 0)
        bsz = his_mask.size(0)
        his_mask = his_mask.repeat(1, self.mult_k).view(bsz * self.mult_k, -1)

        batch_np = self.his_train_memmap[his_id.cpu().numpy()] if self.training else self.his_dev_memmap[his_id.cpu().numpy()]
        his_embs = self.memmap_to_tensor(batch_np, task_embs_aligned.device, requires_grad=True)
        if his_embs.ndim == 2 and his_embs.size(-1) != self.emb_emb_size:
            raise ValueError(f"History emb dim mismatch: got {his_embs.size(-1)}, expected {self.emb_emb_size}")
        elif his_embs.ndim == 1:
            his_embs = his_embs.unsqueeze(0)

        # 🔹 Ensure dtype matches the Linear layer before projection (prevents mat1/mat2 dtype mismatch)
        align_first: nn.Linear = self.align_mlp[0]  # type: ignore
        target_dtype = align_first.weight.dtype
        his_embs = his_embs.to(dtype=target_dtype)

        his_embs_align = self.align_mlp(his_embs).view(bsz * self.mult_k, -1, self.llm_emb_size)
        task_embs_raw = self._last_task_vecs_raw
        his_weight = torch.bmm(his_embs, task_embs_raw.unsqueeze(-1))
        his_weight = his_weight.masked_fill(his_mask.unsqueeze(-1), -torch.inf)
        his_weight = torch.nn.functional.softmax(his_weight, dim=1)

        return torch.bmm(his_embs_align.transpose(1, 2), his_weight).squeeze(-1)

    def gated_cross_attention(self, task_embs, user_embs):
        attn_out, _ = self.cross_attn(task_embs, user_embs, user_embs)
        gate_val = torch.sigmoid(self.gate(torch.cat([task_embs, attn_out], dim=-1)))
        return gate_val * attn_out + (1 - gate_val) * task_embs

    def forward(self, llm_input_ids, llm_attention_mask, labels,
                emb_input_ids, emb_attention_mask, emb_token_type_ids,
                his_id, session_ids=None, graph_node_ids=None):

        task_embs = self.obtain_task_emb(emb_input_ids, emb_attention_mask, emb_token_type_ids)
        profile_embs = self.obtain_profile_emb(his_id, task_embs)
        if profile_embs.size(-1) != self.llm_emb_size:
            profile_embs = self.align_mlp(profile_embs)

        # --- Session fix ---
        session_embs = None
        if session_ids is not None:
            batch_np = (self.his_train_memmap if self.training else self.his_dev_memmap)[session_ids.cpu().numpy()]
            session_vecs = self.memmap_to_tensor(batch_np, task_embs.device, requires_grad=True)
            session_embs = self.session_encoder(session_vecs)[:, -1, :]
            if session_embs.size(-1) != self.llm_emb_size:
                session_embs = self.align_mlp_session(session_embs)

        # --- Graph fix ---
        graph_embs = None
        if self.graph_memmap is not None and graph_node_ids is not None:
            batch_np = self.graph_memmap[graph_node_ids.cpu().numpy()]
            graph_embs = self.memmap_to_tensor(batch_np, task_embs.device, requires_grad=True)
            if graph_embs.size(-1) != self.llm_emb_size:
                if graph_embs.dim() == 2:
                    graph_embs = self.align_mlp_graph(graph_embs)
                else:
                    bsz, seq_len, dim = graph_embs.size()
                    graph_embs = self.align_mlp_graph(graph_embs.view(-1, dim)).view(bsz, seq_len, -1)

        # Combine personalization signals
        user_embs_list = []
        if profile_embs is not None:
            user_embs_list.append(profile_embs.unsqueeze(1))
        if session_embs is not None:
            user_embs_list.append(session_embs.unsqueeze(1))
        if graph_embs is not None:
            user_embs_list.append(graph_embs)
        if not user_embs_list:
            user_embs_list.append(torch.zeros(task_embs.size(0), 1, self.llm_emb_size, device=task_embs.device))

        combined_user_embs = torch.cat(user_embs_list, dim=1)
        fused_task_embs = self.gated_cross_attention(task_embs.unsqueeze(1), combined_user_embs)

        # Get original input embeddings
        input_embs = self.llm_model.get_input_embeddings()(llm_input_ids)

        # === FIX: handle different fused_task_emb shapes ===
        if fused_task_embs.dim() == 2:
            fused_task_embs_expanded = fused_task_embs.unsqueeze(1).expand(-1, llm_input_ids.size(1), -1)
        elif fused_task_embs.dim() == 3:
            if fused_task_embs.size(1) == 1:
                fused_task_embs_expanded = fused_task_embs.expand(-1, llm_input_ids.size(1), -1)
            else:
                fused_task_embs_expanded = fused_task_embs
        else:
            raise ValueError(f"Unexpected fused_task_embs shape: {fused_task_embs.shape}")

        inst_token_emb = self.align_mlp_inst(self.inst_token).unsqueeze(0).unsqueeze(0)
        inst_token_emb = inst_token_emb.expand(llm_input_ids.size(0), llm_input_ids.size(1), -1)

        # Add personalization everywhere
        input_embs = input_embs + fused_task_embs_expanded + inst_token_emb

        if self.training:
            out = self.llm_model(inputs_embeds=input_embs,
                                 attention_mask=llm_attention_mask.long(),
                                 labels=labels.long(),
                                 use_cache=False,
                                 return_dict=True)
            return SequenceClassifierOutput(loss=out.loss)
        else:
            out = self.llm_model.generate(inputs_embeds=input_embs,
                                          attention_mask=llm_attention_mask,
                                          max_new_tokens=self.max_new_len,
                                          num_beams=4,
                                          num_return_sequences=1,
                                          return_dict_in_generate=True,
                                          do_sample=False)
            return [torch.FloatTensor([0.0]).to(llm_input_ids.device), out['sequences']]