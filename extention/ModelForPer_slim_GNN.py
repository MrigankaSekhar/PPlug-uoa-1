import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import numpy as np
import json
import torch.nn.functional as F


class PersonalLLM_Slim(nn.Module):
    """
    Slim variant with:
    - Frozen Flan-T5 backbone
    - Memmap-based history and graph embeddings
    - Session-aware Transformer
    - Gated cross-attention fusion applied to ALL encoder tokens
    """

    

    def __init__(self, llm_model, emb_model, max_input_len, max_new_len, task_id,
                 llm_tokenizer=None,
                 use_inst_token=True,
                 use_align_mlp_inst=True,
                 use_align_mlp=True,
                 use_session_encoder=True,
                 use_align_mlp_session=True,
                 use_align_mlp_graph=True,
                 use_cross_attn=True,
                 use_gate=True,
                 use_profile=True,   # ✅ profile/history embeddings ON/OFF
                 use_session=True,   # ✅ session embeddings ON/OFF
                 use_graph=True):    # ✅ graph embeddings ON/OFF
        super().__init__()
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.llm_tokenizer = llm_tokenizer
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len

        self.llm_config = self.llm_model.config
        self.llm_emb_size = self.llm_config.hidden_size
        self.emb_config = self.emb_model.config
        self.emb_emb_size = self.emb_config.hidden_size

        # ✅ NEW: If a source is disabled, also disable its dependent modules
        if not use_session:
            use_session_encoder = False
            use_align_mlp_session = False
        if not use_graph:
            use_align_mlp_graph = False

        # Store flags for A/B testing
        self.use_inst_token = use_inst_token
        self.use_align_mlp_inst = use_align_mlp_inst
        self.use_align_mlp = use_align_mlp
        self.use_align_mlp_session = use_align_mlp_session
        self.use_align_mlp_graph = use_align_mlp_graph
        self.use_session_encoder = use_session_encoder
        self.use_cross_attn = use_cross_attn
        self.use_gate = use_gate

        # ✅ NEW: source toggles (A/B)
        self.use_profile = use_profile
        self.use_session = use_session
        self.use_graph = use_graph

        # ✅ NEW: debug controls (avoid noisy/meaningless per-batch prints)
        self.enable_cosine_check = True         # ✅ Enable cosine checks to monitor profile↔graph alignment
        self.cosine_check_every = 25  # Check every 25 steps
        self.cosine_check_min_n = 16

        # ✅ NEW: gate monitor controls
        self.enable_gate_monitor = True
        self.gate_monitor_every = 10  # ✅ Print every 10 steps (was 50)
        self.gate_monitor_min_n = 64  # ✅ Lower threshold to see early-stage behavior

        self._forward_step = 0

        # === Load memmap history ===
        train_npy_path = f"../bge_emb/task_{task_id}_train_bge.memmap.npy"
        dev_npy_path   = f"../bge_emb/task_{task_id}_dev_bge.memmap.npy"

        # 🧭 These files contain precomputed sentence / profile embeddings (from the BGE model)
        #    stored as NumPy memory maps for fast random access. Each row corresponds to
        #    one history or review item (e.g., user profile element).
        #    
        # During training, slices are pulled from `task_{task_id}_train_bge.npy`;
        # during evaluation, from `task_{task_id}_dev_bge.npy`.
        # These fixed semantic vectors are *not trainable* — they act as the grounding
        # for the personalization component (profile/history context). 

        # ⚙️ Metrics Interpretation (printouts seen during training):
        #   • [Cosine‑Check] → Measures alignment between `profile_embs` (from BGE profiles)
        #                       and `graph_embs` (from GNN). Values:
        #                           ≈0.0–0.1 ⇒ nearly orthogonal (diverse info, early training)
        #                           ≈0.3–0.6 ⇒ moderate correlation (beginning to align)
        #                           >0.7     ⇒ strong alignment (embedding fusion stabilized)
        #
        #   • [Gate‑Monitor] → Mean activation of the sigmoid gate combining task vs. personalized signals:
        #                           ≈0.0 ⇒ model ignoring personalization (all task)
        #                           ≈0.5 ⇒ balanced fusion (50% task, 50% persona)
        #                           ≈1.0 ⇒ model fully preferring personalization / attention outputs
        #
        #   These diagnostics help verify gating and representation fusion during fine‑tuning.
        #   Ideal trend: Cosine slowly increases; Gate moves toward 0.6‑0.9 while loss steadily decreases.



        if not (os.path.exists(train_npy_path) and os.path.exists(dev_npy_path)):
            raise FileNotFoundError("Memmap files missing — run conversion first.")

        # Load meta for train memmap
        meta_path = train_npy_path.replace('_bge.memmap.npy', '_offsets.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        total_vectors = meta["total_vectors"]
        dim = meta["dim"]

        self.his_train_memmap = np.memmap(train_npy_path, mode='r', dtype='float32', shape=(total_vectors, dim))
        # Same fix for dev
        dev_meta_path = dev_npy_path.replace('_bge.memmap.npy', '_offsets.json')
        with open(dev_meta_path, 'r') as f:
            dev_meta = json.load(f)
        dev_total_vectors = dev_meta["total_vectors"]
        dev_dim = dev_meta["dim"]
        self.his_dev_memmap = np.memmap(dev_npy_path, mode='r', dtype='float32', shape=(dev_total_vectors, dev_dim))

        # === Load memmap graph ===
        graph_path = f"../graph_emb/task_{task_id}_graph.npy"
        self.graph_memmap = np.load(graph_path, mmap_mode='r') if os.path.exists(graph_path) else None
        if self.graph_memmap is None:
            print("⚠ No graph embeddings found — skipping")

        # Trainable personalization modules with (Multi‑Layer Perceptron aligners)
        # ------------------------------------------------------------------
        if self.use_inst_token:
            # Layer: Instruction Token Embedding
            # Purpose: A single learnable vector appended to the sequence to inject
            #          a "prompt-like" control signal for the LLM during generation.
            self.inst_token = nn.Parameter(torch.rand(self.emb_emb_size), requires_grad=True)

        # For alignment MLPs, we use mult_k=1 for direct dim-matching
        self.mult_k = 1

        if self.use_align_mlp_inst:
            # Layer: Align MLP (for instruction token) 
            # Purpose: Projects instruction embedding from emb_model space (BGE dims)
            #          to LLM embedding space (Flan-T5 hidden size), allowing 
            #          fusion with the original token embedding table.
            self.align_mlp_inst = nn.Sequential(
                nn.Linear(self.emb_emb_size, self.llm_emb_size),
                nn.GELU(),
                nn.Linear(self.llm_emb_size, self.llm_emb_size)
            )

        if self.use_align_mlp:
            # Layer: Align MLP (for general profile embeddings)
            # Purpose: Projects static profile/history embeddings to LLM space.
            self.align_mlp = nn.Sequential(
                nn.Linear(self.emb_emb_size, self.llm_emb_size),
                nn.GELU(),
                nn.Linear(self.llm_emb_size, self.llm_emb_size)
            )

        if self.use_align_mlp_session:
            # Layer: Align MLP Session
            # Purpose: Projects short-term session embeddings from session encoder
            #          into the LLM token embedding space.
            self.align_mlp_session = nn.Sequential(
                nn.Linear(self.emb_emb_size, self.llm_emb_size),
                nn.GELU(),
                nn.Linear(self.llm_emb_size, self.llm_emb_size)
            )

        if self.use_align_mlp_graph:
            # Layer: Align MLP Graph
            # Purpose: Projects collaborative graph embeddings (user-item-relations)
            #          into the LLM token embedding space so they can be attended 
            #          alongside profile and session signals.
            self.align_mlp_graph = nn.Sequential(
                nn.Linear(self.emb_emb_size, self.llm_emb_size),
                nn.GELU(),
                nn.Linear(self.llm_emb_size, self.llm_emb_size)
            )

        if self.use_session_encoder:
            # Layer: Session-aware Transformer Encoder
            # Purpose: Learns temporal context from recent clickstream or short-term 
            #          history (session_ids), output is zone that captures recency.
            self.session_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_emb_size, nhead=4, batch_first=True),
                num_layers=2
            )
            # ✅ NEW: Add learnable positional encoding for recency
            self.session_pos_enc = nn.Parameter(
                torch.randn(1, 20, self.emb_emb_size) * 0.02,  # Support up to 20 session items
                requires_grad=True
            )

        if self.use_cross_attn:
            # Layer: Gated Multi-Head Cross-Attention
            # Purpose: Allows the LLM to attend to personalized vectors (profile, 
            #          session, graph) with a learned gate to control influence.
            self.cross_attn = nn.MultiheadAttention(embed_dim=self.llm_emb_size, num_heads=8, batch_first=True)
        
        if self.use_gate:
            # Layer: Attention Gate
            # Purpose: Scalar gate computed per token by concatenating original 
            #          task embeddings and cross-attended outputs; balances 
            #          personalization vs. pure task relevance.
            self.gate_norm = nn.LayerNorm(self.llm_emb_size * 2)
            self.gate = nn.Linear(self.llm_emb_size * 2, 1)
            # ✅ Start neutral (sigmoid(0) = 0.5)
            nn.init.constant_(self.gate.bias, 0.0)

        # ✅ Freeze full LLM backbone
        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False

        # ✅ Freeze full emb_model again (BGE encoder) — keep it as a fixed semantic space
        for _, p in self.emb_model.named_parameters():
            p.requires_grad = False

        # ✅ Ensure all fusion/projection layers remain trainable (but only if they exist)
        for layer_name in [
            'align_mlp', 'align_mlp_inst', 'align_mlp_session', 'align_mlp_graph',
            'session_encoder', 'cross_attn', 'gate'
        ]:
            layer = getattr(self, layer_name, None)
            if layer is not None:
                for _, p in layer.named_parameters():
                    p.requires_grad = True

        # ✅ NEW: If session disabled, freeze any leftover session params (extra safety)
        if not use_session:
            for name in ["align_mlp_session", "session_encoder", "session_pos_enc"]:
                layer = getattr(self, name, None)
                if layer is None:
                    continue
                if isinstance(layer, torch.nn.Parameter):
                    layer.requires_grad = False
                else:
                    for _, p in layer.named_parameters():
                        p.requires_grad = False

        # ✅ NEW: If graph disabled, freeze any leftover graph params (extra safety)
        if not use_graph:
            layer = getattr(self, "align_mlp_graph", None)
            if layer is not None:
                for _, p in layer.named_parameters():
                    p.requires_grad = False
                
        self._printed_debug_shapes = False

        # ✅ Ensure session_pos_enc remains trainable
        if hasattr(self, 'session_pos_enc'):
            self.session_pos_enc.requires_grad = True

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

    def _safe_softmax_masked(self, scores: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        scores: (B, L, 1) or (B, L)
        mask:   (B, L) where True means 'masked out'
        Returns softmax(scores) over dim with masked positions excluded.
        If an entire row is masked, returns all zeros (no NaNs).
        """
        if scores.dim() == 3 and mask.dim() == 2:
            mask_exp = mask.unsqueeze(-1)
        else:
            mask_exp = mask

        scores = scores.masked_fill(mask_exp, float("-inf"))
        w = torch.nn.functional.softmax(scores, dim=dim)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        denom = w.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        w = w / denom
        return w

    def memmap_to_tensor(self, np_array, device, requires_grad=False):
        """
        Convert a NumPy memmap slice to float32 torch tensor on target device.
        """
        if not np_array.flags.writeable:
            np_array = np_array.copy()
        tensor = torch.from_numpy(np_array).to(device).float()
        tensor.requires_grad = requires_grad
        return tensor

    def obtain_task_emb(self, emb_input_ids, emb_attention_mask, emb_token_type_ids):
        task_outputs = self.emb_model(
            input_ids=emb_input_ids,
            attention_mask=emb_attention_mask,
            token_type_ids=emb_token_type_ids,
            return_dict=True
        )
        cls = task_outputs.last_hidden_state[:, 0]
        task_vecs_raw = torch.nn.functional.normalize(cls, p=2, dim=1)
        self._last_task_vecs_raw = task_vecs_raw
        return self.align_mlp(task_vecs_raw) if self.use_align_mlp else task_vecs_raw

    def obtain_profile_emb(self, his_id, task_embs_aligned):
        his_mask = torch.eq(his_id, 0)
        bsz = his_mask.size(0)
        his_mask = his_mask.repeat(1, self.mult_k).view(bsz * self.mult_k, -1)

        batch_np = self.his_train_memmap[his_id.cpu().numpy()] if self.training else self.his_dev_memmap[his_id.cpu().numpy()]
        # ✅ FIX: memmap vectors are fixed inputs; don't build grads through them
        his_embs = self.memmap_to_tensor(batch_np, task_embs_aligned.device, requires_grad=False)

        if his_embs.ndim == 2 and his_embs.size(-1) != self.emb_emb_size:
            raise ValueError(f"History emb dim mismatch: got {his_embs.size(-1)}, expected {self.emb_emb_size}")
        elif his_embs.ndim == 1:
            his_embs = his_embs.unsqueeze(0)

        if self.use_align_mlp:
            # 🔹 Ensure dtype matches the Linear layer before projection (prevents mat1/mat2 dtype mismatch)
            align_first: nn.Linear = self.align_mlp[0]  # type: ignore
            target_dtype = align_first.weight.dtype
            his_embs = his_embs.to(dtype=target_dtype)
            his_embs_align = self.align_mlp(his_embs).view(bsz * self.mult_k, -1, self.llm_emb_size)
        else:
            his_embs_align = his_embs

        task_embs_raw = self._last_task_vecs_raw
        his_weight = torch.bmm(his_embs, task_embs_raw.unsqueeze(-1))
        his_weight = self._safe_softmax_masked(his_weight, his_mask, dim=1)
        his_weight = his_weight.to(his_embs.dtype)

        return torch.bmm(his_embs_align.transpose(1, 2), his_weight).squeeze(-1)

    def gated_cross_attention(self, task_embs, user_embs):
        if not self.use_cross_attn:
            return task_embs
        attn_out, _ = self.cross_attn(task_embs, user_embs, user_embs)
        if self.use_gate:
            gate_input = torch.cat([task_embs, attn_out], dim=-1)
            gate_input = self.gate_norm(gate_input)
            gate_val = torch.sigmoid(self.gate(gate_input))
            return gate_val * attn_out + (1 - gate_val) * task_embs
        else:
            return attn_out

    def _get_rating_token_ids(self) -> torch.Tensor:
        if self.llm_tokenizer is None:
            raise ValueError("llm_tokenizer must be provided to compute rating token ids")
        ids = [self.llm_tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
        return torch.tensor(ids, dtype=torch.long)

    def _rating_only_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels.ne(-100)
        if not valid.any():
            return logits.new_tensor(0.0)

        first_pos = valid.float().argmax(dim=1)
        bsz = logits.size(0)
        gathered = logits[torch.arange(bsz, device=logits.device), first_pos]  # (B, vocab)

        rating_ids = self._get_rating_token_ids().to(device=logits.device)
        rating_logits = gathered.index_select(dim=-1, index=rating_ids)  # (B, 5)

        gold_token = labels[torch.arange(bsz, device=logits.device), first_pos]

        # keep only examples where gold_token is one of the rating_ids
        is_rating = (gold_token.unsqueeze(1) == rating_ids.unsqueeze(0)).any(dim=1)
        print(f"[DEBUG] rating_label_match: {is_rating.float().mean().item():.3f}")
        if not is_rating.any():
            return logits.new_tensor(0.0)

        rating_logits = rating_logits[is_rating]
        gold_token = gold_token[is_rating]

        gold_class = (gold_token.unsqueeze(1) == rating_ids.unsqueeze(0)).float().argmax(dim=1)
        return F.cross_entropy(rating_logits, gold_class)

    def forward(self, llm_input_ids, llm_attention_mask, labels,
                emb_input_ids, emb_attention_mask, emb_token_type_ids,
                his_id, session_ids=None, graph_node_ids=None, graph_node_mask=None):

        self._forward_step += 1

        task_embs = self.obtain_task_emb(emb_input_ids, emb_attention_mask, emb_token_type_ids)
        profile_embs = None
        session_embs = None
        graph_embs = None

        # Detect if we are on GPU and in eval mode → can downcast to fp16
        use_fp16_cache = (not self.training) and (task_embs.device.type == "cuda")

        # ==========================================================
        # 🔹 A/B: disable sources by nulling their IDs
        # ==========================================================
        if not getattr(self, "use_profile", True):
            his_id = None
        if not getattr(self, "use_session", True):
            session_ids = None
        if not getattr(self, "use_graph", True):
            graph_node_ids = None

        # ==========================================================
        # 🔹 1) COLLECT UNIQUE IDS PER SOURCE (keep spaces separate!)
        # ==========================================================
        his_like_ids = []
        if his_id is not None:
            his_like_ids.append(his_id.view(-1))
        if session_ids is not None:
            his_like_ids.append(session_ids.view(-1))

        if his_like_ids:
            unique_his_ids = torch.unique(torch.cat(his_like_ids)).cpu().numpy()
        else:
            unique_his_ids = np.array([], dtype=np.int64)

        # Graph ids are in a different id-space
        unique_graph_ids = np.array([], dtype=np.int64)
        if graph_node_ids is not None:
            flat_graph = graph_node_ids.view(-1)
            flat_graph = flat_graph[flat_graph >= 0]
            unique_graph_ids = torch.unique(flat_graph).cpu().numpy() if flat_graph.numel() > 0 else np.array([], dtype=np.int64)

        # ==========================================================
        # 🔹 2) LOAD MEMMAP SLICES ONCE PER SOURCE TYPE INTO CACHE
        # ==========================================================
        def maybe_fp16(t: torch.Tensor) -> torch.Tensor:
            return t.half() if use_fp16_cache else t

        # History/session cache (ONLY history/session IDs)
        his_cache = {}
        if unique_his_ids.size > 0:
            his_memmap_src = self.his_train_memmap if self.training else self.his_dev_memmap
            his_cache = {
                int(id_val): maybe_fp16(self.memmap_to_tensor(
                    his_memmap_src[int(id_val)], task_embs.device, requires_grad=False
                ))
                for id_val in unique_his_ids if int(id_val) < len(his_memmap_src) and int(id_val) >= 0
            }

        # Graph cache (ONLY graph IDs)
        graph_cache = {}
        if self.graph_memmap is not None and unique_graph_ids.size > 0:
            graph_cache = {
                int(id_val): maybe_fp16(self.memmap_to_tensor(
                    self.graph_memmap[int(id_val)], task_embs.device, requires_grad=False
                ))
                for id_val in unique_graph_ids if int(id_val) < len(self.graph_memmap) and int(id_val) >= 0
            }
        # ==========================================================
        # 🔹 3) PROFILE EMBEDDINGS (refactor: call obtain_profile_emb)
        # ==========================================================
        if his_id is not None:
            profile_embs = self.obtain_profile_emb(his_id=his_id, task_embs_aligned=task_embs)

        # ==========================================================
        # 🔹 4) SESSION EMBEDDINGS FROM SAME CACHE
        # ==========================================================
        if session_ids is not None:
            sess_embs = torch.stack([
                his_cache.get(int(i.item()), torch.zeros(self.emb_emb_size, device=task_embs.device))
                for i in session_ids.view(-1)
            ])
            sess_embs = sess_embs.view(session_ids.size(0), session_ids.size(1), -1)
            
            # ✅ ADD: Apply temporal positional encoding (from fix #2)
            seq_len = sess_embs.size(1)
            if seq_len <= self.session_pos_enc.size(1):
                sess_embs = sess_embs + self.session_pos_enc[:, :seq_len, :]
            
            # ✅ NEW: Add recency decay weighting (exponential down-weighting of older items)
            decay = torch.exp(-0.1 * torch.arange(seq_len, device=sess_embs.device).float())
            decay = decay / decay.sum()  # Normalize to sum=1
            sess_embs = sess_embs * decay.view(1, -1, 1)  # Apply decay per timestep
            
            if self.use_session_encoder:
                session_embs = self.session_encoder(sess_embs)[:, -1, :]
            else:
                session_embs = sess_embs[:, -1, :]
            
            if session_embs.size(-1) != self.llm_emb_size and self.use_align_mlp_session:
                session_embs = self.align_mlp_session(session_embs)

        # ==========================================================
        # 🔹 5) GRAPH EMBEDDINGS FROM GRAPH CACHE
        # ==========================================================
        if self.graph_memmap is not None and graph_node_ids is not None:
            flat_ids = graph_node_ids.view(-1)
            g_embs = torch.stack([
                graph_cache.get(int(i.item()), torch.zeros(self.emb_emb_size, device=task_embs.device))
                if int(i.item()) >= 0 else torch.zeros(self.emb_emb_size, device=task_embs.device)
                for i in flat_ids
            ])
            if graph_node_ids.ndim > 1:
                g_embs = g_embs.view(graph_node_ids.size(0), graph_node_ids.size(1), -1)

            # ✅ Apply explicit mask if provided (extra safety)
            if graph_node_mask is not None:
                m = graph_node_mask.to(device=task_embs.device).unsqueeze(-1).float()
                if m.shape[:2] == g_embs.shape[:2]:
                    g_embs = g_embs * m

            if g_embs.size(-1) != self.llm_emb_size and self.use_align_mlp_graph:
                if g_embs.dim() == 2:
                    g_embs = self.align_mlp_graph(g_embs)
                else:
                    bsz, seq_len, dim = g_embs.size()
                    g_embs = self.align_mlp_graph(g_embs.view(-1, dim)).view(bsz, seq_len, -1)
            graph_embs = g_embs


        # ==========================================================
        # 🔹 (Optional) Metric Checks — Throttled and only if meaningful
        # ==========================================================
        if (
            self.enable_cosine_check
            and (self._forward_step % self.cosine_check_every == 0)
            and (graph_embs is not None)
            and (profile_embs is not None)
        ):
            with torch.no_grad():
                p = profile_embs.mean(dim=1) if profile_embs.ndim == 3 else profile_embs
                g = graph_embs.mean(dim=1) if graph_embs.ndim == 3 else graph_embs

                p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                if p.size(-1) == g.size(-1):
                    p_norm = torch.linalg.norm(p, dim=-1)
                    g_norm = torch.linalg.norm(g, dim=-1)
                    valid = (p_norm > 1e-6) & (g_norm > 1e-6)

                    if int(valid.sum().item()) >= self.cosine_check_min_n:
                        cos_sim = torch.nn.functional.cosine_similarity(p[valid], g[valid], dim=-1)
                        mean_cos = float(cos_sim.mean().item())
                        std_cos = float(cos_sim.std(unbiased=False).item())
                        print(f"[Cosine‑Check] step={self._forward_step} mean={mean_cos:.4f} ±{std_cos:.4f} (n={cos_sim.numel()})")
                # else: stay silent (dim mismatch not useful in logs)

        # ✅ NEW: Session vs Profile cosine check
        if (
            self.enable_cosine_check
            and (self._forward_step % self.cosine_check_every == 0)
            and (session_embs is not None)
            and (profile_embs is not None)
        ):
            with torch.no_grad():
                s = session_embs if session_embs.ndim == 2 else session_embs.mean(dim=1)
                p = profile_embs if profile_embs.ndim == 2 else profile_embs.mean(dim=1)
                
                s = torch.nan_to_num(s, nan=0.0)
                p = torch.nan_to_num(p, nan=0.0)
                
                if s.size(-1) == p.size(-1):
                    s_norm = torch.linalg.norm(s, dim=-1)
                    p_norm = torch.linalg.norm(p, dim=-1)
                    valid = (s_norm > 1e-6) & (p_norm > 1e-6)
                    
                    if int(valid.sum().item()) >= 16:
                        cos_sim = torch.nn.functional.cosine_similarity(s[valid], p[valid], dim=-1)
                        mean_cos = float(cos_sim.mean().item())
                        print(f"[Session↔Profile] step={self._forward_step} cos={mean_cos:.4f}")

        # ==========================================================
        # 🔹 6) COMBINE SIGNALS + CONTINUE EXACTLY LIKE BEFORE
        # ==========================================================
        user_embs_list = []
        target_dim = self.llm_emb_size
        def _ensure_llm_dim(x: torch.Tensor, align_layer: nn.Module = None):
            if x is None: return None
            if x.size(-1) != target_dim:
                return align_layer(x) if align_layer is not None else x
            return x

        if profile_embs is not None:
            user_embs_list.append(_ensure_llm_dim(profile_embs, getattr(self, "align_mlp", None)).unsqueeze(1))
        if session_embs is not None:
            user_embs_list.append(_ensure_llm_dim(session_embs, getattr(self, "align_mlp_session", None)).unsqueeze(1))
        if graph_embs is not None:
            if graph_embs.ndim == 2:
                user_embs_list.append(_ensure_llm_dim(graph_embs, getattr(self, "align_mlp_graph", None)).unsqueeze(1))
            else:
                user_embs_list.append(graph_embs)

        if not user_embs_list:
            user_embs_list.append(torch.zeros(task_embs.size(0), 1, target_dim, device=task_embs.device))

        combined_user_embs = torch.cat(user_embs_list, dim=1)
        fused_task_embs = self.gated_cross_attention(task_embs.unsqueeze(1), combined_user_embs)

        gate_reg = None
        if getattr(self, "use_gate", False) and hasattr(self, "gate"):
            attn_out, _ = self.cross_attn(task_embs.unsqueeze(1), combined_user_embs, combined_user_embs)
            gate_input = torch.cat([task_embs.unsqueeze(1), attn_out], dim=-1)
            gate_input_normed = self.gate_norm(gate_input)
            gate_pre = self.gate(gate_input_normed)
            gate_vals = torch.sigmoid(gate_pre)

            reg_weight = 0.0
            mean_gate_tensor = gate_vals.mean()
            std_gate_tensor = gate_vals.std(unbiased=False) + 1e-6
            reg_balance = (mean_gate_tensor - 0.5).pow(2)
            reg_flat = (1.0 / std_gate_tensor)

            ent_reg = -(gate_vals * torch.log(gate_vals + 1e-8) + (1 - gate_vals) * torch.log(1 - gate_vals + 1e-8)).mean()
            gate_reg = reg_weight * (reg_balance + 0.1 * reg_flat + 0.5 * ent_reg)

            # ✅ Gate monitor (throttled, meaningful n)
            if (
                getattr(self, "enable_gate_monitor", False)
                and (self._forward_step % getattr(self, "gate_monitor_every", 50) == 0)
            ):
                with torch.no_grad():
                    gflat = gate_vals.detach().view(-1)
                    n = int(gflat.numel())
                    if n >= getattr(self, "gate_monitor_min_n", 256):
                        mean_gate = float(gflat.mean().item())
                        std_gate = float(gflat.std(unbiased=False).item())
                        sat_lo = float((gflat < 0.05).float().mean().item())
                        sat_hi = float((gflat > 0.95).float().mean().item())
                        print(
                            f"[Gate‑Monitor] step={self._forward_step} "
                            f"mean={mean_gate:.3f} std={std_gate:.3f} "
                            f"sat(<0.05)={sat_lo:.2%} sat(>0.95)={sat_hi:.2%} (n={n})"
                        )
                    else:
                        # too few values -> skip printing
                        pass

        if hasattr(self, "gate") and (self._forward_step % 50 == 0):
            with torch.no_grad():
                gw = self.gate.weight.detach()
                print(f"[Param-Norm] step={self._forward_step} gate.weight.norm={gw.norm().item():.6f}")

        # Get original input embeddings
        input_embs = self.llm_model.get_input_embeddings()(llm_input_ids)

        # === CHANGE: inject personalization only into the first token ===
        # Build a per-position delta that is zero everywhere except position 0
        if fused_task_embs.dim() == 2:
            fused_vec = fused_task_embs  # (B, H)
        else:
            fused_vec = fused_task_embs[:, 0, :]  # (B, H) when (B,1,H)

        delta = torch.zeros_like(input_embs)
        delta[:, 0, :] = fused_vec

        if self.use_inst_token:
            inst_vec = self.align_mlp_inst(self.inst_token) if self.use_align_mlp_inst else self.inst_token
            inst_vec = inst_vec.to(device=input_embs.device, dtype=input_embs.dtype).unsqueeze(0)  # (1, H)
            delta[:, 0, :] = delta[:, 0, :] + inst_vec.expand(delta.size(0), -1)

        input_embs = input_embs + delta

        if self.training:
            out = self.llm_model(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask.long(),
                labels=labels.long(),
                use_cache=False,
                return_dict=True
            )

            # === CHANGE: rating-only loss (align train with eval constraint) ===
            loss = self._rating_only_loss(out.logits, labels.long())

            if gate_reg is not None:
                loss = loss + gate_reg

            return {"loss": loss, "logits": out.logits}
        else:
            # ✅ Compute real loss during eval
            eval_out = self.llm_model(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask.long(),
                labels=labels.long(),
                use_cache=False,
                return_dict=True
            )
            loss = self._rating_only_loss(eval_out.logits, labels.long())

            # ✅ Generate sequences with STRICT single-token constraint
            if self.llm_tokenizer is None:
                raise ValueError("llm_tokenizer must be provided for constrained generation")
            
            # Get actual token IDs for ratings 1-5
            rating_token_ids = [
                self.llm_tokenizer.encode(str(i), add_special_tokens=False)[0] 
                for i in range(1, 6)
            ]
            
            # Create logits processor to force rating tokens only
            from transformers import LogitsProcessorList, LogitsProcessor
            
            class RatingConstraintProcessor(LogitsProcessor):
                """Forces model to only output rating tokens 1-5."""
                def __init__(self, allowed_token_ids, eos_token_id):
                    self.allowed_token_ids = set(allowed_token_ids)
                    self.eos_token_id = eos_token_id
                
                def __call__(self, input_ids, scores):
                    # On first generation step: allow only rating tokens
                    if input_ids.size(1) == 1:  # Decoder start token only
                        mask = torch.full_like(scores, float('-inf'))
                        for tid in self.allowed_token_ids:
                            mask[:, tid] = 0
                        return scores + mask
                    
                    # On subsequent steps: force EOS to stop generation
                    else:
                        mask = torch.full_like(scores, float('-inf'))
                        mask[:, self.eos_token_id] = 0
                        return scores + mask
            
            logits_processor = LogitsProcessorList([
                RatingConstraintProcessor(
                    rating_token_ids, 
                    self.llm_model.config.eos_token_id
                )
            ])
            
            gen_out = self.llm_model.generate(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask,
                max_new_tokens=2,      # Allow rating + EOS
                min_new_tokens=1,      # Force at least rating token
                num_beams=1,           # Greedy (fastest)
                do_sample=False,       # Deterministic
                logits_processor=logits_processor,  # ✅ Single token + EOS
                pad_token_id=self.llm_model.config.pad_token_id,
                eos_token_id=self.llm_model.config.eos_token_id,
                return_dict_in_generate=True
            )
            return [loss, gen_out['sequences']]