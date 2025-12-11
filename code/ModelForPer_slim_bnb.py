
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, BitsAndBytesConfig

class PersonalLLM_Slim_BNB(nn.Module):
    """
    Slim variant of PersonalLLM with optional BitsAndBytesConfig quantization.
    Supports both 4-bit and 8-bit quantization.
    Keeps LLM frozen, trains only adapters + embedding alignment layers.
    """

    def __init__(self, model_args, emb_model, max_input_len, max_new_len, task_id,
                 use_4bit=False, use_8bit=False):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.task_id = task_id

        # ---------------- LLM Model with optional quantization ----------------
        if use_4bit:
            print("[Info] Loading model in 4-bit quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.llm_model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_path,
                quantization_config=quant_config,
                device_map="auto"
            )
        elif use_8bit:
            print("[Info] Loading model in 8-bit quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None
            )
            self.llm_model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_path,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            print("[Info] Loading model in full precision...")
            self.llm_model = T5ForConditionalGeneration.from_pretrained(model_args.model_path)

        # Freeze all LLM parameters
        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False

        self.llm_emb_size = self.llm_model.config.d_model
        self.emb_model = emb_model
        self.emb_emb_size = emb_model.config.hidden_size
        self.mult_k = 1

        # ---------------- Embedding alignment MLP ----------------
        self.align_mlp = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size * self.mult_k),
            nn.GELU(),
            nn.Linear(self.llm_emb_size * self.mult_k, self.llm_emb_size * self.mult_k)
        )

        # ---------------- History embedding tables ----------------
        his_train_emb = torch.cat([
            torch.zeros(1, self.emb_emb_size),
            torch.load(f"../bge_emb/task_{task_id}_train_bge.emb")
        ], 0)
        self.his_train_emb_table = nn.Embedding(his_train_emb.size(0), self.emb_emb_size)
        self.his_train_emb_table.weight = nn.Parameter(his_train_emb)
        for _, p in self.his_train_emb_table.named_parameters():
            p.requires_grad = False

        his_dev_emb = torch.cat([
            torch.zeros(1, self.emb_emb_size),
            torch.load(f"../bge_emb/task_{task_id}_dev_bge.emb")
        ], 0)
        self.his_dev_emb_table = nn.Embedding(his_dev_emb.size(0), self.emb_emb_size)
        self.his_dev_emb_table.weight = nn.Parameter(his_dev_emb)
        for _, p in self.his_dev_emb_table.named_parameters():
            p.requires_grad = False

        # ---------------- Gated Cross Attention ----------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.llm_emb_size,
            num_heads=8,
            batch_first=True
        )
        self.gate = nn.Linear(self.llm_emb_size * 2, 1)

    def obtain_task_emb(self, emb_input_ids, emb_attention_mask, emb_token_type_ids):
        task_inputs = {
            'input_ids': emb_input_ids,
            'attention_mask': emb_attention_mask,
            'token_type_ids': emb_token_type_ids
        }
        task_outputs = self.emb_model(**task_inputs)
        # Raw BGE embedding (768)
        task_vecs_raw = torch.nn.functional.normalize(
            task_outputs[0][:, 0], p=2, dim=1
        )
        # Save for later use in obtain_profile_emb
        self._last_task_vecs_raw = task_vecs_raw
        # Project into LLM space (512)
        task_vecs_aligned = self.align_mlp(task_vecs_raw)
        return task_vecs_aligned

    def obtain_profile_emb(self, his_id, task_embs_aligned):
        """
        task_embs_aligned: LLM-space embedding (512)
        But we use stored raw BGE-space embedding (768) for scoring against his_embs.
        """
        his_mask = torch.eq(his_id, 0)
        bsz = his_mask.size(0)
        his_mask = his_mask.repeat(1, self.mult_k).view(bsz * self.mult_k, -1)

        # Raw history embeddings from BGE table (768)
        if self.training:
            his_embs = self.his_train_emb_table(his_id)
        else:
            his_embs = self.his_dev_emb_table(his_id)

        # Align history embeddings for LLM space fusion
        his_embs_align = self.align_mlp(his_embs).view(
            bsz * self.mult_k, -1, self.llm_emb_size
        )

        # Use raw task embedding in 768-dim space for similarity scoring
        task_embs_raw = self._last_task_vecs_raw  # (B, 768)
        his_weight = torch.bmm(his_embs, task_embs_raw.unsqueeze(-1))
        his_weight = his_weight.masked_fill(his_mask.unsqueeze(-1), -torch.inf)
        his_weight = torch.nn.functional.softmax(his_weight, dim=1)
        his_weight = his_weight.to(his_embs.dtype)

        # Weighted sum in LLM space
        profile_embs = torch.bmm(
            torch.transpose(his_embs_align, 1, 2), his_weight
        ).squeeze(-1)
        return profile_embs

    def gated_cross_attention(self, task_embs, user_embs):
        attn_out, _ = self.cross_attn(task_embs, user_embs, user_embs)
        gate_val = torch.sigmoid(
            self.gate(torch.cat([task_embs, attn_out], dim=-1))
        )
        return gate_val * attn_out + (1 - gate_val) * task_embs

    def forward(
        self,
        llm_input_ids, llm_attention_mask, labels,
        emb_input_ids, emb_attention_mask, emb_token_type_ids,
        his_id,
        session_input=None, graph_ids=None
    ):
        task_embs = self.obtain_task_emb(
            emb_input_ids, emb_attention_mask, emb_token_type_ids
        )
        profile_embs = self.obtain_profile_emb(his_id, task_embs)

        combined_embs = profile_embs.unsqueeze(1)  # (B, 1, E)
        fused_task_embs = self.gated_cross_attention(
            task_embs.unsqueeze(1),
            combined_embs
        )

        # ===== INJECTION POINT =====
        # Get standard token embeddings
        inputs_embeds = self.llm_model.encoder.embed_tokens(llm_input_ids)

        # Inject fused_task_embs into the sequence — e.g., add to first token embedding
        inputs_embeds[:, 0:1, :] = inputs_embeds[:, 0:1, :] + fused_task_embs

        # Now run the LLM with modified embeddings
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_attention_mask,
            labels=labels
        )

        loss = outputs.loss
        seq_out = outputs.logits
        return [loss, seq_out]
