
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os

class PersonalLLM_Slim(nn.Module):
    """
    Slim academic variant of PersonalLLM that:
    - Keeps LLM frozen
    - Adds Session-Aware Transformer (NEW Layer SIX)
    - Adds Graph Embedding table (NEW Layer SEVEN)
    - Adds Gated Cross-Attention fusion (NEW Layer EIGHT)
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

        # Layer TWO: User Behavior Encoder (long-term history embeddings)
        his_train_emb = torch.cat([torch.zeros(1, self.emb_emb_size),
                                   torch.load(f"../bge_emb/task_{task_id}_train_bge.emb")], dim=0)
        self.his_train_emb_table = nn.Embedding(his_train_emb.size(0), self.emb_emb_size)
        self.his_train_emb_table.weight = nn.Parameter(his_train_emb)
        for _, p in self.his_train_emb_table.named_parameters():
            p.requires_grad = False

        his_dev_emb = torch.cat([torch.zeros(1, self.emb_emb_size),
                                 torch.load(f"../bge_emb/task_{task_id}_dev_bge.emb")], dim=0)
        self.his_dev_emb_table = nn.Embedding(his_dev_emb.size(0), self.emb_emb_size)
        self.his_dev_emb_table.weight = nn.Parameter(his_dev_emb)
        for _, p in self.his_dev_emb_table.named_parameters():
            p.requires_grad = False

        # NEW Layer SEVEN: Graph Embedding Encoder (precomputed offline)
        try:
            graph_emb = torch.cat([torch.zeros(1, self.emb_emb_size),
                                   torch.load(f"../graph_emb/task_{task_id}_graph.emb")], dim=0)
            self.graph_emb_table = nn.Embedding(graph_emb.size(0), self.emb_emb_size)
            self.graph_emb_table.weight = nn.Parameter(graph_emb)
            for _, p in self.graph_emb_table.named_parameters():
                p.requires_grad = False
        except FileNotFoundError:
            self.graph_emb_table = None
            print("⚠ Graph embeddings not found — skipping")

        # Layer FOUR: Instruction Token Embedding
        self.inst_token = nn.Parameter(torch.rand(self.emb_emb_size), requires_grad=True)

        # Freeze LLM weights
        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False

        self.mult_k = 1
        self.align_mlp_inst = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size * self.mult_k),
            nn.GELU(),
            nn.Linear(self.llm_emb_size * self.mult_k, self.llm_emb_size * self.mult_k)
        )
        self.align_mlp = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size * self.mult_k),
            nn.GELU(),
            nn.Linear(self.llm_emb_size * self.mult_k, self.llm_emb_size * self.mult_k)
        )

        # NEW Layer SIX: Session-Aware Transformer
        self.session_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_emb_size, nhead=4, batch_first=True),
            num_layers=2
        )

        # NEW Layer EIGHT: Gated Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.llm_emb_size, num_heads=8, batch_first=True
        )
        self.gate = nn.Linear(self.llm_emb_size * 2, 1)

    # Layer ONE: Task embedding extraction
    def obtain_task_emb(self, emb_input_ids, emb_attention_mask, emb_token_type_ids):
        task_inputs = {
            'input_ids': emb_input_ids,
            'attention_mask': emb_attention_mask,
            'token_type_ids': emb_token_type_ids
        }
        task_outputs = self.emb_model(**task_inputs)
        return torch.nn.functional.normalize(task_outputs[0][:, 0], p=2, dim=1)

    # Layer THREE: Input-aware Personal Aggregator
    def obtain_profile_emb(self, his_id, task_embs):
        his_mask = torch.eq(his_id, 0)
        bsz = his_mask.size(0)
        his_mask = his_mask.repeat(1, self.mult_k).view(bsz * self.mult_k, -1)

        his_embs = self.his_train_emb_table(his_id) if self.training else self.his_dev_emb_table(his_id)
        his_embs_align = self.align_mlp(his_embs).view(bsz * self.mult_k, -1, self.llm_emb_size)

        his_weight = torch.bmm(his_embs, task_embs.unsqueeze(-1))
        his_weight = his_weight.masked_fill(his_mask.unsqueeze(-1), -torch.inf)
        his_weight = torch.nn.functional.softmax(his_weight, dim=1)
        profile_embs = torch.bmm(his_embs_align.transpose(1, 2), his_weight).squeeze(-1)
        return profile_embs

    # NEW Layer EIGHT: Gated Cross-Attention fusion
    def gated_cross_attention(self, task_embs, user_embs):
        attn_out, _ = self.cross_attn(task_embs, user_embs, user_embs)
        gate_val = torch.sigmoid(self.gate(torch.cat([task_embs, attn_out], dim=-1)))
        return gate_val * attn_out + (1 - gate_val) * task_embs

    def forward(
        self,
        llm_input_ids, llm_attention_mask, labels,
        emb_input_ids, emb_attention_mask, emb_token_type_ids,
        his_id,
        session_ids=None, graph_node_ids=None
    ):
        # Layer ONE
        task_embs = self.obtain_task_emb(emb_input_ids, emb_attention_mask, emb_token_type_ids)

        # Layer THREE (depends on Layer TWO)
        profile_embs = self.obtain_profile_emb(his_id, task_embs)

        # NEW Layer SIX: Session embeddings from Transformer
        session_embs = None
        if session_ids is not None:
            session_tbl = self.his_train_emb_table if self.training else self.his_dev_emb_table
            session_vecs = session_tbl(session_ids)
            session_embs = self.session_encoder(session_vecs)[:, -1, :]  # last step summary

        # NEW Layer SEVEN: Graph embeddings lookup
        graph_embs = None
        if self.graph_emb_table is not None and graph_node_ids is not None:
            graph_embs = self.graph_emb_table(graph_node_ids)

        # Combine personalization signals
        user_embs_list = [profile_embs.unsqueeze(1)]
        if session_embs is not None:
            user_embs_list.append(session_embs.unsqueeze(1))
        if graph_embs is not None:
            user_embs_list.append(graph_embs)
        combined_user_embs = torch.cat(user_embs_list, dim=1)

        # NEW Layer EIGHT: Gated Cross-Attention fusion
        fused_task_embs = self.gated_cross_attention(task_embs.unsqueeze(1), combined_user_embs)

        # Layer FOUR + Layer ONE injection point
        input_embs = self.llm_model.get_input_embeddings()(llm_input_ids)
        input_embs[llm_input_ids == self.llm_model.vocab_size-1] = fused_task_embs.view(-1, self.llm_emb_size)
        input_embs[llm_input_ids == self.llm_model.vocab_size-2] = self.align_mlp_inst(self.inst_token)

        # Layer FIVE: LLM Decoder/Generator
        if self.training:
            reader_output = self.llm_model(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask.long(),
                labels=labels.long(),
                use_cache=False,
                return_dict=True
            )
            return SequenceClassifierOutput(loss=reader_output.loss)
        else:
            reader_output = self.llm_model.generate(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask,
                max_new_tokens=self.max_new_len,
                num_beams=4,
                num_return_sequences=1,
                return_dict_in_generate=True,
                do_sample=False
            )
            seq_out = reader_output['sequences']
            loss = torch.FloatTensor([0.0]).to(llm_input_ids.device)
            return [loss, seq_out]
