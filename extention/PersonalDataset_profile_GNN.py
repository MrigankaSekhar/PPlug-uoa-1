import copy
import json
import sys
import torch
import os

class PersonalDataset:
    """
    Dataset class for Personalized LLM training with long-term history (his_id),
    short-term session context (session_ids), and graph node IDs.

    This version avoids duplicate signals by:
      - Making `session_ids` a short slice of history (last `max_session_len` items).
      - Optionally mapping `his_id` values into graph node IDs for richer GNN input.

    Attributes:
        data_file: Path to JSONL file with {"input", "output", "his_id"} records
        max_input_len: Max tokens for LLM input
        max_new_len: Max tokens for output sequence
        max_his_len: Pad/crop size for long-term history IDs
        max_session_len: Pad/crop size for short-term session IDs (default 3)
        graph_emb_path: Path to precomputed graph embeddings (.npy)
        his_to_graph_path: Path to JSON mapping his_id → graph node ID
    """
    def __init__(self, data_file, max_input_len, max_new_len, max_his_len,
                 llm_tokenizer, emb_tokenizer, graph_emb_path=None,
                 his_to_graph_path=None, max_session_len=3):

        # Store basic configuration
        self.data_file = data_file
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.max_his_len = max_his_len
        self.max_session_len = max_session_len
        self.llm_tokenizer = llm_tokenizer
        self.emb_tokenizer = emb_tokenizer
        self.graph_emb_path = graph_emb_path
        self.his_to_graph_path = his_to_graph_path

        self.llm_tokenizer.model_max_length = sys.maxsize
        self.emb_tokenizer.model_max_length = sys.maxsize

        # --- Load mapping from his_id to graph node IDs if provided ---
        self.his_to_graph = {}
        if his_to_graph_path and os.path.exists(his_to_graph_path):
            try:
                self.his_to_graph = json.load(open(his_to_graph_path))
                print(f"📂 Loaded his_to_graph mapping ({len(self.his_to_graph)} entries)")
            except Exception as e:
                print(f"⚠️ Could not load his_to_graph mapping: {e}")

        # Read all data lines from JSONL file into memory
        with open(self.data_file, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        """Number of records in the dataset."""
        return len(self.lines)

    def pad_his(self, his_ids, pad_to_len=None):
        """
        Pad or truncate a list of history IDs to a fixed length.

        Args:
            his_ids: list of integer IDs (may be shorter than pad length)
            pad_to_len: override max_his_len if provided

        Returns:
            torch.LongTensor of shape (pad_to_len,)
        """
        pad_len = pad_to_len if pad_to_len is not None else self.max_his_len
        his_ids = his_ids[:pad_len]                      # truncate
        his_ids += [0] * (pad_len - len(his_ids))        # pad with zeros
        return torch.tensor(his_ids, dtype=torch.long)

    def __getitem__(self, idx):
        """
        Build a single training sample dict for DataLoader.

        Output dict keys:
            llm_input_ids: token IDs for LLM with special tokens
            llm_attention_mask: mask for LLM input
            labels: target output token IDs
            emb_input_ids: input for embedding model (no special tokens)
            emb_attention_mask: mask for embedding model
            emb_token_type_ids: token type IDs (currently all zeros)
            his_id: padded long-term history ID list
            session_ids: padded recent history slice
            graph_node_ids: graph IDs for GNN input, from mapping if available
        """
        # Parse single data line into components
        input_str, output_str, his_id_list = self.parse_data(self.lines[idx])

        # --- Profile history IDs ---
        # Full long-term profile padded to max_his_len
        his_id = self.pad_his(his_id_list, pad_to_len=self.max_his_len)

        # --- Session IDs ---
        # Recent slice of history (short-term context)
        recent_session_ids = his_id_list[-self.max_session_len:]
        session_ids = self.pad_his(recent_session_ids, pad_to_len=self.max_session_len)

        # --- Graph node IDs ---
        # Map history IDs to graph node IDs via loaded mapping
        graph_node_ids_list = [
            self.his_to_graph[str(hid)]
            for hid in his_id_list if str(hid) in self.his_to_graph
        ]
        if not graph_node_ids_list:
            graph_node_ids_list = [0]  # fallback placeholder (node 0)
        graph_node_ids = torch.tensor(graph_node_ids_list, dtype=torch.long)

        # --- Tokenize for LLM backbone ---
        llm_encoded = self.llm_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        llm_input_ids = llm_encoded["input_ids"].squeeze(0)

        # Append special personalization tokens
        inst_id = self.llm_tokenizer.convert_tokens_to_ids("[INST_PER_TOKEN]")
        spc_id = self.llm_tokenizer.convert_tokens_to_ids("[SPC_PER_TOKEN]")
        llm_input_ids = torch.cat([llm_input_ids, torch.tensor([inst_id, spc_id], dtype=torch.long)])

        # Crop to max length after adding tokens
        if llm_input_ids.size(0) > self.max_input_len:
            llm_input_ids = llm_input_ids[:self.max_input_len]
        llm_attention_mask = torch.ones_like(llm_input_ids)

        # --- Tokenize for embedding model ---
        emb_encoded = self.emb_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )
        emb_input_ids = emb_encoded["input_ids"].squeeze(0)
        emb_attention_mask = emb_encoded["attention_mask"].squeeze(0)
        emb_token_type_ids = torch.zeros_like(emb_input_ids)

        # ✅ Crop embedding inputs too, to avoid similar warnings
        if emb_input_ids.size(0) > self.max_input_len:
            emb_input_ids = emb_input_ids[:self.max_input_len]
            emb_attention_mask = emb_attention_mask[:self.max_input_len]
            emb_token_type_ids = emb_token_type_ids[:self.max_input_len]

        return {
            "llm_input_ids": llm_input_ids,
            "llm_attention_mask": llm_attention_mask,
            "labels": self.llm_tokenizer(
                output_str,
                max_length=self.max_new_len,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(0),
            "emb_input_ids": emb_input_ids,
            "emb_attention_mask": emb_attention_mask,
            "emb_token_type_ids": emb_token_type_ids,
            "his_id": his_id,
            "session_ids": session_ids,
            "graph_node_ids": graph_node_ids
        }


    def parse_data(self, line):
        """
        Parse a JSONL line from the dataset file.

        Returns:
            input_str: truncated string for model input (pre-truncated so no warning)
            output_str: string for model output (label)
            his_id_list: raw history IDs from the record
        """
        data = json.loads(line)

        # ✅ Tokenize once with truncation and no unnecessary decode/encode cycle
        token_ids = self.llm_tokenizer.encode(
            data["input"],
            truncation=True,
            max_length=self.max_input_len
        )

        # Directly decode truncated IDs into string for downstream tokenizers
        input_str = self.llm_tokenizer.decode(token_ids, skip_special_tokens=True)

        output_str = data["output"]
        his_id_list = data["his_id"]

        return input_str, output_str, his_id_list