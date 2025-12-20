import copy
import json
import torch

class PersonalDataset:
    def __init__(self, data_file, max_input_len, max_new_len, max_his_len,
                 llm_tokenizer, emb_tokenizer, graph_emb_path=None):
        self.data_file = data_file
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.emb_tokenizer = emb_tokenizer
        self.graph_emb_path = graph_emb_path

        # Load data lines
        with open(self.data_file, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def pad_his(self, his_ids):
        his_ids = his_ids[:self.max_his_len]
        his_ids += [0] * (self.max_his_len - len(his_ids))
        return torch.tensor(his_ids, dtype=torch.long)

    def __getitem__(self, idx):
        input_str, output_str, his_id = self.parse_data(self.lines[idx])

        # Tokenize for LLM with special tokens inserted
        llm_encoded = self.llm_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )

        llm_input_ids = llm_encoded["input_ids"].squeeze(0)

        # Always append [INST_PER_TOKEN] and [SPC_PER_TOKEN]
        inst_id = self.llm_tokenizer.convert_tokens_to_ids("[INST_PER_TOKEN]")
        spc_id = self.llm_tokenizer.convert_tokens_to_ids("[SPC_PER_TOKEN]")
        llm_input_ids = torch.cat([llm_input_ids, torch.tensor([inst_id, spc_id], dtype=torch.long)])

        # Ensure within max length after adding
        if len(llm_input_ids) > self.max_input_len:
            llm_input_ids = llm_input_ids[:self.max_input_len]

        llm_attention_mask = torch.ones_like(llm_input_ids)

        # Tokenize for embeddings
        emb_encoded = self.emb_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )

        emb_input_ids = emb_encoded["input_ids"].squeeze(0)
        emb_attention_mask = emb_encoded["attention_mask"].squeeze(0)
        emb_token_type_ids = torch.zeros_like(emb_input_ids)

        sample = {
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
            "session_ids": his_id,      # for now same as his_id if no separate session data
            "graph_node_ids": torch.tensor([0], dtype=torch.long)  # placeholder if none
        }

        return sample

    def parse_data(self, line):
        data = json.loads(line)
        input_str = copy.deepcopy(data["input"])
        input_str = self.llm_tokenizer.decode(
            self.llm_tokenizer.encode(input_str)[:self.max_input_len][:-1]
        )
        output_str = data["output"]
        his_id = self.pad_his(data["his_id"])
        return input_str, output_str, his_id