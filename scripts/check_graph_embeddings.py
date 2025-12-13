import torch
import json

emb = torch.load("../graph_emb/task_3_graph.emb")
print(emb.shape)

def load_json_lines(path):
    with open(path) as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError:
            f.seek(0)
            return json.load(f)

max_id = -1
for split in ["../LaMP_time_3_id/train_aug_input.json", "../LaMP_time_3_id/dev_profile.json"]:
    data = load_json_lines(split)
    for sample in data:
        ids = sample.get("his_id", [])
        # flatten if nested lists
        if ids and isinstance(ids[0], list):
            ids = [i for sub in ids for i in sub]
        if ids:
            max_id = max(max_id, max(ids))
print("Max his_id in dataset:", max_id)