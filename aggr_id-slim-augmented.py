import os
import json
import copy
import random

# Paths for augmented files
USE_SUBSET = True
idx = 3  # Only process LaMP-3

dir_name = f"LaMP_time_{idx}"
subset_dir = f"{dir_name}_subset"
data_dir = subset_dir if USE_SUBSET else dir_name
id_dir = data_dir + "_id"

os.makedirs(id_dir, exist_ok=True)

# --- TRAIN ---
train_questions_file = os.path.join(data_dir, "train_questions_augmented.json")
train_outputs_file = os.path.join(data_dir, "train_outputs_augmented.json")
train_out_file = os.path.join(id_dir, "train_aug_input.json")

print(f"Using TRAIN questions from: {train_questions_file}")
print(f"Using TRAIN outputs from: {train_outputs_file}")

with open(train_questions_file, "r") as f:
    train_questions = json.load(f)
with open(train_outputs_file, "r") as f:
    train_outputs = json.load(f)["golds"]

# Build a mapping from id to output
output_map = {entry["id"]: entry["output"] for entry in train_outputs}

datas = []
for entry in train_questions:
    output_entry = {}
    output_entry["input"] = entry["input"]
    output_entry["his_id"] = copy.deepcopy(entry.get("his_id", []))  # <-- robust to missing field
    output_entry["id"] = entry["id"]
    # Attach output if available
    if entry["id"] in output_map:
        output_entry["output"] = output_map[entry["id"]]
    datas.append(output_entry)

with open(train_out_file, "w") as f_out:
    for data in datas:
        if len(data["his_id"]) > 2:
            for _ in range(10):
                del_n = int(len(data["his_id"]) * random.uniform(0, 0.5)) + 1
                new_his = random.sample(data["his_id"], len(data["his_id"]) - del_n)
                new_data = copy.deepcopy(data)
                new_data["his_id"] = new_his
                f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

# --- DEV ---
dev_questions_file = os.path.join(data_dir, "dev_questions_augmented.json")
dev_outputs_file = os.path.join(data_dir, "dev_outputs_augmented.json")
dev_out_file = os.path.join(id_dir, "dev_profile.json")

print(f"Using DEV questions from: {dev_questions_file}")
print(f"Using DEV outputs from: {dev_outputs_file}")

with open(dev_questions_file, "r") as f:
    dev_questions = json.load(f)
with open(dev_outputs_file, "r") as f:
    dev_outputs = json.load(f)["golds"]

output_map_dev = {entry["id"]: entry["output"] for entry in dev_outputs}

# --- NEW: Load train his_ids for extra safety ---
train_questions_file = os.path.join(data_dir, "train_questions_augmented.json")
with open(train_questions_file, "r") as f:
    train_questions = json.load(f)
train_his_ids = set()
for entry in train_questions:
    train_his_ids.update(entry.get("his_id", []))

datas_dev = []
for entry in dev_questions:
    # Skip if any his_id overlaps with train
    if any(hid in train_his_ids for hid in entry.get("his_id", [])):
        continue
    output_entry = {}
    output_entry["input"] = entry["input"]
    output_entry["his_id"] = copy.deepcopy(entry.get("his_id", []))
    output_entry["id"] = entry["id"]
    if entry["id"] in output_map_dev:
        output_entry["output"] = output_map_dev[entry["id"]]
    datas_dev.append(output_entry)

with open(dev_out_file, "w") as f_out:
    for data in datas_dev:
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
