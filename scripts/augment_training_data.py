import json

def make_outputs_file(original_outputs_path, augmented_data, out_path):
    # Load original outputs
    with open(original_outputs_path) as f:
        orig_golds = json.load(f)["golds"]
    golds = list(orig_golds)  # Start with originals

    # Add augmented outputs
    for entry in augmented_data:
        if "output" in entry:
            golds.append({"id": entry["id"], "output": entry["output"]})
    with open(out_path, "w") as f:
        json.dump({"golds": golds}, f)

def augment_data(data, split_name):
    augmented_data = []
    filtered_data = []
    for entry in data:
        # Try to get his_id directly, or extract from profile
        his_ids = entry.get("his_id")
        if not his_ids:
            # If not present, try to extract from profile
            his_ids = [str(h.get("id")) for h in entry.get("profile", []) if "id" in h]
        # Only proceed if his_ids is non-empty
        if not his_ids:
            continue  # Skip entries with no history

        # Add the original entry, but with his_id field
        orig_entry = dict(entry)
        orig_entry["his_id"] = his_ids
        filtered_data.append(orig_entry)

        # For each question type, add both "yes" and "no" examples
        question_templates = [
            ("sentiment", "Does this user tend to rate positive reviews higher than negative ones?"),
            ("popularity", "Does this user tend to rate popular items higher than less popular ones?"),
            ("purchase", f"Would this user purchase this item? Review: {entry.get('input', '')}"),
            ("similarity", f"Is the following review similar to those the user has written before? review: {entry.get('input', '')}")
        ]
        for qtype, template in question_templates:
            for label in ["yes", "no"]:
                augmented_data.append({
                    "input": template,
                    "his_id": his_ids,
                    "id": f"{entry['id']}_{qtype}_{label}",
                    "output": label
                })
    return filtered_data + augmented_data

def collect_all_his_ids(data):
    all_his = set()
    for entry in data:
        his_ids = entry.get("his_id")
        if not his_ids:
            his_ids = [str(h.get("id")) for h in entry.get("profile", []) if "id" in h]
        all_his.update(his_ids)
    return all_his

# --- Train ---
with open("../LaMP_time_3_subset/train_questions.json") as f:
    train_data = json.load(f)
train_augmented = augment_data(train_data, "train")
with open("../LaMP_time_3_subset/train_questions_augmented.json", "w") as f:
    json.dump(train_augmented, f)
make_outputs_file("../LaMP_time_3_subset/train_outputs.json", train_augmented, "../LaMP_time_3_subset/train_outputs_augmented.json")

# --- Dev ---
with open("../LaMP_time_3_subset/dev_questions.json") as f:
    dev_data = json.load(f)

# --- NEW: Remove dev entries with any his_id in train ---
train_his_ids = collect_all_his_ids(train_augmented)
def dev_entry_has_overlap(entry):
    his_ids = entry.get("his_id")
    if not his_ids:
        his_ids = [str(h.get("id")) for h in entry.get("profile", []) if "id" in h]
    return any(hid in train_his_ids for hid in his_ids)

dev_augmented_raw = augment_data(dev_data, "dev")
dev_augmented = [entry for entry in dev_augmented_raw if not dev_entry_has_overlap(entry)]

with open("../LaMP_time_3_subset/dev_questions_augmented.json", "w") as f:
    json.dump(dev_augmented, f)
make_outputs_file("../LaMP_time_3_subset/dev_outputs.json", dev_augmented, "../LaMP_time_3_subset/dev_outputs_augmented.json")