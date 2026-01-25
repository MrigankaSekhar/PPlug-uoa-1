import json

def build_global_idmap(train_path, dev_path, output_path):
    def get_review_ids(path):
        with open(path) as f:
            data = json.load(f)
        ids = set()
        for entry in data:
            for his in entry.get("profile", []):
                rid = str(his.get("id"))
                if rid:
                    ids.add(rid)
        return ids

    train_ids = get_review_ids(train_path)
    dev_ids = get_review_ids(dev_path)
    all_ids = sorted(train_ids | dev_ids)
    idmap = {rid: idx for idx, rid in enumerate(all_ids)}
    with open(output_path, "w") as f:
        json.dump(idmap, f, indent=2)
    print(f"✅ Saved global idmap: {output_path} ({len(idmap)} entries)")

if __name__ == "__main__":
    build_global_idmap(
        "../LaMP_time_3_subset/train_questions.json",
        "../LaMP_time_3_subset/dev_questions.json",
        "../bge_emb/task_3_global_bge_idmap.json"
    )