
import json
import os
import numpy as np
from collections import Counter, defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def basic_stats(questions, outputs=None):
    stats = {}
    users = set()
    items = set()
    categories = Counter()
    ratings_by_user = defaultdict(list)
    profile_lens = []

    for q in questions:
        uid = q.get("user_id") or f"user_{q.get('id')}"
        users.add(uid)

        # Items
        item_key = q.get("item_id") or q.get("item_title") or f"item_{q.get('id')}"
        items.add(item_key)

        # Categories
        if "category" in q and q["category"] is not None:
            categories[str(q["category"])] += 1

        # Ratings
        if "rating" in q:
            ratings_by_user[uid].append(q["rating"])

        # Profile lengths
        if "profile" in q and isinstance(q["profile"], list):
            profile_lens.append(len(q["profile"]))

    stats["num_questions"] = len(questions)
    stats["num_users"] = len(users)
    stats["num_items"] = len(items)
    stats["num_categories"] = len(categories)
    stats["top_categories"] = categories.most_common(5)
    stats["mean_profile_len"] = np.mean(profile_lens) if profile_lens else 0
    stats["p90_profile_len"] = np.percentile(profile_lens, 90) if profile_lens else 0

    if ratings_by_user:
        # Average rating stats
        avg_ratings = [np.mean(r) for r in ratings_by_user.values()]
        stats["mean_avg_rating_per_user"] = float(np.mean(avg_ratings))
        stats["p90_avg_rating_per_user"] = float(np.percentile(avg_ratings, 90))

        # NEW: Rating count per user stats
        rating_counts = [len(r) for r in ratings_by_user.values()]
        stats["mean_rating_count_per_user"] = float(np.mean(rating_counts))
        stats["p90_rating_count_per_user"] = float(np.percentile(rating_counts, 90))
        stats["max_rating_count_per_user"] = int(np.max(rating_counts))
    else:
        stats["mean_avg_rating_per_user"] = None
        stats["p90_avg_rating_per_user"] = None
        stats["mean_rating_count_per_user"] = None
        stats["p90_rating_count_per_user"] = None
        stats["max_rating_count_per_user"] = None

    # Output-driven stats
    if outputs and isinstance(outputs, list):
        if all(isinstance(o, dict) and "output" in o for o in outputs):
            lengths = [len(str(o["output"]).split()) for o in outputs]
            stats["mean_output_length"] = np.mean(lengths)
            stats["p90_output_length"] = np.percentile(lengths, 90)
        else:
            stats["output_unique_values"] = len(set(map(str, outputs)))
    return stats

def compare_splits(train_q, dev_q):
    """Compare overlap between train and dev"""
    train_users = {q.get("user_id") for q in train_q}
    dev_users = {q.get("user_id") for q in dev_q}

    train_items = {q.get("item_id") for q in train_q}
    dev_items = {q.get("item_id") for q in dev_q}

    train_cats = {str(q.get("category")) for q in train_q if q.get("category")}
    dev_cats = {str(q.get("category")) for q in dev_q if q.get("category")}

    overlap = {
        "user_overlap_pct": (len(train_users & dev_users) / len(dev_users) * 100) if dev_users else 0.0,
        "item_overlap_pct": (len(train_items & dev_items) / len(dev_items) * 100) if dev_items else 0.0,
        "category_overlap_pct": (
            len(train_cats & dev_cats) / len(dev_cats) * 100
        ) if dev_cats else 0.0
    }
    return overlap

if __name__ == "__main__":
    base_dir = "../LaMP_time_3_subset"
    train_questions = load_json(os.path.join(base_dir,"train_questions.json"))
    dev_questions = load_json(os.path.join(base_dir,"dev_questions.json"))
    train_outputs = load_json(os.path.join(base_dir,"train_outputs.json"))
    dev_outputs = load_json(os.path.join(base_dir,"dev_outputs.json"))

    print("\n==== TRAIN STATS ====")
    train_stats = basic_stats(train_questions, train_outputs)
    for k,v in train_stats.items():
        print(f"{k}: {v}")

    print("\n==== DEV STATS ====")
    dev_stats = basic_stats(dev_questions, dev_outputs)
    for k,v in dev_stats.items():
        print(f"{k}: {v}")

    print("\n==== CROSS-SPLIT OVERLAP ====")
    overlap_stats = compare_splits(train_questions, dev_questions)
    for k,v in overlap_stats.items():
        print(f"{k}: {v:.2f}%")
