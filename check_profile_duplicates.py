import json
from collections import defaultdict

def collect_profile_ids(filename, split_name):
    with open(filename) as f:
        data = json.load(f)
    id_to_questions = defaultdict(list)
    for q_idx, q in enumerate(data):
        for profile in q['profile']:
            id_to_questions[profile['id']].append((split_name, q_idx))
    return id_to_questions

# Collect IDs from both dev and train splits
dev_ids = collect_profile_ids('LaMP_time_3_subset/dev_questions.json', 'dev')
train_ids = collect_profile_ids('LaMP_time_3_subset/train_questions.json', 'train')

# Merge both dictionaries
all_ids = defaultdict(list)
for rid, qs in dev_ids.items():
    all_ids[rid].extend(qs)
for rid, qs in train_ids.items():
    all_ids[rid].extend(qs)

# Find duplicates across splits and questions
duplicates = {rid: qs for rid, qs in all_ids.items() if len(qs) > 1}
if duplicates:
    print("Duplicate review IDs found across questions and/or splits:")
    for rid, qs in duplicates.items():
        locations = ', '.join([f"{split}[{q_idx}]" for split, q_idx in qs])
        print(f"Review ID {rid} appears in: {locations}")
else:
    print("No duplicate review IDs across dev and train questions.")