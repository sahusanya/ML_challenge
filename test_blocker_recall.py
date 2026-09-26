import pandas as pd
from src.scalable_blocking import ScalableBlocker


S2_FILE = "dataset/blocker_test/source2.tsv"
S3_FILE = "dataset/blocker_test/source3.tsv"

GT_FILE = "../student_resource/dataset/train/train_ground_truth.tsv"
S1_FILE = "../student_resource/dataset/train/train_source1.tsv"


# ---------------------------------------------------------
# 1. Get IDs that are actually inside our 200k test index
# ---------------------------------------------------------

print("Loading indexed candidate IDs...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

indexed_ids = set(s2["entity_id"])
indexed_ids.update(s3["entity_id"])

print("Indexed IDs:", len(indexed_ids))


# ---------------------------------------------------------
# 2. Find S1 entities whose true matches are in our index
# ---------------------------------------------------------

print("Scanning ground truth...")

valid_s1 = []

for chunk in pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    for _, row in chunk.iterrows():

        matched = str(row["matched_entity_ids"])

        if not matched or matched == "nan":
            continue

        true_matches = set(matched.split(","))

        # Keep entities where at least one true match
        # exists inside our 200k indexed records.
        indexed_matches = true_matches.intersection(indexed_ids)

        if indexed_matches:
            valid_s1.append(
                (
                    row["source1_entity_id"],
                    indexed_matches,
                    true_matches
                )
            )

        if len(valid_s1) >= 100:
            break

    if len(valid_s1) >= 100:
        break


print("Found test S1 entities:", len(valid_s1))


# ---------------------------------------------------------
# 3. Load the corresponding S1 records
# ---------------------------------------------------------

target_s1_ids = {x[0] for x in valid_s1}

s1_records = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    matches = chunk[
        chunk["entity_id"].isin(target_s1_ids)
    ]

    for _, row in matches.iterrows():
        s1_records[row["entity_id"]] = row

    if len(s1_records) >= len(target_s1_ids):
        break


# ---------------------------------------------------------
# 4. Query blocker
# ---------------------------------------------------------

blocker = ScalableBlocker(
    "models/blocker_test.db"
)


total_true = 0
total_found = 0

print("\nRunning blocker recall test...\n")


for s1_id, indexed_matches, all_true_matches in valid_s1:

    row = s1_records.get(s1_id)

    if row is None:
        continue

    candidates = blocker.get_candidates(
        business_name=row["business_name"],
        business_address=row["business_address"],
        country=row["country"]
    )

    candidate_ids = {
        candidate[0]
        for candidate in candidates
    }

    found = indexed_matches.intersection(candidate_ids)

    # True matches that the blocker failed to retrieve
    missed = indexed_matches - candidate_ids

    total_true += len(indexed_matches)
    total_found += len(found)

    print("=" * 70)
    print("S1:", s1_id)
    print("Name:", row["business_name"])
    print("Address:", row["business_address"])
    print("Country:", row["country"])

    print("True matches in index:")
    print(indexed_matches)

    print("Found by blocker:")
    print(found)

    print("MISSED:")
    print(missed)

    print(
        f"Recall: {len(found)}/{len(indexed_matches)}"
    )
    print("Name:", row["business_name"])

    print("True matches in index:")
    print(indexed_matches)

    print("Found by blocker:")
    print(found)

    print(
        f"Recall: {len(found)}/{len(indexed_matches)}"
    )


# ---------------------------------------------------------
# 5. Overall recall
# ---------------------------------------------------------

if total_true > 0:

    recall = total_found / total_true

    print("\n" + "=" * 70)
    print("OVERALL BLOCKER RECALL")
    print("=" * 70)

    print("True matches:", total_true)
    print("Found:", total_found)
    print(f"Recall: {recall:.4f}")