import pandas as pd

from src.scalable_blocking import ScalableBlocker


# Take 10 real Source 1 records
s1 = pd.read_csv(
    "../student_resource/dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=10
)

blocker = ScalableBlocker(
    "models/blocker_test.db"
)


for _, row in s1.iterrows():

    candidates = blocker.get_candidates(
        business_name=row["business_name"],
        business_address=row["business_address"],
        country=row["country"]
    )

    print("\n" + "=" * 70)

    print("S1:", row["entity_id"])
    print("Name:", row["business_name"])
    print("Address:", row["business_address"])
    print("Country:", row["country"])

    print("\nCandidate count:", len(candidates))

    print("First 10 candidates:")

    for candidate in candidates[:10]:
        print("  ", candidate)