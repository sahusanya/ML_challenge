import pandas as pd
from src.scalable_blocking import ScalableBlocker

S2_FILE = "dataset/blocker_test/source2.tsv"
S3_FILE = "dataset/blocker_test/source3.tsv"

# Missed true matches from the latest 5M blocker test
missed_ids = {
    "S3-478195123",
    "S3-384364074",
    "S3-804600254",
    "S2-660036492",
    "S3-274817120",
    "S2-383871912",
    "S3-138350041",
    "S3-204655096",
    "S2-23141904",
    "S3-82080725",
}

# Corresponding S1 records
s1_info = {
    "S1-55344266": {
        "name": "Raj Investments LLP",
        "address": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
        "country": "India",
    },
    "S1-318373630": {
        "name": "Red Ventures Private Limited",
        "address": "Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1",
        "country": "India",
    },
    "S1-86989137": {
        "name": "Laxmi Golden Investments Private Limited",
        "address": "New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka",
        "country": "India",
    },
    "S1-789009573": {
        "name": "Hotel Enterprises Limited",
        "address": "Delhi",
        "country": "India",
    },
    "S1-282467635": {
        "name": "Prem & Sons Pvt Ltd",
        "address": "D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka",
        "country": "India",
    },
    "S1-727602285": {
        "name": "Swastik Om Solutions LLP",
        "address": "201/D, Aditya, Svp Nagar Andheri (W), Mumbai, Mumbai City, Maharashtra",
        "country": "India",
    },
    "S1-502736054": {
        "name": "First Seven Exports Pvt Ltd",
        "address": "Maharashtra, At Shahagad Tq. Ambad, Jalna",
        "country": "India",
    },
}

blocker = ScalableBlocker("models/blocker_test.db")

print("\nSearching for missed records...\n")

found = {}

for source_file, source_name in [
    (S2_FILE, "S2"),
    (S3_FILE, "S3"),
]:
    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        matches = chunk[chunk["entity_id"].isin(missed_ids)]

        for _, row in matches.iterrows():
            found[row["entity_id"]] = {
                "source": source_name,
                "entity_id": row["entity_id"],
                "name": row["business_name"],
                "address": row["business_address"],
                "country": row["country"],
            }

        if len(found) == len(missed_ids):
            break

print("=" * 100)
print("MISSED TRUE MATCH RECORDS")
print("=" * 100)

for entity_id, record in found.items():
    print("\n" + "-" * 100)
    print("ID:", entity_id)
    print("SOURCE:", record["source"])
    print("NAME:", record["name"])
    print("ADDRESS:", record["address"])
    print("COUNTRY:", record["country"])

print("\n" + "=" * 100)
print("BLOCKER CANDIDATES FOR RELATED S1s")
print("=" * 100)

for s1_id, s1 in s1_info.items():
    candidates = blocker.get_candidates(
        business_name=s1["name"],
        business_address=s1["address"],
        country=s1["country"],
    )

    candidate_ids = {x[0] for x in candidates}

    relevant = [
        entity_id
        for entity_id in missed_ids
        if entity_id in candidate_ids
    ]

    print("\nS1:", s1_id)
    print("NAME:", s1["name"])
    print("MISSED TRUE IDS FOUND NOW:", relevant)

    if relevant:
        print("Candidate position:")
        for i, candidate in enumerate(candidates, 1):
            if candidate[0] in relevant:
                print("  ", i, candidate)
