import pandas as pd
from src.preprocess import normalize_text

S2_FILE = "dataset/blocker_test/source2.tsv"
S3_FILE = "dataset/blocker_test/source3.tsv"

cases = {
    "S1-432931957": {
        "name": "Internal Medicine Signature Associates L.L.C.",
        "address": "2780 Rock Creek Road, Creston, NC",
        "true": ["S3-284924975", "S2-958761019"],
    },
    "S1-152222475": {
        "name": "Ram Maa Logistics Private Limited",
        "address": "G-1, Floor-Grd, One Avighna Park, Mahadeo Palav Marg, Curry Road Parel, Mumbai, Maharashtra",
        "true": ["S2-449137844"],
    },
    "S1-518197819": {
        "name": "Primary Care Clinic LLC",
        "address": "351 11th Street, Washington, DC",
        "true": ["S3-14411598"],
    },
    "S1-72444401": {
        "name": "New Solutions",
        "address": "113/154, 1St Floor, Swaroop Nagar, Swarup Nagar, Kanpur Nagar, Uttar Pradesh",
        "true": ["S3-505153321"],
    },
    "S1-116043204": {
        "name": "Red Consultants Pvt. Ltd.",
        "address": "C/O Tapas Kumar Betal, Bhogpur, Purba Medinipur, Panskura, East Midnapore, West Bengal",
        "true": ["S3-85523430"],
    },
    "S1-314714647": {
        "name": "Seabird (India) Projects-Lucknow",
        "address": "Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh",
        "true": ["S3-192802051"],
    },
    "S1-282467635": {
        "name": "Prem & Sons Pvt Ltd",
        "address": "D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka",
        "true": ["S3-138350041"],
    },
    "S1-789009573": {
        "name": "Hotel Enterprises Limited",
        "address": "Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi",
        "true": ["S2-383871912"],
    },
    "S1-55344266": {
        "name": "Raj Investments LLP",
        "address": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
        "true": ["S3-384364074", "S3-478195123"],
    },
}

target_ids = set()

for case in cases.values():
    target_ids.update(case["true"])

records = {}

for filepath, source in [
    (S2_FILE, "S2"),
    (S3_FILE, "S3"),
]:
    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        matches = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        for _, row in matches.iterrows():
            records[row["entity_id"]] = {
                "source": source,
                "name": row["business_name"],
                "address": row["business_address"],
            }

        if len(records) == len(target_ids):
            break


def tokens(text):
    return set(
        normalize_text(text).split()
    )


print("\n" + "=" * 100)
print("REMAINING MISSED MATCH ANALYSIS")
print("=" * 100)

for s1_id, case in cases.items():

    s1_name_tokens = tokens(case["name"])
    s1_address_tokens = tokens(case["address"])

    print("\n" + "-" * 100)
    print("S1:", s1_id)
    print("NAME:", case["name"])
    print("ADDRESS:", case["address"])

    for true_id in case["true"]:

        record = records.get(true_id)

        if not record:
            print("\n", true_id, "NOT FOUND")
            continue

        candidate_name_tokens = tokens(
            record["name"]
        )

        candidate_address_tokens = tokens(
            record["address"]
        )

        name_overlap = (
            s1_name_tokens &
            candidate_name_tokens
        )

        address_overlap = (
            s1_address_tokens &
            candidate_address_tokens
        )

        print("\nTRUE MATCH:", true_id)
        print("SOURCE:", record["source"])
        print("NAME:", record["name"])
        print("ADDRESS:", record["address"])

        print(
            "NAME TOKEN OVERLAP:",
            sorted(name_overlap)
        )

        print(
            "ADDRESS TOKEN OVERLAP:",
            sorted(address_overlap)
        )
