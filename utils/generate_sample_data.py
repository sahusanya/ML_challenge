import os
import random
import pandas as pd

def add_noise_to_name(name):
    replacements = [
        ("Corporation", "Corp"), ("Private", "Pvt"), ("Limited", "Ltd"),
        ("Company", "Co"), ("and", "&"), ("Street", "St"), ("Road", "Rd")
    ]
    if random.random() < 0.6:
        orig, rpl = random.choice(replacements)
        name = name.replace(orig, rpl)
    if random.random() < 0.2 and len(name) > 5:
        # introduce minor typo
        idx = random.randint(1, len(name) - 2)
        name = name[:idx] + name[idx+1:]
    return name

def add_noise_to_address(address):
    replacements = [
        ("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Boulevard", "Blvd")
    ]
    if random.random() < 0.6:
        orig, rpl = random.choice(replacements)
        address = address.replace(orig, rpl)
    if random.random() < 0.3:
        address += ", Near ATM"
    return address

def create_dataset(split="train", num_entities=100):
    os.makedirs(f"dataset/{split}", exist_ok=True)
    
    countries_train = ["US", "India"]
    countries_test = ["US", "India", "France"]
    countries = countries_train if split == "train" else countries_test

    names = [
        "Acme Global Logistics Corporation", "Apex Technology Solutions Private Limited",
        "Beacon Retail Outlets Company", "Cascade Financial Services Group",
        "Dynamic Power Systems Inc", "Echo Wave Communications Ltd",
        "Frontier Energy Resources", "Global Health Care Services",
        "Horizon Industrial Machinery", "Imperial Food and Beverage Products",
        "Jupiter Cloud Systems", "Krypton Cyber Security Solutions",
        "Lunar Software Systems", "Metro Urban Infrastructure",
        "Nexus Digital Media Agency", "Omega Engineering Works",
        "Pinnacle Real Estate Group", "Quantum Robotics Technologies",
        "Radiant Chemical Industries", "Starlight Entertainment Network"
    ]

    addresses = [
        "12 Main Street, Suite 400, New York, NY",
        "456 Park Avenue, Floor 12, San Jose, CA",
        "789 MG Road, Sector 5, Bengaluru, KA",
        "101 Connaught Place, Block B, New Delhi, DL",
        "202 Market Street, San Francisco, CA",
        "303 Industrial Layout, Phase 1, Hyderabad, TS",
        "404 Broadway Avenue, Suite 10, Chicago, IL",
        "505 Ring Road, Anna Nagar, Chennai, TN",
        "606 Rue de Rivoli, Paris, Ile-de-France",
        "707 Boulevard Haussmann, Paris, France"
    ]

    prefix = "S1" if split == "train" else "T1"
    
    s1_rows = []
    s2_rows = []
    s3_rows = []
    ground_truth = []

    s2_counter = 1
    s3_counter = 1

    for i in range(1, num_entities + 1):
        s1_id = f"{prefix}-{i:05d}"
        base_name = random.choice(names) + f" {i}"
        base_address = random.choice(addresses)
        country = random.choice(countries)

        s1_rows.append({
            "entity_id": s1_id,
            "business_name": base_name,
            "business_address": base_address,
            "country": country
        })

        matched_ids = []

        # 40% chance of S2 match
        if random.random() < 0.6:
            s2_id = f"{'S2' if split == 'train' else 'T2'}-{s2_counter:05d}"
            s2_counter += 1
            s2_rows.append({
                "entity_id": s2_id,
                "business_name": add_noise_to_name(base_name),
                "business_address": add_noise_to_address(base_address),
                "country": country
            })
            matched_ids.append(s2_id)

        # 30% chance of S3 match
        if random.random() < 0.5:
            s3_id = f"{'S3' if split == 'train' else 'T3'}-{s3_counter:05d}"
            s3_counter += 1
            s3_rows.append({
                "entity_id": s3_id,
                "business_name": add_noise_to_name(base_name),
                "business_address": add_noise_to_address(base_address),
                "country": country
            })
            matched_ids.append(s3_id)

        if split == "train":
            ground_truth.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched_ids)
            })

    # Add some distractor / unmatched records to S2 and S3
    for _ in range(20):
        s2_id = f"{'S2' if split == 'train' else 'T2'}-{s2_counter:05d}"
        s2_counter += 1
        s2_rows.append({
            "entity_id": s2_id,
            "business_name": "Unrelated Distractor Enterprise " + str(s2_counter),
            "business_address": "999 Distractor Highway, City",
            "country": random.choice(countries)
        })

        s3_id = f"{'S3' if split == 'train' else 'T3'}-{s3_counter:05d}"
        s3_counter += 1
        s3_rows.append({
            "entity_id": s3_id,
            "business_name": "Random Non-Matching Firm " + str(s3_counter),
            "business_address": "888 Unknown Boulevard, Town",
            "country": random.choice(countries)
        })

    # Save files with sep="\t"
    pd.DataFrame(s1_rows).to_csv(f"dataset/{split}/{split}_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows).to_csv(f"dataset/{split}/{split}_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows).to_csv(f"dataset/{split}/{split}_source3.tsv", sep="\t", index=False)
    
    if split == "train":
        pd.DataFrame(ground_truth).to_csv(f"dataset/{split}/{split}_ground_truth.tsv", sep="\t", index=False)

    print(f"Generated {split} split in dataset/{split}/: S1={len(s1_rows)}, S2={len(s2_rows)}, S3={len(s3_rows)}")

if __name__ == "__main__":
    create_dataset("train", 150)
    create_dataset("test", 100)
