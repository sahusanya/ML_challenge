import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from src.feature_engineering import PairFeatureExtractor
from src.preprocess import normalize_text
from src.scalable_blocking import ScalableBlocker


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "models/blocker_test.db"

S1_FILE = "../student_resource/dataset/train/train_source1.tsv"
GT_FILE = "../student_resource/dataset/train/train_ground_truth.tsv"

MODEL_PATH = "models/real_er_model.pkl"

# Small first experiment.
# We can increase these after confirming the pipeline works.
N_MATCHED_S1 = 150
N_SINGLETON_S1 = 50

NAME_LIMIT = 50
ADDRESS_LIMIT = 50


# ============================================================
# F0.5
# ============================================================

def compute_f05(gt_set, pred_set):

    if not gt_set and not pred_set:
        return 1.0

    if not gt_set or not pred_set:
        return 0.0

    tp = len(gt_set.intersection(pred_set))

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    if precision + recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def evaluate_f05(df_pairs, probs, threshold, gt_dict):

    temp = df_pairs.copy()
    temp["prob"] = probs

    matched = temp[temp["prob"] >= threshold]

    pred_dict = (
        matched
        .groupby("source1_entity_id")["candidate_key"]
        .apply(set)
        .to_dict()
    )

    scores = []

    for s1_id, gt in gt_dict.items():

        pred = pred_dict.get(
            s1_id,
            set()
        )

        scores.append(
            compute_f05(gt, pred)
        )

    return float(np.mean(scores))


# ============================================================
# START
# ============================================================

print("==============================================")
print("FAST REAL ENTITY RESOLUTION TRAINING")
print("==============================================")


# ============================================================
# 1. LOAD MATCHED TRAINING S1 ENTITIES
# ============================================================

print("\nLoading ground truth...")

selected_gt = {}

for chunk in pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    chunk = chunk.fillna("")

    for _, row in chunk.iterrows():

        s1_id = row["source1_entity_id"]

        matched = row[
            "matched_entity_ids"
        ].strip()

        matches = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }

        if matches:

            selected_gt[s1_id] = matches

        if len(selected_gt) >= N_MATCHED_S1:
            break

    if len(selected_gt) >= N_MATCHED_S1:
        break


# ============================================================
# 2. LOAD SINGLETON EXAMPLES
# ============================================================

print("Loading singleton examples...")

singletons = {}

for chunk in pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    chunk = chunk.fillna("")

    for _, row in chunk.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        matched = row[
            "matched_entity_ids"
        ].strip()

        if not matched:

            singletons[s1_id] = set()

        if len(singletons) >= N_SINGLETON_S1:
            break

    if len(singletons) >= N_SINGLETON_S1:
        break


# Don't accidentally replace matched examples
# with singleton entries.

for s1_id, matches in singletons.items():

    if s1_id not in selected_gt:

        selected_gt[s1_id] = matches


target_ids = set(selected_gt)

print(
    "Selected S1 entities:",
    len(target_ids)
)


# ============================================================
# 3. LOAD S1 RECORDS
# ============================================================

print("\nLoading S1 records...")

s1_records = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    chunk = chunk.fillna("")

    found = chunk[
        chunk["entity_id"].isin(target_ids)
    ]

    for _, row in found.iterrows():

        s1_records[
            row["entity_id"]
        ] = row

    if len(s1_records) >= len(target_ids):
        break


print(
    "S1 records loaded:",
    len(s1_records)
)


# ============================================================
# 4. OPEN SQLITE BLOCKER
# ============================================================

print("\nOpening blocker database...")

conn = sqlite3.connect(DB_PATH)

# IMPORTANT:
# Create the blocker ONCE.
blocker = ScalableBlocker(DB_PATH)


# ============================================================
# 5. GENERATE CANDIDATE PAIRS
# ============================================================

print("\nGenerating candidate pairs...")

pair_rows = []

total_entities = len(selected_gt)

for idx, (s1_id, gt_matches) in enumerate(
    selected_gt.items(),
    start=1
):

    row = s1_records.get(s1_id)

    if row is None:
        continue


    # --------------------------------------------------------
    # Normalize S1
    # --------------------------------------------------------

    name = normalize_text(
        row["business_name"]
    )

    address = normalize_text(
        row["business_address"]
    )

    country = row["country"]


    # --------------------------------------------------------
    # Get candidates from scalable blocker
    # --------------------------------------------------------

    candidates = blocker.get_candidates(
        business_name=name,
        business_address=address,
        country=country,
        name_limit=NAME_LIMIT,
        address_limit=ADDRESS_LIMIT,
        conn=conn
    )


    if not candidates:
        continue


    # --------------------------------------------------------
    # candidates contains:
    #
    # (entity_id, source)
    #
    # Keep BOTH because S2 and S3 are separate namespaces.
    # --------------------------------------------------------

    candidate_pairs = list(
        set(candidates)
    )


    if not candidate_pairs:
        continue


    # --------------------------------------------------------
    # Build SQL query using entity_id + source
    # --------------------------------------------------------

    conditions = []
    params = []

    for entity_id, source in candidate_pairs:

        conditions.append(
            "(entity_id = ? AND source = ?)"
        )

        params.extend([
            entity_id,
            source
        ])


    sql = f"""
        SELECT
            entity_id,
            source,
            country,
            business_name,
            business_address
        FROM candidates
        WHERE {" OR ".join(conditions)}
    """


    # --------------------------------------------------------
    # Fetch directly from SQLite.
    #
    # This avoids pd.read_sql_query() creating a potentially
    # huge intermediate DataFrame.
    # --------------------------------------------------------

    rows = conn.execute(
        sql,
        params
    ).fetchall()


    # --------------------------------------------------------
    # Create pair records
    # --------------------------------------------------------

    for cand in rows:

        cand_id = cand[0]
        cand_source = cand[1]
        cand_country = cand[2]
        cand_name = cand[3]
        cand_address = cand[4]


        # Candidate identity must include source.
        #
        # Example:
        # S2-123
        # S3-123
        #
        # They must remain different entities.
        candidate_key = cand_id


        # Ground truth IDs include S2-/S3-
        # so compare against the full ID.
        full_candidate_id = cand_id


        label = int(
            full_candidate_id in gt_matches
        )


        pair_rows.append({

            "source1_entity_id":
                s1_id,

            "candidate_entity_id":
                cand_id,

            "candidate_source":
                cand_source,

            "candidate_key":
                candidate_key,


            # S1 original fields

            "s1_business_name":
                row["business_name"],

            "s1_business_address":
                row["business_address"],

            "s1_country":
                row["country"],


            # S1 normalized fields

            "s1_norm_name":
                name,

            "s1_norm_address":
                address,


            # Candidate original fields

            "cand_business_name":
                cand_name,

            "cand_business_address":
                cand_address,

            "cand_country":
                cand_country,


            # Candidate normalized fields

            "cand_norm_name":
                normalize_text(
                    cand_name
                ),

            "cand_norm_address":
                normalize_text(
                    cand_address
                ),


            # Label

            "label":
                label
        })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if idx % 10 == 0:

        print(
            f"Processed {idx}/{total_entities} "
            f"S1 entities | "
            f"pairs so far: {len(pair_rows)}"
        )


conn.close()


# ============================================================
# 6. BUILD DATAFRAME
# ============================================================

df_pairs = pd.DataFrame(
    pair_rows
)


if df_pairs.empty:

    raise RuntimeError(
        "No candidate pairs were generated. "
        "Check blocker database and paths."
    )


print("\n==============================================")
print("CANDIDATE DATA")
print("==============================================")

print(
    "Total candidate pairs:",
    len(df_pairs)
)

print(
    "Positive pairs:",
    int(df_pairs["label"].sum())
)

print(
    "Negative pairs:",
    int(
        (df_pairs["label"] == 0).sum()
    )
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\nExtracting features...")

extractor = PairFeatureExtractor()

X = extractor.extract_pair_features(
    df_pairs
)

y = df_pairs[
    "label"
].values


print(
    "Feature matrix:",
    X.shape
)


# ============================================================
# 8. GROUP SPLIT BY S1
# ============================================================

unique_s1 = df_pairs[
    "source1_entity_id"
].unique()


if len(unique_s1) < 2:

    raise RuntimeError(
        "Not enough S1 entities for train/validation split."
    )


train_s1, val_s1 = train_test_split(
    unique_s1,
    test_size=0.25,
    random_state=42
)


train_mask = df_pairs[
    "source1_entity_id"
].isin(train_s1)


val_mask = df_pairs[
    "source1_entity_id"
].isin(val_s1)


X_train = X[
    train_mask
]

y_train = y[
    train_mask
]


X_val = X[
    val_mask
]

y_val = y[
    val_mask
]


val_pairs = df_pairs[
    val_mask
].copy()


val_gt = {

    s1_id: selected_gt[s1_id]

    for s1_id in val_s1

    if s1_id in selected_gt

}


print("\nTraining pairs:", len(X_train))
print("Validation pairs:", len(X_val))
print("Training S1:", len(train_s1))
print("Validation S1:", len(val_s1))


# ============================================================
# 9. TRAIN LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    num_leaves=31,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    verbose=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# 10. THRESHOLD OPTIMIZATION
# ============================================================

print("\nOptimizing F0.5 threshold...")

val_probs = model.predict_proba(
    X_val
)[:, 1]


best_threshold = 0.5
best_f05 = -1


for threshold in np.arange(
    0.10,
    0.96,
    0.01
):

    score = evaluate_f05(
        val_pairs,
        val_probs,
        threshold,
        val_gt
    )


    if score > best_f05:

        best_f05 = score
        best_threshold = threshold


# ============================================================
# 11. RESULTS
# ============================================================

print("\n==============================================")
print("RESULT")
print("==============================================")


print(
    f"Validation Macro F0.5: "
    f"{best_f05:.4f}"
)


print(
    f"Optimal threshold: "
    f"{best_threshold:.2f}"
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


artifacts = {

    "model":
        model,

    "feature_cols":
        list(X.columns),

    "optimal_threshold":
        best_threshold,

    "validation_f05":
        best_f05
}


with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        artifacts,
        f
    )


print("\nModel saved to:")
print(MODEL_PATH)

print("\nTraining complete.")