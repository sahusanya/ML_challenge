import os
import pickle
import sqlite3

import pandas as pd

from src.preprocess import load_source_file, preprocess_dataframe
from src.scalable_blocking import ScalableBlocker
from src.feature_engineering import PairFeatureExtractor
from utils.validate_submission import validate_submission


TEST_DIR = "dataset/test"
MODEL_PATH = "models/real_er_model.pkl"
DB_PATH = "models/test_blocker.db"

CANDIDATE_OUTPUT = "output/candidate_pairs.tsv"
MATCHING_OUTPUT = "output/matching_results.tsv"


def run_test_inference():

    print("=" * 60)
    print("REAL TEST SET INFERENCE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load trained model
    # ---------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    threshold = artifacts["optimal_threshold"]

    name_limit = artifacts.get("name_limit", 50)
    address_limit = artifacts.get("address_limit", 50)

    print(f"Model: {MODEL_PATH}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Name candidate limit: {name_limit}")
    print(f"Address candidate limit: {address_limit}")

    # ---------------------------------------------------------
    # 2. Load test data
    # ---------------------------------------------------------

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")

    print("\nLoading test files...")

    df_s1 = preprocess_dataframe(load_source_file(s1_path))
    df_s2 = preprocess_dataframe(load_source_file(s2_path))
    df_s3 = preprocess_dataframe(load_source_file(s3_path))

    print(f"S1 records: {len(df_s1)}")
    print(f"S2 records: {len(df_s2)}")
    print(f"S3 records: {len(df_s3)}")

    # ---------------------------------------------------------
    # 3. Open scalable blocker
    # ---------------------------------------------------------

    print("\nOpening test blocker database...")

    blocker = ScalableBlocker(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    # ---------------------------------------------------------
    # 4. Generate candidate pairs
    # ---------------------------------------------------------

    print("\nGenerating candidate pairs...")

    candidate_rows = []
    candidate_pairs_for_output = []

    s1_count = len(df_s1)

    for idx, s1 in df_s1.iterrows():

        if (idx + 1) % 50 == 0 or idx + 1 == s1_count:
            print(
                f"Processed {idx + 1}/{s1_count} S1 entities | "
                f"candidate pairs: {len(candidate_rows)}"
            )

        s1_id = s1["entity_id"]

        records = blocker.get_candidate_records(
            business_name=s1["business_name"],
            business_address=s1["business_address"],
            country=s1["country"],
            name_limit=name_limit,
            address_limit=address_limit,
            conn=conn
        )

        seen = set()

        for (
            cand_id,
            cand_source,
            cand_country,
            cand_name,
            cand_address
        ) in records:

            key = (cand_id, cand_source)

            if key in seen:
                continue

            seen.add(key)

            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": cand_source,

                "s1_business_name": s1["business_name"],
                "s1_business_address": s1["business_address"],
                "s1_country": s1["country"],

                "s1_norm_name": s1["norm_name"],
                "s1_norm_address": s1["norm_address"],

                "cand_business_name": cand_name,
                "cand_business_address": cand_address,
                "cand_country": cand_country,

                "cand_norm_name": "",
                "cand_norm_address": ""
            })

            candidate_pairs_for_output.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": cand_source
            })

    conn.close()

    print("\nCandidate generation complete.")
    print(f"Total candidate pairs: {len(candidate_rows)}")

    # ---------------------------------------------------------
    # 5. Build candidate dataframe
    # ---------------------------------------------------------

    if not candidate_rows:

        print("No candidates generated.")

        matching_rows = [
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ""
            }
            for s1_id in df_s1["entity_id"]
        ]

        os.makedirs("output", exist_ok=True)

        pd.DataFrame(matching_rows).to_csv(
            MATCHING_OUTPUT,
            sep="\t",
            index=False
        )

        pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source"
            ]
        ).to_csv(
            CANDIDATE_OUTPUT,
            sep="\t",
            index=False
        )

        print("Empty submission generated.")
        return

    df_pairs = pd.DataFrame(candidate_rows)

    # ---------------------------------------------------------
    # 6. Normalize candidate records
    # ---------------------------------------------------------

    print("\nPreparing candidate records...")

    # We have the complete test S2/S3 data locally.
    # Use entity_id + source to retrieve normalized values.

    df_s2_lookup = df_s2[
        [
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "norm_name",
            "norm_address"
        ]
    ].copy()

    df_s2_lookup["candidate_source"] = "S2"

    df_s3_lookup = df_s3[
        [
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "norm_name",
            "norm_address"
        ]
    ].copy()

    df_s3_lookup["candidate_source"] = "S3"

    df_candidates = pd.concat(
        [df_s2_lookup, df_s3_lookup],
        ignore_index=True
    )

    df_candidates = df_candidates.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "country": "cand_country",
            "business_name": "cand_business_name",
            "business_address": "cand_business_address",
            "norm_name": "cand_norm_name",
            "norm_address": "cand_norm_address"
        }
    )

    df_pairs = df_pairs.drop(
        columns=[
            "cand_business_name",
            "cand_business_address",
            "cand_country",
            "cand_norm_name",
            "cand_norm_address"
        ]
    )

    df_pairs = df_pairs.merge(
        df_candidates,
        on=["candidate_entity_id", "candidate_source"],
        how="left"
    )

    print(f"Pair dataframe shape: {df_pairs.shape}")

    # ---------------------------------------------------------
    # 7. Save candidate_pairs.tsv
    # ---------------------------------------------------------

    os.makedirs("output", exist_ok=True)

    candidate_df = pd.DataFrame(candidate_pairs_for_output)

    candidate_grouped = (
        candidate_df
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .reset_index()
    )

    candidate_grouped.columns = [
        "source1_entity_id",
        "candidate_entity_ids"
    ]

    candidate_grouped.to_csv(
        CANDIDATE_OUTPUT,
        sep="\t",
        index=False
    )

    print(
        f"Candidate file written: {CANDIDATE_OUTPUT}"
    )

    # ---------------------------------------------------------
    # 8. Extract features
    # ---------------------------------------------------------

    print("\nExtracting pair features...")

    extractor = PairFeatureExtractor()

    X_test = extractor.extract_pair_features(df_pairs)

    print(f"Feature matrix: {X_test.shape}")

    # ---------------------------------------------------------
    # 9. Predict
    # ---------------------------------------------------------

    print("\nPredicting match probabilities...")

    probabilities = model.predict_proba(X_test)[:, 1]

    df_pairs["match_prob"] = probabilities

    # ---------------------------------------------------------
    # 10. Apply threshold
    # ---------------------------------------------------------

    df_surviving = df_pairs[
        df_pairs["match_prob"] >= threshold
    ].copy()

    print(
        f"Pairs above threshold: {len(df_surviving)}"
    )

    # ---------------------------------------------------------
    # 11. Build matching_results.tsv
    # ---------------------------------------------------------

    matched_dict = (
        df_surviving
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(list)
        .to_dict()
    )

    matching_rows = []

    for s1_id in df_s1["entity_id"]:

        matched_ids = matched_dict.get(s1_id, [])

        # Remove duplicates while preserving order.
        unique_ids = []
        seen = set()

        for entity_id in matched_ids:

            if entity_id not in seen:
                seen.add(entity_id)
                unique_ids.append(entity_id)

        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(unique_ids)
        })

    df_matching = pd.DataFrame(matching_rows)

    df_matching.to_csv(
        MATCHING_OUTPUT,
        sep="\t",
        index=False
    )

    print(
        f"Matching file written: {MATCHING_OUTPUT}"
    )

    print(
        f"Rows in matching_results.tsv: {len(df_matching)}"
    )

    # ---------------------------------------------------------
    # 12. Validate submission
    # ---------------------------------------------------------

    print("\nRunning official validator...")

    validate_submission(
        matching_path=MATCHING_OUTPUT,
        candidate_path=CANDIDATE_OUTPUT,
        test_dir=TEST_DIR
    )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_test_inference()
