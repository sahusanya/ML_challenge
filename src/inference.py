import os
import pickle
import numpy as np
import pandas as pd

from src.preprocess import load_source_file, preprocess_dataframe
from src.blocking import CandidateBlocker
from src.feature_engineering import PairFeatureExtractor
from utils.validate_submission import validate_submission

def run_test_inference(
    test_dir="dataset/test",
    model_path="models/er_model.pkl",
    candidate_output_path="output/candidate_pairs.tsv",
    matching_output_path="output/matching_results.tsv"
):
    print("=== Starting Test Set Inference & Submission Generation ===")
    
    # 1. Load saved model artifacts
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    blocker = artifacts["blocker"]
    optimal_threshold = artifacts["optimal_threshold"]
    
    print(f"Loaded model from {model_path}. Using optimal classification threshold: {optimal_threshold:.2f}")

    # 2. Load test source files
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    df_s1 = load_source_file(s1_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)

    df_s1 = preprocess_dataframe(df_s1)
    df_s2 = preprocess_dataframe(df_s2)
    df_s3 = preprocess_dataframe(df_s3)

    # 3. Stage 1: Candidate Generation (Blocking)
    print("Generating Candidate Pairs for Test Set...")
    candidate_pairs_map = blocker.fit_transform_candidates(df_s1, df_s2, df_s3)

    # Save output/candidate_pairs.tsv
    blocker.save_candidate_pairs(candidate_pairs_map, output_path=candidate_output_path)

    # 4. Construct Pairwise Test Features
    df_s1_indexed = df_s1.set_index("entity_id")
    df_cand_indexed = pd.concat([df_s2, df_s3], ignore_index=True).set_index("entity_id")

    pair_rows = []
    for s1_id in df_s1["entity_id"]:
        cand_list = candidate_pairs_map.get(s1_id, [])
        s1_rec = df_s1_indexed.loc[s1_id]

        for cand_id in cand_list:
            if cand_id not in df_cand_indexed.index:
                continue
            cand_rec = df_cand_indexed.loc[cand_id]

            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "s1_business_name": s1_rec["business_name"],
                "s1_business_address": s1_rec["business_address"],
                "s1_country": s1_rec["country"],
                "s1_norm_name": s1_rec["norm_name"],
                "s1_norm_address": s1_rec["norm_address"],
                "cand_business_name": cand_rec["business_name"],
                "cand_business_address": cand_rec["business_address"],
                "cand_country": cand_rec["country"],
                "cand_norm_name": cand_rec["norm_name"],
                "cand_norm_address": cand_rec["norm_address"]
            })

    if not pair_rows:
        # Fallback if no candidate pairs generated at all
        print("Warning: No candidate pairs were generated during test inference.")
        matching_rows = [{"source1_entity_id": s1_id, "matched_entity_ids": ""} for s1_id in df_s1["entity_id"]]
        pd.DataFrame(matching_rows).to_csv(matching_output_path, sep="\t", index=False)
        return

    df_test_pairs = pd.DataFrame(pair_rows)

    # Extract Features
    print("Extracting Test Pair Similarity Features...")
    extractor = PairFeatureExtractor()
    X_test = extractor.extract_pair_features(df_test_pairs)

    # 5. Predict Match Probabilities
    print("Predicting Match Probabilities with Model...")
    probs = model.predict_proba(X_test)[:, 1]
    df_test_pairs["match_prob"] = probs

    # Filter by Optimal Threshold
    df_surviving = df_test_pairs[df_test_pairs["match_prob"] >= optimal_threshold]

    matched_dict = df_surviving.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()

    # 6. Build Final matching_results.tsv
    matching_rows = []
    for s1_id in df_s1["entity_id"]:
        matched_ids = matched_dict.get(s1_id, [])
        # Preserve unique ordering
        unique_matched = []
        seen = set()
        for m_id in matched_ids:
            if m_id not in seen:
                seen.add(m_id)
                unique_matched.append(m_id)

        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(unique_matched)
        })

    os.makedirs(os.path.dirname(matching_output_path), exist_ok=True)
    df_out = pd.DataFrame(matching_rows)
    df_out.to_csv(matching_output_path, sep="\t", index=False)
    print(f"Successfully generated {len(df_out)} matching result rows in {matching_output_path}")

    # 7. Validate Submission Format
    validate_submission(
        matching_path=matching_output_path,
        candidate_path=candidate_output_path,
        test_dir=test_dir
    )

if __name__ == "__main__":
    run_test_inference()
