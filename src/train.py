import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.preprocess import load_source_file, preprocess_dataframe
from src.blocking import CandidateBlocker
from src.feature_engineering import PairFeatureExtractor

def compute_entity_f05(gt_set: set, pred_set: set) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity based on ground truth set and predicted set.
    Supports singletons cleanly:
    - GT empty, Pred empty -> F0.5 = 1.0
    - GT empty, Pred non-empty -> F0.5 = 0.0
    - GT non-empty, Pred empty -> F0.5 = 0.0
    """
    if len(gt_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(gt_set) == 0 and len(pred_set) > 0:
        return 0.0
    if len(gt_set) > 0 and len(pred_set) == 0:
        return 0.0

    tp = len(gt_set.intersection(pred_set))
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def evaluate_macro_f05(df_val_pairs: pd.DataFrame, y_pred_probs: np.ndarray, threshold: float, gt_dict: dict) -> float:
    """
    Computes Macro F_0.5 across all validation Source 1 entities at a given prediction probability threshold.
    """
    df_eval = df_val_pairs.copy()
    df_eval["prob"] = y_pred_probs
    
    # Filter candidates meeting threshold
    df_matched = df_eval[df_eval["prob"] >= threshold]
    
    pred_dict = df_matched.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    f05_scores = []
    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        f05_scores.append(compute_entity_f05(gt_set, pred_set))

    return float(np.mean(f05_scores)) if f05_scores else 0.0

def train_entity_resolution_model(
    train_dir="dataset/train",
    model_output_path="models/er_model.pkl",
    model_type="lightgbm"
):
    print("=== Starting Model Training & Optimization Pipeline ===")
    
    # 1. Load data
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    df_s1 = load_source_file(s1_path)
    df_s2 = load_source_file(s2_path)
    df_s3 = load_source_file(s3_path)
    
    df_s1 = preprocess_dataframe(df_s1)
    df_s2 = preprocess_dataframe(df_s2)
    df_s3 = preprocess_dataframe(df_s3)

    # 2. Parse Ground Truth into dictionary s1_id -> set(matched_ids)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt_dict = {}
    gt_pairs_set = set()

    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        matched_list = [m.strip() for m in matched_str.split(",") if m.strip()]
        gt_dict[s1_id] = set(matched_list)
        for m_id in matched_list:
            gt_pairs_set.add((s1_id, m_id))

    # 3. Stage 1: Candidate Generation (Blocking)
    print("Running Candidate Generation (Blocking)...")
    blocker = CandidateBlocker(top_k=50)
    candidate_pairs_map = blocker.fit_transform_candidates(df_s1, df_s2, df_s3)

    # 4. Construct Pairwise Dataset (S1, Candidate)
    print("Constructing Candidate Pairwise Dataset...")
    df_s1_indexed = df_s1.set_index("entity_id")
    df_cand_indexed = pd.concat([df_s2, df_s3], ignore_index=True).set_index("entity_id")

    pair_rows = []
    for s1_id, cand_list in candidate_pairs_map.items():
        if s1_id not in df_s1_indexed.index:
            continue
        s1_rec = df_s1_indexed.loc[s1_id]
        
        for cand_id in cand_list:
            if cand_id not in df_cand_indexed.index:
                continue
            cand_rec = df_cand_indexed.loc[cand_id]
            is_match = 1 if (s1_id, cand_id) in gt_pairs_set else 0

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
                "cand_norm_address": cand_rec["norm_address"],
                "label": is_match
            })

    df_pairs = pd.DataFrame(pair_rows)
    print(f"Total candidate pairs generated: {len(df_pairs)} (Positive matches: {df_pairs['label'].sum()})")

    # 5. Extract Similarity Features
    print("Extracting Similarity Features...")
    extractor = PairFeatureExtractor()
    df_features = extractor.extract_pair_features(df_pairs)
    feature_cols = list(df_features.columns)

    X = df_features
    y = df_pairs["label"].values

    # Group-based Train / Validation split by Source 1 Entity ID
    unique_s1_ids = df_s1["entity_id"].unique()
    train_s1_ids, val_s1_ids = train_test_split(unique_s1_ids, test_size=0.25, random_state=42)

    train_mask = df_pairs["source1_entity_id"].isin(train_s1_ids)
    val_mask = df_pairs["source1_entity_id"].isin(val_s1_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    val_pairs = df_pairs[val_mask].copy()
    val_gt_dict = {s1_id: gt_dict[s1_id] for s1_id in val_s1_ids if s1_id in gt_dict}

    # 6. Model Training
    print(f"Training {model_type.upper()} Classifier...")
    if model_type.lower() == "catboost":
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42
        )
        model.fit(X_train, y_train)
    else:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

    # 7. Threshold Tuning for Macro F0.5
    print("Optimizing Classification Probability Threshold for Macro F_0.5...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f05 = -1.0

    thresholds = np.arange(0.10, 0.95, 0.01)
    for thresh in thresholds:
        score = evaluate_macro_f05(val_pairs, val_probs, thresh, val_gt_dict)
        if score > best_f05:
            best_f05 = score
            best_threshold = thresh

    print(f"\n[RESULTS] Validation Macro F_0.5 Score: {best_f05:.4f} at Optimal Threshold: {best_threshold:.2f}")

    # Save model, blocker vectorizer, extractor, optimal threshold, and feature names
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    artifacts = {
        "model": model,
        "blocker": blocker,
        "optimal_threshold": best_threshold,
        "feature_cols": feature_cols,
        "best_val_f05": best_f05
    }
    
    with open(model_output_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model and artifacts successfully saved to {model_output_path}")
    return artifacts

if __name__ == "__main__":
    train_entity_resolution_model()
