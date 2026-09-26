import os
import sys
import pandas as pd

def validate_submission(
    matching_path="output/matching_results.tsv",
    candidate_path="output/candidate_pairs.tsv",
    test_dir="dataset/test"
):
    print("=== Submitting Submission Format & Rule Validator ===")
    errors = []
    warnings = []

    # 1. Check file existence
    if not os.path.exists(matching_path):
        errors.append(f"Matching results file not found at: {matching_path}")
    if not os.path.exists(candidate_path):
        errors.append(f"Candidate pairs file not found at: {candidate_path}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return False

    # Load test source files to get ground truth test IDs
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not (os.path.exists(test_s1_path) and os.path.exists(test_s2_path) and os.path.exists(test_s3_path)):
        warnings.append(f"Test files not found in {test_dir}. Skipping ID existence verification against test files.")
        valid_test_s1_ids = None
        valid_test_cand_ids = None
    else:
        df_s1 = pd.read_csv(test_s1_path, sep="\t")
        df_s2 = pd.read_csv(test_s2_path, sep="\t")
        df_s3 = pd.read_csv(test_s3_path, sep="\t")
        valid_test_s1_ids = set(df_s1["entity_id"].astype(str))
        valid_test_cand_ids = set(df_s2["entity_id"].astype(str)).union(set(df_s3["entity_id"].astype(str)))

    # Validate matching_results.tsv
    with open(matching_path, "r", encoding="utf-8") as f:
        matching_lines = f.read().splitlines()

    if not matching_lines:
        errors.append("matching_results.tsv is empty.")
    else:
        header = matching_lines[0].split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"Invalid header in matching_results.tsv: {header}. Expected ['source1_entity_id', 'matched_entity_ids']")

    matching_dict = {}
    for i, line in enumerate(matching_lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(f"Line {i} in matching_results.tsv does not have exactly two tab-separated columns: '{line}'")
            continue
        s1_id, match_str = parts[0].strip(), parts[1].strip()
        if s1_id in matching_dict:
            errors.append(f"Duplicate source1_entity_id '{s1_id}' in matching_results.tsv at line {i}")
        
        matches = [m.strip() for m in match_str.split(",") if m.strip()]
        if len(matches) != len(set(matches)):
            errors.append(f"Duplicate entity IDs in matching list for '{s1_id}': {match_str}")
        
        matching_dict[s1_id] = matches

    # Validate candidate_pairs.tsv
    with open(candidate_path, "r", encoding="utf-8") as f:
        candidate_lines = f.read().splitlines()

    if not candidate_lines:
        errors.append("candidate_pairs.tsv is empty.")
    else:
        header = candidate_lines[0].split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"Invalid header in candidate_pairs.tsv: {header}. Expected ['source1_entity_id', 'candidate_entity_ids']")

    candidate_dict = {}
    for i, line in enumerate(candidate_lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(f"Line {i} in candidate_pairs.tsv does not have exactly two tab-separated columns: '{line}'")
            continue
        s1_id, cand_str = parts[0].strip(), parts[1].strip()
        if s1_id in candidate_dict:
            errors.append(f"Duplicate source1_entity_id '{s1_id}' in candidate_pairs.tsv at line {i}")
        
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        if len(cands) != len(set(cands)):
            errors.append(f"Duplicate candidate IDs for '{s1_id}': {cand_str}")
            
        candidate_dict[s1_id] = cands

    # Check alignment and constraints
    if valid_test_s1_ids is not None:
        missing_in_matching = valid_test_s1_ids - set(matching_dict.keys())
        if missing_in_matching:
            errors.append(f"{len(missing_in_matching)} Source 1 test entities are missing from matching_results.tsv")
            
        missing_in_candidates = valid_test_s1_ids - set(candidate_dict.keys())
        if missing_in_candidates:
            errors.append(f"{len(missing_in_candidates)} Source 1 test entities are missing from candidate_pairs.tsv")

    # Check subset rule: matches must be a subset of candidates
    subset_violations = 0
    invalid_cand_id_count = 0
    self_match_count = 0

    for s1_id, matches in matching_dict.items():
        cands = set(candidate_dict.get(s1_id, []))
        for m in matches:
            if m.startswith("S1-"):
                self_match_count += 1
            if m not in cands:
                subset_violations += 1
            if valid_test_cand_ids is not None and m not in valid_test_cand_ids:
                invalid_cand_id_count += 1

    if subset_violations > 0:
        errors.append(f"Found {subset_violations} matched IDs that were not present in candidate_pairs.tsv")
    if self_match_count > 0:
        errors.append(f"Found {self_match_count} self-matches (S1 IDs) in matching_results.tsv")
    if invalid_cand_id_count > 0:
        errors.append(f"Found {invalid_cand_id_count} matched IDs that do not exist in test set Source 2 or Source 3")

    if warnings:
        for w in warnings:
            print(f"[WARNING] {w}")

    if errors:
        print(f"\n[FAIL] Validation failed with {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
        return False
    else:
        print("\n[PASS] Validation successful! Submission files are 100% compliant with all rules.")
        return True

if __name__ == "__main__":
    test_d = sys.argv[1] if len(sys.argv) > 1 else "dataset/test"
    matching_p = sys.argv[2] if len(sys.argv) > 2 else "output/matching_results.tsv"
    candidate_p = sys.argv[3] if len(sys.argv) > 3 else "output/candidate_pairs.tsv"
    
    success = validate_submission(matching_p, candidate_p, test_d)
    sys.exit(0 if success else 1)
