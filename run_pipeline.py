import os
import sys

from utils.generate_sample_data import create_dataset
from src.train import train_entity_resolution_model
from src.inference import run_test_inference
from utils.validate_submission import validate_submission

def main():
    print("==========================================================")
    print("      END-TO-END BUSINESS ENTITY RESOLUTION PIPELINE      ")
    print("==========================================================")

    # 1. Ensure dataset directories exist
    if not (os.path.exists("dataset/train/train_source1.tsv") and os.path.exists("dataset/test/test_source1.tsv")):
        print("\n[INFO] Dataset files not found in dataset/. Generating synthetic sample data...")
        create_dataset("train", num_entities=150)
        create_dataset("test", num_entities=100)
    else:
        print("\n[INFO] Existing dataset found in dataset/train and dataset/test.")

    # 2. Train Model & Optimize Macro F0.5 Threshold
    print("\n--- STAGE 1, 2 & 3: TRAINING & THRESHOLD OPTIMIZATION ---")
    train_artifacts = train_entity_resolution_model(
        train_dir="dataset/train",
        model_output_path="models/er_model.pkl",
        model_type="lightgbm"
    )

    # 3. Run Inference on Test Set
    print("\n--- INFERENCE & OUTPUT GENERATION ---")
    run_test_inference(
        test_dir="dataset/test",
        model_path="models/er_model.pkl",
        candidate_output_path="output/candidate_pairs.tsv",
        matching_output_path="output/matching_results.tsv"
    )

    # 4. Final Submission Validation
    print("\n--- FINAL SUBMISSION VALIDATION ---")
    valid = validate_submission(
        matching_path="output/matching_results.tsv",
        candidate_path="output/candidate_pairs.tsv",
        test_dir="dataset/test"
    )

    if valid:
        print("\n[SUCCESS] Pipeline execution complete! Both output/candidate_pairs.tsv and output/matching_results.tsv are valid and ready.")
    else:
        print("\n[ERROR] Pipeline completed but submission validation reported errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
