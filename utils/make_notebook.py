import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Business Entity Resolution Challenge - Interactive ML Pipeline\n",
                "\n",
                "This notebook implements an end-to-end Machine Learning pipeline for Business Entity Resolution using:\n",
                "- **pandas** & **scikit-learn**\n",
                "- **TF-IDF Sparse Candidate Blocking** (Stage 1)\n",
                "- **Pairwise Similarity Feature Engineering** (Levenshtein, Jaro-Winkler, Jaccard, Token Overlap, Open-set Country match) (Stage 2)\n",
                "- **LightGBM Binary Classifier** & **Macro $F_{0.5}$ Threshold Tuning** (Stage 3)\n",
                "- **Submission Output Generation & Format Validation** (Stage 4)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Imports and Setup"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "\n",
                "# Add project root directory to path\n",
                "sys.path.append(os.getcwd())\n",
                "\n",
                "from src.preprocess import load_source_file, preprocess_dataframe, normalize_text\n",
                "from src.blocking import CandidateBlocker\n",
                "from src.feature_engineering import PairFeatureExtractor\n",
                "from src.train import train_entity_resolution_model\n",
                "from src.inference import run_test_inference\n",
                "from utils.validate_submission import validate_submission\n",
                "from utils.generate_sample_data import create_dataset\n",
                "\n",
                "print(\"Project modules imported successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Ingest Data & Inspect Dataset"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate dataset if missing\n",
                "if not (os.path.exists(\"dataset/train/train_source1.tsv\") and os.path.exists(\"dataset/test/test_source1.tsv\")):\n",
                "    create_dataset(\"train\", 150)\n",
                "    create_dataset(\"test\", 100)\n",
                "\n",
                "df_s1 = load_source_file(\"dataset/train/train_source1.tsv\")\n",
                "df_s2 = load_source_file(\"dataset/train/train_source2.tsv\")\n",
                "df_s3 = load_source_file(\"dataset/train/train_source3.tsv\")\n",
                "\n",
                "print(f\"Source 1 Records: {len(df_s1)}\")\n",
                "print(f\"Source 2 Records: {len(df_s2)}\")\n",
                "print(f\"Source 3 Records: {len(df_s3)}\")\n",
                "df_s1.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Stage 1: Candidate Generation (Blocking)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_s1_prep = preprocess_dataframe(df_s1)\n",
                "df_s2_prep = preprocess_dataframe(df_s2)\n",
                "df_s3_prep = preprocess_dataframe(df_s3)\n",
                "\n",
                "blocker = CandidateBlocker(top_k=50)\n",
                "candidate_map = blocker.fit_transform_candidates(df_s1_prep, df_s2_prep, df_s3_prep)\n",
                "print(\"Total S1 entities blocked:\", len(candidate_map))\n",
                "print(\"Sample candidate list for S1-00001:\", candidate_map.get(\"S1-00001\", [])[:5])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Stage 2 & 3: Model Training & Macro $F_{0.5}$ Threshold Optimization"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "artifacts = train_entity_resolution_model(\n",
                "    train_dir=\"dataset/train\",\n",
                "    model_output_path=\"models/er_model.pkl\",\n",
                "    model_type=\"lightgbm\"\n",
                ")\n",
                "\n",
                "print(f\"Optimal Threshold: {artifacts['optimal_threshold']:.2f}\")\n",
                "print(f\"Validation Macro F0.5 Score: {artifacts['best_val_f05']:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Stage 4: Test Set Inference & Output Generation"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "run_test_inference(\n",
                "    test_dir=\"dataset/test\",\n",
                "    model_path=\"models/er_model.pkl\",\n",
                "    candidate_output_path=\"output/candidate_pairs.tsv\",\n",
                "    matching_output_path=\"output/matching_results.tsv\"\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Submission Format & Constraint Validation"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "is_valid = validate_submission(\n",
                "    matching_path=\"output/matching_results.tsv\",\n",
                "    candidate_path=\"output/candidate_pairs.tsv\",\n",
                "    test_dir=\"dataset/test\"\n",
                ")\n",
                "print(\"Submission format fully valid?\", is_valid)"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("entity_resolution_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created entity_resolution_pipeline.ipynb successfully!")
