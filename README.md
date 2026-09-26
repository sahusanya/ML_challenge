# Production Business Entity Resolution ML Pipeline

An end-to-end Machine Learning pipeline for large-scale Business Entity Resolution (ER) using **pandas**, **scikit-learn**, **TF-IDF sparse candidate blocking**, **string similarity feature engineering**, and **LightGBM / CatBoost** classification with threshold optimization for **Macro $F_{0.5}$**.

---

## 📁 Repository & Pipeline Architecture

```
d:/Amazon_ML/
├── src/
│   ├── preprocess.py          # TSV loading & abbreviation text normalization
│   ├── blocking.py            # Stage 1: TF-IDF char_wb candidate generation (Top 50)
│   ├── feature_engineering.py # Stage 2: Levenshtein, Jaro-Winkler, Jaccard, country match
│   ├── train.py               # Stage 3: LightGBM model training & Macro F0.5 optimization
│   └── inference.py           # Stage 4: Test set inference & output TSV generation
├── utils/
│   ├── generate_sample_data.py # Synthetic train/test TSV dataset generator
│   └── validate_submission.py  # Standard format & constraint validator
├── dataset/
│   ├── train/                 # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                  # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── output/
│   ├── candidate_pairs.tsv    # Blocking candidates output
│   └── matching_results.tsv   # Scored entity matching predictions
├── models/
│   └── er_model.pkl           # Saved model & optimal threshold artifacts
├── run_pipeline.py            # Complete end-to-end execution script
├── requirements.txt           # Environment dependencies
└── README.md                  # Instructions & documentation
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Execute the full pipeline with a single command:
```bash
python run_pipeline.py
```

---

## ⚙️ Modular Usage

### Step 1: Preprocessing & Candidate Blocking
```python
from src.preprocess import load_source_file, preprocess_dataframe
from src.blocking import CandidateBlocker

df_s1 = preprocess_dataframe(load_source_file("dataset/test/test_source1.tsv"))
df_s2 = preprocess_dataframe(load_source_file("dataset/test/test_source2.tsv"))
df_s3 = preprocess_dataframe(load_source_file("dataset/test/test_source3.tsv"))

blocker = CandidateBlocker(top_k=50)
candidate_map = blocker.fit_transform_candidates(df_s1, df_s2, df_s3)
blocker.save_candidate_pairs(candidate_map, "output/candidate_pairs.tsv")
```

### Step 2: Model Training & Threshold Tuning
```python
from src.train import train_entity_resolution_model

artifacts = train_entity_resolution_model(
    train_dir="dataset/train",
    model_output_path="models/er_model.pkl",
    model_type="lightgbm" # or "catboost"
)
```

### Step 3: Test Set Inference & Validation
```python
from src.inference import run_test_inference

run_test_inference(
    test_dir="dataset/test",
    model_path="models/er_model.pkl",
    candidate_output_path="output/candidate_pairs.tsv",
    matching_output_path="output/matching_results.tsv"
)
```

---

## 📊 Key Highlights & Design Choices

1. **Explicit Tab-Separation (`.tsv`)**: All file IO uses `sep="\t"` to ensure business names/addresses containing commas are parsed cleanly.
2. **Standardized Text Normalization**: Handles common legal suffixes (`corp`, `inc`, `ltd`, `pvt`, `co`) and address abbreviations (`rd`, `st`, `ave`, `blvd`, `dr`, `pkwy`, `&`).
3. **High-Recall TF-IDF Blocking**: Uses character n-grams (`char_wb`, range 3-4) and sparse dot products to compute top 50 candidate pairs per Source 1 entity in sub-second time.
4. **Comprehensive Pairwise Features**: Includes Levenshtein similarity, Jaro-Winkler distance, Jaccard token similarity, exact token overlap ratios, and binary country matching.
5. **Open-Set Country Support**: `country_match` is a binary comparison feature (`s1.country == cand.country`). No hardcoded or one-hot encoded country features to handle unseen countries like France.
6. **Macro $F_{0.5}$ Optimization**: Precision-heavy evaluation metric ($F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$) tuned across probability thresholds $[0.10, 0.95]$ to minimize false merges and reward correct singletons.
