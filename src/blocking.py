import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess_dataframe

class CandidateBlocker:
    """
    Stage 1 Blocking: Candidate Generation using TF-IDF vectorization
    and efficient sparse matrix multiplication for cosine similarity.
    """
    def __init__(self, top_k=50, ngram_range=(3, 4)):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True
        )

    def fit_transform_candidates(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        """
        Fits TF-IDF vectorizer across all sources and computes top-K candidate pairs
        for each Source 1 record against combined (Source 2 + Source 3) records.
        """
        # Preprocess dataframes if normalized columns are not present
        if "combined_text" not in df_s1.columns:
            df_s1 = preprocess_dataframe(df_s1)
        if "combined_text" not in df_s2.columns:
            df_s2 = preprocess_dataframe(df_s2)
        if "combined_text" not in df_s3.columns:
            df_s3 = preprocess_dataframe(df_s3)

        # Concatenate S2 and S3 candidates
        df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        
        # Combine all texts to build TF-IDF vocabulary
        all_texts = pd.concat([df_s1["combined_text"], df_candidates["combined_text"]], ignore_index=True)
        self.vectorizer.fit(all_texts)

        # Transform S1 and S2+S3 texts to sparse matrices
        X_s1 = self.vectorizer.transform(df_s1["combined_text"])
        X_cand = self.vectorizer.transform(df_candidates["combined_text"])

        # Compute sparse cosine similarity via matrix multiplication
        # X_s1 is (N_s1, V), X_cand.T is (V, N_cand) -> sim_matrix is (N_s1, N_cand)
        sim_matrix = X_s1.dot(X_cand.T)

        candidate_ids_cand = df_candidates["entity_id"].values
        s1_ids = df_s1["entity_id"].values

        candidate_pairs_map = {}
        
        # Extract top-K candidates per S1 entity
        for i in range(sim_matrix.shape[0]):
            s1_id = s1_ids[i]
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                candidate_pairs_map[s1_id] = []
                continue
            
            # Find indices of top-K similarity scores
            indices = row.indices
            data = row.data
            
            if len(data) > self.top_k:
                top_indices_local = np.argpartition(data, -self.top_k)[-self.top_k:]
                sorted_local = top_indices_local[np.argsort(-data[top_indices_local])]
                top_indices = indices[sorted_local]
            else:
                top_indices = indices[np.argsort(-data)]

            # Get entity IDs and preserve uniqueness while maintaining order
            top_cand_ids = []
            seen = set()
            for idx in top_indices:
                cand_id = candidate_ids_cand[idx]
                if cand_id not in seen:
                    seen.add(cand_id)
                    top_cand_ids.append(cand_id)

            candidate_pairs_map[s1_id] = top_cand_ids

        return candidate_pairs_map

    def save_candidate_pairs(self, candidate_pairs_map: dict, output_path: str = "output/candidate_pairs.tsv"):
        """
        Saves candidate pairs dictionary to output/candidate_pairs.tsv in format:
        source1_entity_id \t candidate_entity_ids (comma-separated)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rows = []
        for s1_id, cands in candidate_pairs_map.items():
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(cands)
            })
        
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, sep="\t", index=False)
        print(f"Saved {len(df_out)} blocking candidate rows to {output_path}")
