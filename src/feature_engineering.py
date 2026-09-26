import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz import distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Fast Pure-Python fallback string functions if rapidfuzz is missing
def pure_levenshtein_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len2 + 1):
            temp = dp[j]
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = temp
    dist = dp[len2]
    return 1.0 - dist / max(len1, len2)

def pure_jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = (max(len1, len2) // 2) - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    # Prefix scale
    l = 0
    max_l = min(4, min(len1, len2))
    while l < max_l and s1[l] == s2[l]:
        l += 1

    return jaro + l * p * (1 - jaro)

def compute_levenshtein_ratio(s1: str, s2: str) -> float:
    if HAS_RAPIDFUZZ:
        return distance.Levenshtein.normalized_similarity(s1, s2)
    return pure_levenshtein_ratio(s1, s2)

def compute_jaro_winkler(s1: str, s2: str) -> float:
    if HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(s1, s2)
    return pure_jaro_winkler(s1, s2)

def compute_jaccard_similarity(tokens1: set, tokens2: set) -> float:
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def compute_token_overlap(tokens1: list, tokens2: list):
    if not tokens1 and not tokens2:
        return 0, 1.0
    if not tokens1 or not tokens2:
        return 0, 0.0
    set1, set2 = set(tokens1), set(tokens2)
    overlap_count = len(set1.intersection(set2))
    overlap_ratio = overlap_count / max(len(set1), len(set2))
    return overlap_count, overlap_ratio

class PairFeatureExtractor:
    """
    Computes pairwise similarity features for (Source 1, Candidate) entity pairs.
    """
    def __init__(self):
        pass

    def extract_pair_features(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Input df_pairs must contain:
        s1_business_name, s1_business_address, s1_country, s1_norm_name, s1_norm_address
        cand_business_name, cand_business_address, cand_country, cand_norm_name, cand_norm_address
        """
        records = []
        for idx, row in df_pairs.iterrows():
            name1 = row["s1_norm_name"]
            name2 = row["cand_norm_name"]
            addr1 = row["s1_norm_address"]
            addr2 = row["cand_norm_address"]
            
            country1 = str(row["s1_country"]).strip().lower()
            country2 = str(row["cand_country"]).strip().lower()

            tok_name1 = name1.split()
            tok_name2 = name2.split()
            tok_addr1 = addr1.split()
            tok_addr2 = addr2.split()

            # Name similarities
            name_lev = compute_levenshtein_ratio(name1, name2)
            name_jw = compute_jaro_winkler(name1, name2)
            name_jaccard = compute_jaccard_similarity(set(tok_name1), set(tok_name2))
            name_overlap_cnt, name_overlap_ratio = compute_token_overlap(tok_name1, tok_name2)

            # Address similarities
            addr_lev = compute_levenshtein_ratio(addr1, addr2)
            addr_jw = compute_jaro_winkler(addr1, addr2)
            addr_jaccard = compute_jaccard_similarity(set(tok_addr1), set(tok_addr2))
            addr_overlap_cnt, addr_overlap_ratio = compute_token_overlap(tok_addr1, tok_addr2)

            # Country binary feature (Do NOT one-hot encode country!)
            country_match = 1.0 if (country1 == country2 and country1 != "") else 0.0

            # Length differences
            len_diff_name = abs(len(name1) - len(name2))
            len_diff_addr = abs(len(addr1) - len(addr2))

            records.append({
                "name_levenshtein_ratio": name_lev,
                "name_jaro_winkler": name_jw,
                "name_jaccard": name_jaccard,
                "name_token_overlap_count": name_overlap_cnt,
                "name_token_overlap_ratio": name_overlap_ratio,
                "address_levenshtein_ratio": addr_lev,
                "address_jaro_winkler": addr_jw,
                "address_jaccard": addr_jaccard,
                "address_token_overlap_count": addr_overlap_cnt,
                "address_token_overlap_ratio": addr_overlap_ratio,
                "country_match": country_match,
                "len_diff_name": len_diff_name,
                "len_diff_addr": len_diff_addr
            })

        return pd.DataFrame(records)
