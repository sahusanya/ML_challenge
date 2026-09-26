import re
import pandas as pd

# Abbreviation dictionary for entity resolution normalization
ABBREVIATION_MAP = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bco\b": "company",
    r"\bllc\b": "llc",
    r"\bdept\b": "department",
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bsq\b": "square",
    r"&": " and "
}

def normalize_text(text: str) -> str:
    """
    Normalizes input text by lowercasing, stripping whitespace,
    standardizing common abbreviations, and removing special characters.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    text = text.lower().strip()
    
    # Expand/standardize common abbreviations
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_source_file(file_path: str) -> pd.DataFrame:
    """
    Reads TSV source files with explicit tab separator and string dtype.
    """
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    df.fillna("", inplace=True)
    required_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in file {file_path}")
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized business_name, business_address, and combined_text columns to dataframe.
    """
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_text)
    df["norm_address"] = df["business_address"].apply(normalize_text)
    df["combined_text"] = (df["norm_name"] + " " + df["norm_address"]).str.strip()
    return df
