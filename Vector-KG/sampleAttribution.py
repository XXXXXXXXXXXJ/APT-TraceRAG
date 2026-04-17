# sampleAttribution.py
import os
import json
import numpy as np
import pandas as pd
import logging
import faiss
from tqdm import tqdm

# Reuse dependencies from featuresPrepare.py (assumes environment paths are set)
from featgenerator import group_features
from featgenerator.floss_general_feat import FlossFeatures
from path_var import WORKSPACE, LOG_PATH

# --------------------------- Logging configuration ---------------------------
LOG_FILENAME = 'sampleAttribution.log'
log_dir = os.path.join(WORKSPACE, LOG_PATH)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, LOG_FILENAME),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class SampleAttributor:
    def __init__(self, vector_store_dir="./vector_store"):
        self.vector_store_dir = vector_store_dir
        
        # Path definitions
        self.feature_cols_path = os.path.join(vector_store_dir, "feature_columns.json")
        self.vectors_path = os.path.join(vector_store_dir, "combined_vectors.npy")
        self.index_path = os.path.join(vector_store_dir, "combined.index")
        self.samples_meta_path = os.path.join(vector_store_dir, "samples.json")

        self._load_resources()

    def _load_resources(self):
        logging.info("Loading vector store resources...")
        
        # 1. Load feature column definitions
        if not os.path.exists(self.feature_cols_path):
            raise FileNotFoundError(f"Missing {self.feature_cols_path}")
        with open(self.feature_cols_path, 'r') as f:
            self.feature_columns = json.load(f)
            
        # 2. Load sample metadata (samples.json)
        if not os.path.exists(self.samples_meta_path):
            raise FileNotFoundError(f"Missing {self.samples_meta_path}")
        with open(self.samples_meta_path, 'r') as f:
            raw_samples = json.load(f)
            # Build a fast lookup dict: feature_id -> sample_info
            self.id_to_meta = {item['feature_id']: item for item in raw_samples}
            
        # 3. Do not load combined.index — always rebuild from combined_vectors.npy
        # ----------------------------------------------------------------------
        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError("Missing combined_vectors.npy")

        logging.info(f"Loading vectors from {self.vectors_path}...")
        vectors = np.load(self.vectors_path).astype(np.float32)

        dim = vectors.shape[1]
        logging.info(f"Building fresh FAISS IndexFlatL2 with dim={dim} ...")

        # 100% order consistency; no misalignment
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        logging.info(f"Loaded {len(self.feature_columns)} feature columns.")
        logging.info(f"Index built with {self.index.ntotal} vectors (FlatL2).")

    def extract_features(self, sample_dir):
        """
        Extract raw features, reusing logic from featuresPrepare.py.
        """
        logging.info(f"[1] Extracting raw features from {sample_dir}...")
        
        # === 1. Basic feature extraction ===
        # Assume group_features.load_and_prepare_datasets supports a target directory
        # If not, adjust how target_dir is passed per the actual code
        exif_features, malcat_features, joined_df, adversary_dataset = group_features.load_and_prepare_datasets()
        
        logging.info("[2] Processing core features...")
        final_features = group_features.process_and_merge_features(
            exif_features, malcat_features, joined_df, adversary_dataset
        )

        # === 2. String embedding extraction ===
        logging.info("[3] Generating String Embeddings...")
        floss_feat = FlossFeatures()
        string_embedding_processor = group_features.StringEmbeddingProcessor(
            joined_df=joined_df
        )
        string_embedding_df = string_embedding_processor.process()
        
        # Rename columns and fill missing values
        rename_map = {col: f"str_emb_{col}" for col in string_embedding_df.columns if col != 'hash'}
        string_embedding_df = string_embedding_df.rename(columns=rename_map)
        cols_to_fill = [c for c in string_embedding_df.columns if c != 'hash']
        string_embedding_df[cols_to_fill] = string_embedding_df[cols_to_fill].fillna(0.0)

        # === 3. Merge ===
        logging.info("[4] Merging all features...")
        combined_df = pd.merge(final_features, string_embedding_df, on="hash", how="left")
        
        # Fill missing values
        emb_cols = [c for c in combined_df.columns if str(c).startswith("str_emb_")]
        combined_df[emb_cols] = combined_df[emb_cols].fillna(0.0)
        
        logging.info(f"Extracted raw combined shape: {combined_df.shape}")
        return combined_df

    def align_features(self, raw_df):
        """
        [Key] Align new sample features to the dimensions in features_columns.json.
        """
        logging.info("[5] Aligning features with Vector Store definition...")
        
        hashes = raw_df['hash'].tolist()
        
        # Force alignment via reindex:
        # 1. Drop columns present in raw_df but missing in feature_columns
        # 2. Add columns missing in raw_df but present in feature_columns (fill_value=0)
        # 3. Ensure column order matches exactly
        aligned_df = raw_df.reindex(columns=self.feature_columns, fill_value=0.0)
        
        # Convert to float32 matrix for FAISS
        aligned_matrix = aligned_df.values.astype(np.float32)
        
        logging.info(f"Aligned matrix shape: {aligned_matrix.shape}")
        return hashes, aligned_matrix

    def find_nearest_samples(self, hashes, matrix, top_k=20):
        """
        Query the FAISS index and return Top-K similar samples.
        """
        logging.info(f"[6] Searching top {top_k} nearest neighbors for {len(hashes)} samples...")
        
        # D: distances, I: indices (feature_id in samples.json)
        D, I = self.index.search(matrix, top_k)
        
        results = []
        
        for row_idx, query_hash in enumerate(hashes):
            query_result = {
                "hash": query_hash,
                "matches": []
            }
            
            for rank in range(top_k):
                feature_id = int(I[row_idx, rank])  # Corresponding feature_id
                distance = float(D[row_idx, rank])
                
                if feature_id == -1: continue  # Invalid result
                
                # Lookup in metadata loaded from samples.json
                meta = self.id_to_meta.get(feature_id)
                
                if meta:
                    match_info = {
                        "rank": rank + 1,
                        "match_hash": meta.get("hash"),
                        "match_org": meta.get("org_label"),
                        "is_representative": meta.get("is_representative", False),
                        "distance": distance,
                        "feature_id": feature_id
                    }
                    query_result["matches"].append(match_info)
                else:
                    logging.warning(f"Feature ID {feature_id} returned by FAISS but not found in samples.json")
            
            results.append(query_result)
            
        return results

    def run(self, input_dir, output_file):
        # 1. Extract features
        raw_df = self.extract_features(input_dir)
        
        # 2. Align dimensions
        hashes, aligned_matrix = self.align_features(raw_df)
        
        # 3. Retrieve Top 20
        attribution_results = self.find_nearest_samples(hashes, aligned_matrix, top_k=20)
        
        # 4. Save results
        logging.info(f"[7] Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(attribution_results, f, indent=2)
        
        logging.info("Sample attribution search complete.")

# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # Configuration
    SAMPLE_DIR = "./APT-ClarityExec_1000"
    OUTPUT_FILE = "./sample_attribution_top20.json"
    VECTOR_STORE_DIR = "./vector_store"

    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"Error: Vector store directory '{VECTOR_STORE_DIR}' not found.")
        exit(1)

    try:
        attributor = SampleAttributor(vector_store_dir=VECTOR_STORE_DIR)
        attributor.run(input_dir=SAMPLE_DIR, output_file=OUTPUT_FILE)
    except Exception as e:
        logging.error(f"Fatal Error: {str(e)}", exc_info=True)
        print(f"Failed. Check log for details: {os.path.join(WORKSPACE, LOG_PATH, LOG_FILENAME)}")