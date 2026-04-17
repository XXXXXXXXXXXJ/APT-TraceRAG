# featuresPrepare.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

import faiss

# === Used exactly as in GroupAttribution ===
from featgenerator import group_features
from featgenerator.floss_general_feat import FlossFeatures

from path_var import WORKSPACE, LOG_PATH

# nohup python featuresPrepare.py > output.log 2>&1 &
# nohup python featuresPrepare.py > output_benchmark.log 2>&1 &

# --------------------------- Logging configuration ---------------------------
LOG_FILENAME = 'featuresPrepare.log'
log_dir = os.path.join(WORKSPACE, LOG_PATH)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, LOG_FILENAME)
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# =====================================================
#  VectorStore embedded in featuresPrepare.py (no external file)
# =====================================================
class VectorStore:
    """
    Lightweight FAISS IndexFlatL2 vector store:
    - auto-manages vectors
    - saves index
    """
    def __init__(self, dim: int, metric: str = "euclidean"):
        self.dim = dim
        self.metric = metric

        if metric == "euclidean":
            self.index = faiss.IndexFlatL2(dim)
        elif metric in ("inner", "ip", "cosine"):
            # cosine = inner product + normalization
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.hash_to_id = {}
        self.vectors = []   # list[np.ndarray]

    def add_vector(self, sample_hash: str, vec: np.ndarray):
        vec = vec.astype(np.float32).reshape(1, -1)

        if vec.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vec.shape[1]}")

        self.index.add(vec)
        vid = len(self.vectors)
        self.vectors.append(vec.squeeze())
        self.hash_to_id[sample_hash] = vid
        return vid
    
    def add_vectors_batch(self, hashes: list, matrix: np.ndarray):
        """Add vectors in batch for a large speedup."""
        matrix = matrix.astype(np.float32)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"Dim mismatch: expect {self.dim}, got {matrix.shape[1]}")
        
        start_id = self.index.ntotal
        self.index.add(matrix)
        
        # Batch update mapping
        for i, h in enumerate(hashes):
            self.hash_to_id[h] = start_id + i
        
        return start_id  # Return start ID


# =====================================================
# Main feature preparation flow
# =====================================================
class FeaturesPreparer:
    def run(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        logging.info("[1] Loading datasets...")

        exif_features, malcat_features, joined_df, adversary_dataset = group_features.load_and_prepare_datasets()

        logging.info("[2] Process core features...")
        final_features = group_features.process_and_merge_features(
            exif_features, malcat_features, joined_df, adversary_dataset
        )

        logging.info("[3] String Embedding...")
        floss_feat = FlossFeatures()
        string_embedding_processor = group_features.StringEmbeddingProcessor(
            joined_df=joined_df
        )
        string_embedding_df_features = string_embedding_processor.process()
        logging.info(f"[DEBUG] String Embedding Rows Raw: {len(string_embedding_df_features)}")
        raw_strings_map = string_embedding_processor.get_raw_strings()
        rename_map = {col: f"str_emb_{col}" for col in string_embedding_df_features.columns if col != 'hash'}
        string_embedding_df_features = string_embedding_df_features.rename(columns=rename_map)
        cols_to_fill = [c for c in string_embedding_df_features.columns if c != 'hash']
        string_embedding_df_features[cols_to_fill] = string_embedding_df_features[cols_to_fill].fillna(0.0)

        logging.info("[4] Build combined features...")
        combined_df = pd.merge(final_features, string_embedding_df_features, on="hash", how="left")
        emb_cols = [c for c in combined_df.columns if str(c).startswith("str_emb_")]
        combined_df[emb_cols] = combined_df[emb_cols].fillna(0.0)

        logging.info(f"[+] Combined shape: {combined_df.shape}")
        feature_cols = [c for c in combined_df.columns if c not in ['hash', 'Normalized_Tag']]
        logging.info("[4.5] Saving feature columns for inference alignment...")
        # Save feature column list as JSON (gold standard for inference alignment)
        with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        features_only_df = combined_df[feature_cols]
        
        combined_dim = features_only_df.shape[1]
        logging.info(f"[+] Auto-detected combined_dim = {combined_dim}")
        final_hashes = combined_df['hash'].tolist()
        final_tags = combined_df['Normalized_Tag'].tolist()

        # Create vector store
        logging.info("[5] Initialize VectorStore...")
        sample_vec_store = VectorStore(dim=combined_dim, metric="euclidean")
        combined_matrix = features_only_df.values.astype(np.float32)

        logging.info("[6] Fill FAISS index + build Sample nodes...")
        if len(final_hashes) != combined_matrix.shape[0]:
            raise ValueError("Hash count does not match matrix row count")
        sample_vec_store.add_vectors_batch(final_hashes, combined_matrix)
        samples_meta_df = pd.DataFrame({
            "hash": final_hashes,
            "org_label": final_tags,
            "feature_id": list(range(len(final_hashes))),
            "is_representative": False
        })
        samples_out = samples_meta_df.to_dict(orient="records")

        final_hash_set = set(final_hashes)

        logging.info("[7] Export MalcatFeature nodes...")
        malcat_nodes = {}
        malcat_edges = []  # (Sample)-[:HAS_MALCAT]->(MalcatFeature)
        for idx, row in malcat_features.iterrows():
            h = row["hash"]
            if h not in final_hash_set: continue  # [New] filter
            for col in row.index:
                if col == "hash":
                    continue
                if col not in malcat_nodes:
                    malcat_nodes[col] = {"name": col}

                if int(row[col]) != 0:
                    malcat_edges.append({
                        "hash": h,
                        "feature_name": col
                    })

        logging.info("[8] Export ExifTag nodes...")
        exif_nodes = {}
        exif_edges = [] # (Sample)-[:HAS_EXIFTAG]->(ExifTag)
        # logging.info(f"{exif_features}")
        for idx, row in exif_features.iterrows():
            h = row["hash"]
            if h not in final_hash_set: continue  # [New] filter
            for tag in row.index:
                if tag == "hash":
                    continue

                tag_val = row[tag]
                key = f"{tag}:{tag_val}"

                if key not in exif_nodes:
                    exif_nodes[key] = {
                        "tag_name": tag,
                        "normalized_value": tag_val
                    }

                exif_edges.append({
                    "hash": h,
                    "tag_name": tag,
                    "normalized_value": tag_val
                })
        
        logging.info("[9] Export IoCNorm nodes...")
        # IoC features are already one-hot in joined_df / process_and_merge
        ioc_cols = [c for c in joined_df.columns if c != 'hash']
        ioc_nodes = {}
        ioc_edges = []

        for idx, row in joined_df.iterrows():
            h = row["hash"]
            if h not in final_hash_set: continue  # [New] filter
            for col in ioc_cols:
                val_data = row[col]
                if not val_data: 
                    continue

                if isinstance(val_data, list):
                    for val in val_data:
                        # Skip empty strings
                        if not val: continue
                        
                        # Build node key: "IPv4:1.1.1.1"
                        # Note: use a type prefix to distinguish from other graph nodes
                        key = f"{col}:{val}"
                        
                        if key not in ioc_nodes:
                            ioc_nodes[key] = {
                                "ioc_type": col,        # e.g., IPv4
                                "normalized_value": val # e.g., 1.1.1.1
                            }

                        ioc_edges.append({
                            "hash": h,
                            "ioc_type": col,
                            "normalized_value": val
                        })
                # If non-list data appears (e.g., missed by cleaning), treat as single value
                elif isinstance(val_data, str):
                    if not val_data: continue
                    key = f"{col}:{val_data}"
                    if key not in ioc_nodes:
                        ioc_nodes[key] = {"ioc_type": col, "normalized_value": val_data}
                    ioc_edges.append({"hash": h, "ioc_type": col, "normalized_value": val_data})

        logging.info("[10] Export StringGroup nodes...")
        string_nodes = {}
        string_edges = []
        string_vec_store = VectorStore(dim=384, metric="euclidean")

        # Create one string node per sample
        str_df_indexed = string_embedding_df_features.set_index('hash')
        str_emb_cols = [c for c in string_embedding_df_features.columns if c != 'hash']
        str_vectors_from_combined = combined_df[str_emb_cols].values.astype(np.float32)
        str_keys = [f"string::{h}" for h in final_hashes]
        string_vec_store.add_vectors_batch(str_keys, str_vectors_from_combined)
        str_ids = list(range(len(final_hashes)))
        

        for h, vec_id in zip(final_hashes, str_ids):
            current_raw_strings = raw_strings_map.get(h, [])
            current_raw_strings = current_raw_strings[:100] 
            string_nodes[h] = {
                "group_id": h,
                "centroid_id": vec_id,
                "raw_strings": current_raw_strings
            }
            string_edges.append({
                "hash": h,
                "group_id": h
            })

        logging.info("[11] Saving KG compatible files...")

        json.dump(samples_out, open(f"{output_dir}/samples.json", "w"), indent=2)
        json.dump(malcat_nodes, open(f"{output_dir}/malcat_nodes.json", "w"), indent=2)
        json.dump(malcat_edges, open(f"{output_dir}/malcat_edges.json", "w"), indent=2)

        json.dump(exif_nodes, open(f"{output_dir}/exif_nodes.json", "w"), indent=2)
        json.dump(exif_edges, open(f"{output_dir}/exif_edges.json", "w"), indent=2)

        json.dump(ioc_nodes, open(f"{output_dir}/ioc_nodes.json", "w"), indent=2)
        json.dump(ioc_edges, open(f"{output_dir}/ioc_edges.json", "w"), indent=2)

        json.dump(string_nodes, open(f"{output_dir}/string_nodes.json", "w"), indent=2)
        json.dump(string_edges, open(f"{output_dir}/string_edges.json", "w"), indent=2)

        # ---- Save FAISS ----
        np.save(f"{output_dir}/combined_vectors.npy", combined_matrix)
        json.dump(sample_vec_store.hash_to_id, open(f"{output_dir}/hash_to_vector_id.json", "w"), indent=2)
        faiss.write_index(sample_vec_store.index, f"{output_dir}/combined.index")

        faiss.write_index(string_vec_store.index, f"{output_dir}/string.index")
        np.save(f"{output_dir}/string_vectors.npy", str_vectors_from_combined)
        json.dump(string_vec_store.hash_to_id, open(f"{output_dir}/string_id_map.json", "w"), indent=2)

        logging.info("[✓] Feature preparation finished.  All files exported.")


# =====================================================
# Entry point
# =====================================================
if __name__ == "__main__":
    output_dir = "./samples/vector_store_benchmark"
    preparer = FeaturesPreparer()
    preparer.run(output_dir)
