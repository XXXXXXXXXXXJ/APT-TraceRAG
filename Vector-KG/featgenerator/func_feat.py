import json
import os
import warnings
import pandas as pd
from collections import Counter

from .config import Config
from .util import DataProcessor, Util

warnings.filterwarnings("ignore")

class FunctionFeatures:
    def __init__(self):
        conf = Config()
        self.root_dir = conf.get_root_dir()
        self.func_filename = "function_ports.json"
        
        # === Dimensionality control parameters (for get_normalized_features) ===
        self.TARGET_K_IMPORTS = 80 
        self.TARGET_K_EXPORTS = 20
        
        # === Semantic similarity parameters (for get_features) ===
        self.SIMILARITY_THRESHOLD = 0.85

    def get_dataset(self, root_dir, func_filename):
        hashes_df = []

        def process_function_data(data, hashed_obj):
            # Handle exports
            if "exports" in data and isinstance(data["exports"], list):
                clean_exports = [str(e) for e in data["exports"] if e]
                hashed_obj["export_raw"] = clean_exports
            
            # Handle imports
            if "imports" in data and isinstance(data["imports"], dict):
                for dll, funcs in data["imports"].items():
                    # Basic cleanup: remove extension and lowercase
                    dll_clean = dll.lower().replace(".dll", "")
                    if isinstance(funcs, list):
                        for func in funcs:
                            if func:
                                full_import = f"{dll_clean}:{func}"
                                hashed_obj["import_raw"].append(full_import)

        for subdir, dirs, files in os.walk(root_dir):
            if not files:
                continue

            for file in files:
                file_hash = os.path.basename(os.path.normpath(subdir))
                if file != func_filename:
                    continue
                func_file_path = os.path.join(subdir, func_filename)

                # Initialize as lists for downstream processing
                hashed_obj = {
                    "hash": file_hash,
                    "import_raw": [], 
                    "export_raw": []
                }

                try:
                    if os.path.isfile(func_file_path):
                        with open(func_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            data = json.load(f)
                        if data:
                            process_function_data(data, hashed_obj)

                    hashes_df.append(hashed_obj)
                except Exception as e:
                    print(f'[ERROR] FunctionFeatures error for {file_hash}: {e}')

        return hashes_df

    def get_features(self):
        """
        Get semantically merged feature data.
        This result is suitable for knowledge-graph construction (normalized names, no strong filtering).
        """
        raw_data = self.get_dataset(self.root_dir, self.func_filename)
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Prepare tools
        print("[+] Loading Transformer for Function Semantic Merging...")
        model, tokenizer = Util().get_sentence_tok_model()
        data_proc = DataProcessor()

        # Perform semantic merging for Import and Export columns
        # Example: 'LoadLibraryA' and 'LoadLibraryW' may be merged into 'LoadLibrary'
        target_cols = ['import_raw', 'export_raw']
        
        for col in target_cols:
            print(f"[+] Merging Semantically Similar Functions for {col}...")
            # string_feature_embed_similarity returns the merged Series
            merged_series = data_proc.string_feature_embed_similarity(
                df, 
                col, 
                tokenizer, 
                model, 
                similarity_threshold=self.SIMILARITY_THRESHOLD,
                cardinality_lower_bound=0,
                cardinality_ratio=1.0
            )
            df[col] = merged_series

        return df

    def _select_discriminative_features(self, df, col_name, k, upper_ratio=1.0, min_occurrence=2):
        """
        Helper: frequency-based feature selection (Top-K / mid-frequency retention).
        """
        all_items = []
        for items in df[col_name]:
            all_items.extend(list(set(items)))  # De-duplicate within a sample
        
        if not all_items:
            return set()

        total_samples = len(df)
        counter = Counter(all_items)
        
        upper_limit = total_samples * upper_ratio
        lower_limit = min_occurrence

        # Select candidates
        candidates = {
            feat: count for feat, count in counter.items() 
            if lower_limit <= count <= upper_limit
        }
        
        # Select Top-K
        selected_features = set(
            sorted(candidates, key=candidates.get, reverse=True)[:k]
        )
        
        # Filter DataFrame column
        return df[col_name].apply(
            lambda x: list(set([i for i in x if i in selected_features]))
        )

    def get_normalized_features(self):
        """
        Get final features for vector computation (semantic merge + strong filtering + one-hot).
        """
        # 1. Get semantically merged data
        df = self.get_features()
        if df.empty:
            return pd.DataFrame(columns=['hash'])
        
        data_proc = DataProcessor()

        # 2. Compute numeric features (counts from merged data)
        df['cnt_import_funcs'] = df['import_raw'].apply(len)
        df['cnt_export_funcs'] = df['export_raw'].apply(len)

        # 3. Strong filtering (dimensionality control)
        # Imports: remove common libraries (>70%), keep Top 80
        filtered_imports = self._select_discriminative_features(
            df, 'import_raw', 
            k=self.TARGET_K_IMPORTS, 
            upper_ratio=0.70, 
            min_occurrence=4
        )
        
        # Exports: keep Top 20
        filtered_exports = self._select_discriminative_features(
            df, 'export_raw', 
            k=self.TARGET_K_EXPORTS, 
            upper_ratio=1.0, 
            min_occurrence=2
        )

        # 4. One-hot encoding
        # Use a temporary DataFrame for encoding to avoid mutating the original df
        temp_df = pd.DataFrame({
            'import_filtered': filtered_imports,
            'export_filtered': filtered_exports
        })
        
        df_imports_oh = data_proc.one_hot_encode_list_column(temp_df, 'import_filtered')
        df_imports_oh = df_imports_oh.add_prefix("import_functions_")

        df_exports_oh = data_proc.one_hot_encode_list_column(temp_df, 'export_filtered')
        df_exports_oh = df_exports_oh.add_prefix("export_functions_")

        # 5. Merge outputs
        df_hash = df[['hash']].copy()
        df_hash['hash'] = df_hash['hash'].astype(str)
        df_counts = df[['cnt_import_funcs', 'cnt_export_funcs']]

        result = pd.concat([df_hash, df_counts, df_imports_oh, df_exports_oh], axis=1)
        result = result.fillna(0)
        
        return result