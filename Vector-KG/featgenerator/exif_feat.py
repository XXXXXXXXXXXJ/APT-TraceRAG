import json
import os
import warnings

import dateparser as dp
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers import AutoTokenizer, AutoModel
import concurrent

warnings.filterwarnings("ignore")


import numpy as np

# from util import Util, DataProcessor
# from config import Config

from .util import Util, DataProcessor
from .config import Config



class ExifFeatures():
    def __init__(self):
        conf = Config()
        self.root_dir = conf.get_root_dir()
        self.exif_filename = conf.get_exif_file_name()
    
    def get_dataset(self, root_dir, exif_filename):
        hashes_df = []

        def add_to_set(data_dict, prefix, key, target_set):
            full_key = f"{prefix}:{key}"
            if full_key in data_dict:
                target_set.add(str(data_dict[full_key]))

        def process_exif_data(data, hashed_obj):
            # print(f"[DEBUG-INTERNAL] Keys found: {list(data.keys())[:]}")
            for prop, value in data.items():
                if prop in ['CompanyName', 'Company', 'Software']:
                    hashed_obj['company_name'].add(str(value))
                elif prop in ['LanguageCode', 'Language']:
                    hashed_obj['language'].add(str(value))
                elif prop in ['OSVersion', 'OperatingSystem', 'System']:
                    hashed_obj['os_name'].add(str(value))
                elif prop in ['InternalName', 'Title', 'IconFileName', 'TargetFileDOSName', 'ZipFileName', 'ArchivedFileName']:
                    hashed_obj['filenames'].add(str(value))
                elif prop in ['Keywords', 'Subject', 'Comments', 'ZipFileComment', 'VolumeName']:
                    hashed_obj['keywords'].add(str(value))
                elif prop in ['Author', 'Creator', 'Producer', 'MachineID', 'Originator']:
                    hashed_obj['author_name'].add(str(value))
                elif prop == 'Copyright':
                    hashed_obj['copyright'].add(str(value))

        for subdir, dirs, files in os.walk(root_dir):
            if not files:
                continue

            for file in files:
                file_hash = os.path.basename(os.path.normpath(subdir))
                if file != exif_filename:
                    continue
                exif_file = os.path.join(subdir, exif_filename)

                hashed_obj = {
                    "hash": file_hash,
                    "filesize": "",
                    "filetype": "",
                    "company_name": set(),
                    "os_name": set(),
                    "os_version": set(),
                    "language": set(),
                    "author_name": set(),
                    "filenames": set(),
                    "keywords": set(),
                    "copyright": set()
                }

                try:
                    if os.path.isfile(exif_file):
                        with open(exif_file) as f:
                            data = json.load(f)
                        if data:
                            filesize = data.get('FileSize', 0)
                            filetype = data.get('FileType', '')
                            hashed_obj['filesize'] = filesize
                            hashed_obj['filetype'] = filetype

                            process_exif_data(data, hashed_obj)

                    hashes_df.append(hashed_obj)
                except Exception as e:
                    print('####### Exception occurred:', e)
                    print(file_hash)

        return hashes_df

    def get_features(self):
        hashed_obj = self.get_dataset(self.root_dir, self.exif_filename)
        if len(hashed_obj) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(hashed_obj)
        df_features = df[['hash','filetype', 'filesize']]
        # In the json version of exif tool, we no longer have file size as numeric
        # df_features["filesize"] = pd.to_numeric(df_features["filesize"])
        df_features["filenames"] = df['filenames'].apply(lambda x: sorted(x))
        df_features["company_name"] = df['company_name'].apply(lambda x: sorted(x))
        df_features["author_name"] = df['author_name'].apply(lambda x: sorted(x))
        df_features["keywords"] = df['keywords'].apply(lambda x: sorted(x))
        df_features["copyright"] = df['copyright'].apply(lambda x: sorted(x))

        df_features['lexical_filename'] = df_features['filenames'].apply(lambda x: Util().lexical_features(x))
        df_features['lexical_company_name'] = df_features['company_name'].apply(lambda x: Util().lexical_features(x))
        df_features['lexical_author_name'] = df_features['author_name'].apply(lambda x: Util().lexical_features(x))
        df_features['lexical_keywords'] = df_features['keywords'].apply(lambda x: Util().lexical_features(x))
        df_features['lexical_copyright'] = df_features['copyright'].apply(lambda x: Util().lexical_features(x))
        
        
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_filename']].to_dict(orient="records"), record_prefix="lexical_filename_", meta_prefix="lexical_filename_"))
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_company_name']].to_dict(orient="records"), record_prefix="company_name_", meta_prefix="company_name_"))
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_author_name']].to_dict(orient="records"), record_prefix="author_name_", meta_prefix="author_name_"))
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_keywords']].to_dict(orient="records"), record_prefix="keywords_", meta_prefix="keywords_"))
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_copyright']].to_dict(orient="records"), record_prefix="copyright_", meta_prefix="copyright_"))
        

        df_features.drop(columns=['lexical_filename',
        'lexical_company_name',
        'lexical_keywords',
        'lexical_author_name',
        'lexical_copyright'], inplace=True)

        return df_features

    def get_normalized_features(self):
        exf = self.get_features()
        # Transformer models
        model, tokenizer = Util().get_sentence_tok_model()
        data_proc = DataProcessor()
        data_to_encode = dict()
        absolute_features = [
            'filenames', 
            'author_name', 
            'keywords',
            'copyright',
            'company_name'
        ]
        for elem in absolute_features:
            exf_data_proc = data_proc.string_feature_embed_similarity(exf, elem, tokenizer, model, 0.7, cardinality_lower_bound=0,cardinality_ratio=1.0)
            df_with_correct_name = exf_data_proc.to_frame(name=elem)
            data_to_encode[elem] = data_proc.one_hot_encode_list_column(df_with_correct_name, elem, True)
            df = data_to_encode[elem].reset_index(drop=True)
            unnamed_cols = [col for col in df.columns if col is None or col.strip() == ""]
            # Dropping unnamed columns
            df_cleaned = df.drop(columns=unnamed_cols).add_prefix(elem)
            data_to_encode[elem] = df_cleaned.copy()
        merged_exf = pd.concat(data_to_encode.values(), axis = 1)
        merged_exf['hash'] = exf['hash'].astype(str)
        return merged_exf
    
# if __name__ == "__main__":
#     exif_feautres = ExifFeatures()
#     exf = exif_feautres.get_normalized_features()
#     print("EXIF features shape:", exf.shape)