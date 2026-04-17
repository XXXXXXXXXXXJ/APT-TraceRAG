import ipaddress
import json
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from bitcoinlib.keys import Address
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import AgglomerativeClustering

from .exif_feat import ExifFeatures
from .malcat import MalcatFeatures

from .config import Config
from .floss_general_feat import FlossFeatures
from .util import DataProcessor, StringProcessing, Util


import os
import json
import pandas as pd
from copy import deepcopy

import os
import json
import pandas as pd
from copy import deepcopy

class GroupAttributionFeatures:
    def __init__(self):
        conf = Config()
        self.root_folder = conf.get_root_dir()
        self.raw_dataset_paths = conf.get_regex_filename()
        self.regex_result_path = conf.get_regex_filename()

        self.default_normalized = {
            "hash": "",
            "service_serial_number": set(),
            "asn": set(),
            "country_code": set(),
            "bgp_prefix": set(),
            "issuer_organization": set(),
            "cert_finger_print": set(),
        }

        self.default_regex_keys = {
            "hash": "",
            "IPv4": [], 
            "LinuxFilePath": [],
            "MD5": [], 
            "Email": [],
            "URL": [], "ipaddress": [], "FilePath_1": [], "FilePath_2": [],
            "md5": [], "sha1": [], "sha256": [], "Ethereum": [], "Bitcoin": [],
            "EmailAddress": [], "SlackToken": [], "RSAprivatekey": [],
            "SSHDSAprivatekey": [], "SSHECprivatekey": [], "PGPprivatekeyblock": [],
            "GitHub": [], "GenericAPIKey": [], "GoogleAPIKey": [],
            "GoogleGCPServiceaccount": [], "GoogleGmailAPIKey": [],
            "GoogleGmailOAuth": [], "PayPalBraintreeAccessToken": [],
            "TwitterAccessToken": [], "TwitterOAuth": []
        }

    def _clean_list(self, items):
        if isinstance(items, list):
            return list(set(filter(None, items)))
        return items if items is not None else ""

    def get_ip_domain(self, normalize, obj):
        asn = obj.get("autonomous_system", {})
        location = obj.get("location", {})

        if asn.get("asn"):
            normalize["asn"].add(str(asn["asn"]))
        if asn.get("bgp_prefix"):
            normalize["bgp_prefix"].add(asn["bgp_prefix"])
        if location.get("country_code"):
            normalize["country_code"].add(location["country_code"])

        for serv in obj.get("services", []):
            if serv.get("certificate"):
                normalize["service_serial_number"].add(serv["certificate"])
        return normalize

    def get_hashes(self):
        raw_dataset_hashes = os.listdir(self.root_folder)
        raw_dataset_paths = [
            f"{self.root_folder}{file}{self.raw_dataset_paths}" for file in raw_dataset_hashes
        ]
        return raw_dataset_hashes, raw_dataset_paths


    def normalize_dataset(self, df: pd.DataFrame) -> list[dict]:
        normalized_results = []
        for _, row in df.iterrows():
            normalize = deepcopy(self.default_normalized)
            normalize["hash"] = row.get("hash", "")

            for domain_item in row.get("domain_data", []):
                if domain_item:
                    for sub_item in domain_item:
                        for domain_dict in sub_item or []:
                            if isinstance(domain_dict, dict):
                                self.get_ip_domain(normalize, domain_dict)

            for ip_item in row.get("ip_data", []):
                if ip_item:
                    self.get_ip_domain(normalize, ip_item)

            for cert_item in row.get("cert_data", []):
                for cert_dict in cert_item or []:
                    if isinstance(cert_dict, dict):
                        fingerprint = cert_dict.get("fingerprint_sha256")
                        if fingerprint:
                            normalize["cert_finger_print"].add(fingerprint)
                        orgs = cert_dict.get("issuer_organization", [])
                        if orgs and isinstance(orgs, list):
                            normalize["issuer_organization"].add(orgs[0])

            normalized_results.append(normalize)
        return normalized_results

    def get_regex_dataset(self, hashes, drop_empty_rows=True):
        regex_results = []
        for f_hash in hashes:
            regex_data = deepcopy(self.default_regex_keys)
            regex_data["hash"] = f_hash
            try:
                regex_path = os.path.join(self.root_folder, f_hash, self.regex_result_path)
                with open(regex_path, "r") as f:
                    obj = json.load(f)

                file_data = obj.get(f_hash, {})
                for key in self.default_regex_keys:
                    if key in file_data:
                        regex_data[key] = self._clean_list(file_data[key])
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[ERROR] Regex load failed for {f_hash}: {e}")

            non_empty = sum(bool(regex_data[k]) for k in regex_data if k != "hash")
            if not drop_empty_rows or non_empty >= 2:
                regex_results.append(regex_data)
        return regex_results

    def ensure_all_columns(self, df, required_columns):
        for col in required_columns:
            if col not in df.columns:
                print(f"[WARN] Column '{col}' not found — filling with defaults.")
                df[col] = [[] for _ in range(len(df))]
        return df

    def generate_derived_columns(self, data):
        required = ["Email"]
        self.ensure_all_columns(data, required)

        data["EmailAddressUsername"] = data["Email"].apply(
            lambda x: [i.split("@")[0] for i in x] if isinstance(x, list) else []
        )
        data["EmailAddressDomain"] = data["Email"].apply(
            lambda x: [i.split("@")[1] for i in x if "@" in i] if isinstance(x, list) else []
        )
        return data

def clean_up_data(data):
    """We assume that all datasets will contain sets/lists of elements. Thus, we perform a cleanup here to fix cells that have null values.
    Args:
        data: The dataframe to cleanup.

    Returns:
        A cleaned up dataset.
    """
    df = data.copy()

    # Identify the type of iterable in each column and replace NaNs
    for col in df.columns:
        # Identify the first non-NaN element
        first_non_nan = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(first_non_nan, list):
            # Adjusting the lambda to avoid ambiguous truth value
            df[col] = df[col].apply(
                lambda x: (
                    []
                    if (isinstance(x, float) and np.isnan(x))
                    or (isinstance(x, (list, set)) and not x)
                    else x
                )
            )
        elif isinstance(first_non_nan, set):
            # Adjusting the lambda to avoid ambiguous truth value
            df[col] = df[col].apply(
                lambda x: (
                    set()
                    if (isinstance(x, float) and np.isnan(x))
                    or (isinstance(x, (list, set)) and not x)
                    else x
                )
            )
            # Add further conditions if more types need to be handled
        else:
            # Decide on a default type or leave the column as-is
            # If you want to replace NaN with an empty list by default, uncomment the next line
            # df[col].fillna(value=[], inplace=True)
            pass
    return df


class FeatureProcessor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.tokenizer = Util().get_sentence_tok_model()[1]
        self.model = Util().get_sentence_tok_model()[0]

    SIMILARITY_THRESHOLD = {"default": 0.7, "LinuxPathClean": 0.8}

    EXCEPTION_COLUMNS = ["hash"]

    def process_features(self, joined_df):
        data = self.select_and_clean_data(joined_df)
        data = self.generate_derived_columns(data.copy())
        embeddings_data = self.compute_embeddings(data)
        normalized_features = self.normalize_absolute_features(embeddings_data)
        encoded_features = self.encode_categorical_features(normalized_features)
        final_df = self.merge_features(encoded_features)
        return final_df, embeddings_data, normalized_features

    def select_and_clean_data(self, df):
        return clean_up_data(df)

    def generate_derived_columns(self, data):
        required_columns = ["Email", "LinuxFilePath", "IPv4", "Bitcoin", "MD5"]
        for col in required_columns:
            if col not in data.columns:
                print(f"[WARN] Column '{col}' not found — filling with empty lists.")
                data[col] = [[] for _ in range(len(data))]

        data["EmailAddressUsername"] = data["Email"].apply(
            lambda x: [i.split("@")[0] for i in x] if isinstance(x, list) else []
        )
        data["EmailAddressDomain"] = data["Email"].apply(
            lambda x: [i.split("@")[1] for i in x if "@" in i] if isinstance(x, list) else []
        )
        data["LinuxPathClean"] = data["LinuxFilePath"].apply(
            lambda x: [i for i in x if self.data_processor.is_valid_unix_path(i)] if isinstance(x, list) else []
        )
        data["IPAddressClean"] = data["IPv4"].apply(
            lambda x: self.data_processor.validate_ip_addresses(x) if isinstance(x, list) else []
        )
        data["BitCoinClean"] = data["Bitcoin"].apply(
            lambda x: filter_valid_bitcoin_addresses(x) if isinstance(x, list) else []
        )
        data["MD5"] = data["MD5"].apply(
            lambda x: filter_valid_md5(x) if isinstance(x, list) else []
        )
        return data

    def compute_embeddings(self, data):
        result = {}
        for column in data.columns:
            if column in self.EXCEPTION_COLUMNS:
                continue
            try:
                sim_df = self.data_processor.string_feature_embed_similarity(
                    data,
                    column,
                    self.tokenizer,
                    self.model,
                    self.SIMILARITY_THRESHOLD.get(column, self.SIMILARITY_THRESHOLD["default"]),
                )
                if isinstance(sim_df, pd.Series):
                    sim_df = sim_df.to_frame(name=column)
                elif isinstance(sim_df, pd.DataFrame):
                    if len(sim_df.columns) == 1 and sim_df.columns[0] != column:
                        sim_df.columns = [column]
                else:
                    sim_df = pd.DataFrame()
                if column not in sim_df.columns and not sim_df.empty:
                    print(f"[WARN] Embedding output missing column '{column}' — filling with empty lists.")
                    sim_df[column] = [[] for _ in range(len(data))]
                result[column] = sim_df
            except Exception as e:
                print(f"[ERROR] Failed to embed feature '{column}': {e}")
                result[column] = pd.DataFrame({column: [[] for _ in range(len(data))]})
        return result

    def normalize_absolute_features(self, embeddings_data):
        normalized_features = {}
        for column, df in embeddings_data.items():
            if column not in df.columns:
                print(f"[WARN] Column '{column}' not found in embeddings — skipping normalization.")
                normalized_features[column] = pd.DataFrame({column: [[] for _ in range(len(df))]})
                continue
            try:
                if column == "MD5":
                    normalized_column_data = self.data_processor.normalize_column_using_popularity(
                        df, column, cardinality_lower_bound=4
                    )
                else:
                    normalized_column_data = self.data_processor.normalize_column_using_popularity(
                        df, column
                    )
                if isinstance(normalized_column_data, pd.Series):
                    normalized_features[column] = normalized_column_data.to_frame(name=column)
                elif isinstance(normalized_column_data, pd.DataFrame):
                    # If already a DataFrame, ensure column names are correct
                    normalized_features[column] = normalized_column_data
                    if column not in normalized_features[column].columns and not normalized_features[column].empty:
                        # Force rename the first column
                        normalized_features[column].columns = [column]
                else:
                    # Handle lists or other cases
                    normalized_features[column] = pd.DataFrame(normalized_column_data, columns=[column], index=df.index)
            except Exception as e:
                print(f"[ERROR] Normalization failed for column '{column}': {e}")
                normalized_features[column] = pd.DataFrame({column: [[] for _ in range(len(df))]}, index=df.index)
        return normalized_features

    def encode_categorical_features(self, normalized_features):
        encoded_features = {}
        for column, df in normalized_features.items():
            encoded_df = self.data_processor.one_hot_encode_list_column(df, column)
            encoded_features[column] = encoded_df
        return encoded_features

    def merge_features(self, feature_data_frames):
        return pd.concat(feature_data_frames.values(), axis=1)


class StringEmbeddingProcessor:
    """
    Processes string data by generating embeddings using a Sentence Transformer model,
    optionally filtering based on a provided DataFrame, and preparing the data for clustering.

    Args:
        sentence_transformer_model (str): The model name for Sentence Transformers to generate embeddings.
        joined_df (Optional[pd.DataFrame]): A DataFrame to filter the embeddings based on hash values. Defaults to None.

    Attributes:
        floss_features (FlossFeatures): Instance of FlossFeatures to load raw string datasets.
        string_processor (StringProcessing): Instance of StringProcessing to process raw strings.
        embedding_model (SentenceTransformer): Sentence Transformer model for generating string embeddings.
        joined_df (Optional[pd.DataFrame]): DataFrame used for filtering embeddings.    
    """
    def __init__(
        self,
        sentence_transformer_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        joined_df=None,
    ):
        self.floss_features = FlossFeatures()
        self.string_processor = StringProcessing()
        self.embedding_model = SentenceTransformer(Config().get_string_model())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = self.embedding_model.to(device)
        self.joined_df = joined_df
        self.cached_raw_strings = {}

    def process(self):
        """
        Processes the raw string dataset to generate embeddings, optionally filters them,
        and prepares the data for clustering.

        Returns:
            pd.DataFrame: A DataFrame containing the embeddings and associated hashes, 
            filtered by `joined_df` if provided.
        """
        # Load and filter the dataset
        raw_strings_dataset = self.floss_features.get_dataset(
            self.floss_features.root_dir, self.floss_features.floss_filename
        )
        # print(f"[+] Loaded raw strings dataset with {len(raw_strings_dataset)} entries.")
        raw_string_df = pd.DataFrame(raw_strings_dataset, columns=["hash", "strings"])

        # Process strings
        tqdm.pandas(desc="Filtering Strings") 
        valid_strings_series = raw_string_df["strings"].progress_apply(
            lambda x: self.string_processor.process_strings(strings=list(x))
        )
        self.cached_raw_strings = dict(zip(raw_string_df['hash'], valid_strings_series))
        raw_string_df["filtered_strings"] = valid_strings_series.apply(lambda x: " , ".join(x))

        # Generate embeddings
        all_texts = raw_string_df["filtered_strings"].tolist()
        embeddings_array = self.embedding_model.encode(
            all_texts, 
            batch_size=128, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        string_embeddings_initial = embeddings_array.tolist()
        embeddings_df = pd.DataFrame(string_embeddings_initial)
        embeddings_df["hash"] = raw_string_df["hash"]

        # Match embeddings with the joined dataframe, if provided
        if self.joined_df is not None:
            embeddings_df = embeddings_df[
                embeddings_df["hash"].isin(self.joined_df["hash"])
            ]

        # Prepare data for clustering
        # X_string_embedding = embeddings_df.drop(columns=["hash"])

        return embeddings_df
    def get_raw_strings(self):
        """
        Return the cleaned raw-strings dictionary.
        Returns:
            dict: {hash: [string1, string2, ...]}
        """
        return self.cached_raw_strings


def convert_ipv4_to_ipv6_single(cidr_v4):
    try:
        network_v4 = ipaddress.ip_network(cidr_v4, strict=False)
        # Check if the network is IPv4
        if network_v4.version == 4:
            # Convert to IPv6 using IPv4-mapped format
            ipv4_address = network_v4.network_address.exploded
            ipv6_address = f"::ffff:{ipv4_address}"
            ipv6_cidr = f"{ipv6_address}/{network_v4.prefixlen + 96}"  # Adjust the prefix length
            return ipv6_cidr
    except ValueError:
        pass  # Handle invalid CIDR format or non-IPv4 CIDR

    return None


def find_top_level_ipv6_cidrs(cidr_list):
    # # Example usage:
    # cidr_list = ["112.199.88.0/12", "10.0.0.0/8", "10.0.0.0/32", "192.168.1.0/32", "172.16.0.0/12", "172.16", "2620:100:601f::/48"]
    # top_level_ipv6_cidrs = find_top_level_ipv6_cidrs(cidr_list)
    # print("Top-level IPv6 CIDRs:", top_level_ipv6_cidrs)
    top_level_ipv6_cidrs = []
    cidr_objects = []

    # Filter out invalid or empty CIDRs and separate them by version
    ipv4_cidrs = []
    ipv6_cidrs = []

    for cidr in cidr_list:
        if cidr and "/" in cidr:  # Check for non-empty and valid format
            try:
                network = ipaddress.ip_network(cidr, strict=False)
                if network.version == 4:
                    ipv4_cidrs.append(network)
                elif network.version == 6:
                    ipv6_cidrs.append(network)
            except ValueError:
                pass  # Skip invalid CIDR prefixes

    for cidr1 in ipv4_cidrs:
        is_top_level = True
        for cidr2 in ipv4_cidrs:
            if cidr1 != cidr2 and cidr1.subnet_of(cidr2):
                is_top_level = False
                break

        if is_top_level:
            ipv6_cidr = convert_ipv4_to_ipv6_single(str(cidr1))
            if ipv6_cidr:
                top_level_ipv6_cidrs.append(ipv6_cidr)

    return top_level_ipv6_cidrs


def is_valid_bitcoin_address(address):
    try:
        val_add = Address.parse(address)
    except Exception:
        return False
    return True


def filter_valid_bitcoin_addresses(addresses):
    valid_addresses = [
        address for address in addresses if is_valid_bitcoin_address(address)
    ]
    return valid_addresses


def is_valid_md5(input_string):
    # Lowercasing the input string
    input_string = input_string.lower()

    # Regular expression pattern to identify MD5 hashes
    md5_pattern = re.compile(r"^[a-f0-9]{32}$")

    # Check if the input string matches the MD5 pattern
    if not md5_pattern.match(input_string):
        return False

    # Check if the input string contains at least one hexadecimal character a-f
    hex_char_pattern = re.compile(r"[a-f]")
    if not hex_char_pattern.search(input_string):
        return False

    # Check if the input string contains at least one numeral 0-9
    numeral_pattern = re.compile(r"[0-9]")
    return bool(numeral_pattern.search(input_string))


def filter_valid_md5(md5_list):
    if md5_list is None:
        return []
    if isinstance(md5_list, str):  # Handling single MD5 hash as string
        return [md5_list] if is_valid_md5(md5_list) else []
    return [md5.lower() for md5 in md5_list if is_valid_md5(md5)]


def drop_empty_rows(df: pd.DataFrame, exclude_columns: list = ["hash"]) -> pd.DataFrame:
    """
    Drops rows from a DataFrame where all columns except those specified in exclude_columns are empty.

    Args:
    - df: The input DataFrame.
    - exclude_columns: A list of column names to exclude from the emptiness check.

    Returns:
    - A DataFrame with the empty rows dropped.
    """
    # Determine columns to check for emptiness
    columns_to_check = df.columns.difference(exclude_columns)

    # Identify rows where all values in columns_to_check are empty or NaN
    rows_with_values = df[columns_to_check].notnull() & df[columns_to_check].astype(
        bool
    )
    non_empty_rows = rows_with_values.any(axis=1)

    # Drop rows where all columns in columns_to_check are empty or NaN
    filtered_df = df[non_empty_rows].copy()

    return filtered_df


def load_and_prepare_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares datasets from various sources including adversary mapping, Exif features,
    Malcat features, and Group Attribution features. It normalizes and merges these datasets for further processing.

    Returns:
        Tuple containing DataFrames for Exif features, Malcat features, joined Group Attribution features,
        and the adversary dataset.
    """
    # Load configurations and datasets
    conf = Config()
    adversary_dataset = pd.read_csv(conf.get_adversary_mapping())
    # adversary_dataset['hash'] = adversary_dataset['sha256'].copy()
    #if "hash" in adversary_dataset.columns and "sha256" in adversary_dataset.columns:
    #    adversary_dataset = adversary_dataset.drop(columns=['hash'])
    #    adversary_dataset = adversary_dataset.rename(columns={'sha256': 'hash'})

    # Load and normalize Exif and Malcat features
    exif_features = ExifFeatures().get_normalized_features().assign(hash=lambda df: df['hash'].astype(str))
    malcat_features = MalcatFeatures().get_features().assign(hash=lambda df: df['hash'].astype(str))

    # Load and normalize Group Attribution features
    group_attr = GroupAttributionFeatures()
    feature_hashes, _ = group_attr.get_hashes()
    # regex_features_df = pd.DataFrame(group_attr.get_regex_dataset(feature_hashes, False))

    # Merge datasets
    joined_df = pd.DataFrame(group_attr.get_regex_dataset(feature_hashes, False))

    return exif_features, malcat_features, joined_df, adversary_dataset

def process_and_merge_features(
    exif_features: pd.DataFrame, 
    malcat_features: pd.DataFrame, 
    joined_df: pd.DataFrame, 
    adversary_dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Processes and merges features from Exif, Malcat, and Group Attribution with the adversary dataset.
    It utilizes a feature processor to merge and further process the data.

    Args:
        exif_features: DataFrame containing normalized Exif features.
        malcat_features: DataFrame containing Malcat features.
        joined_df: DataFrame containing joined Group Attribution features.
        adversary_dataset: DataFrame containing adversary dataset information.

    Returns:
        A DataFrame with all features merged and processed, ready for analysis.
    """
    # Initialize the feature processor and process joined_df
    feat_processor = FeatureProcessor()
    merged_result, _, _ = feat_processor.process_features(joined_df)
    merged_result['hash'] = joined_df['hash']
    # Merge Exif and Malcat features with the processed result
    base_df = adversary_dataset[['hash', 'Normalized_Tag']].merge(
        joined_df[['hash']], on='hash', how='inner'
    )
    all_features = base_df.merge(exif_features, on="hash", how="left")
    all_features = all_features.merge(malcat_features, on="hash", how="left")
    all_features = all_features.merge(merged_result, on="hash", how="left")
    cols_to_fill = [c for c in all_features.columns if c not in ['hash', 'Normalized_Tag']]
    all_features[cols_to_fill] = all_features[cols_to_fill].fillna(0)

    # Merge with adversary dataset and drop the 'hash' column
    print(f"[+] Features merged. Shape: {all_features.shape}")
    print(f"[+] Unique Tags: {all_features['Normalized_Tag'].nunique()}")

    return all_features
