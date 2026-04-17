import concurrent
import concurrent.futures
import csv
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Optional, Tuple, Union
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import regex
import textstat
import torch
import torch.nn.functional as torch_ff
import tqdm
from nltk.corpus import words
from scipy.signal import normalize
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    QuantileTransformer,
    label_binarize,
    normalize,
)
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from .config import Config

from .floss_general_feat import FlossFeatures


class Util:
    
    def lexical_features(self, test_data):
        def get_lexical_feature(method_name, value):
            try:
                method = getattr(textstat, method_name)
                return method(value)
            # Attribute error means function does not exist in textstat
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                return 0.0
        method_names = {
            "automated_readability_index": 0,
            "dale_chall_readability_score": 0,
            "difficult_words": 0,
            "monosyllabcount": 0,
            "syllable_count": 0
        }
        for k in method_names:
            method_names[k] = get_lexical_feature(k, test_data)
        return method_names

    def get_sentence_tok_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """Returns the sentence transformer model and tokenizer for the migration task.
        
        Returns:
            Tuple[AutoModel, AutoTokenizer]: The model and tokenizer for the migration task.
        """
        model_path = Config().get_string_model()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer


class StringProcessing():
    def __init__(self):
        pass
    
    def is_garbage_string(self, string, exclusions=None):
        if not all(ord(c) < 128 and c.isprintable() for c in string):
            return True

        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        valid_chars |= set(regex.escape("你好世界Привет мир"))

        for char in string:
            if char not in valid_chars:
                return True

        patterns = [
            r'(;[a-zA-Z0-9])',
            r'([^\w\s])\1+',
            # r'(.)\1+',
            r'([<>\\|])',
        ]

        for pattern in patterns:
            if regex.search(pattern, string):
                return True

        if len(set(string)) == 1:
            return True

        if exclusions:
            for exclusion in exclusions:
                if regex.search(exclusion, string):
                    return True
        return False

    def process_strings(self, strings, exclusions=None):
        valid_strings = []
        valid_strings = [
            s for s in strings 
            if not self.is_garbage_string(s, exclusions)
        ]
        return valid_strings


class DataProcessor():

    def __init__(self):
        self.known_words = set(words.words())

    def is_valid_unix_path(self, file_path):
        try:
            if len(file_path) > 128:  # Adjust the limit as needed
                return False

            # Split the path using both forward slash and backslash
            path_components = [component for component in re.split(r'[\\\/]', file_path) if component]

            # Check if the path components are valid
            for component in path_components:
                if component.startswith('.') or '/' in component or '\\' in component:
                    return False

            # Check if consecutive slashes are present (not allowed)
            if '//' in file_path or '\\\\' in file_path:
                return False

            # Check if at least one segment contains a valid known word according to the dictionary
            for component in path_components:
                words_in_component = re.findall(r'\b\w+\b', component)
                if any(word.lower() in self.known_words and len(word) > 3 for word in words_in_component):
                    return True

            # If none of the above conditions are met, return False
            return False

        except Exception:
            # Any exceptions indicate an invalid path
            return False

    def validate_ip_addresses(self, ip_addresses):
        valid_ips = []

        for ip in ip_addresses:
            try:
                # Attempt to create an IPv4Address object
                IPv4Address(ip)
                valid_ips.append(ip)
            except ValueError:
                pass
            try:
                # Attempt to create an IPv4Address object
                IPv6Address(ip)
                valid_ips.append(ip)
            except ValueError:
                pass
        return valid_ips

    def mean_pooling(self, model_output: BatchEncoding, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings based on attention masks.

        This function calculates the mean of token embeddings, weighted by the attention mask,
        for each input sequence in the batch.

        Parameters:
            model_output: The output of a transformer-based model.
            attention_mask: A binary mask indicating which tokens in the input are valid (1) and which are padding (0).

        Returns:
            Mean-pooled embeddings for each input sequence.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_list_of_texts(self, texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> torch.Tensor:
        """
        Encode a list of text sequences into embeddings using a transformer-based model.

        This function tokenizes the input texts, computes token embeddings, performs mean pooling,
        and normalizes the resulting embeddings.

        Parameters:
            texts: A list of text sequences to be encoded.
            tokenizer: The tokenizer associated with the transformer-based model.
            model: The transformer-based model for encoding text.

        Returns:
            Normalized embeddings for the input text sequences.
        """
        # Tokenize sentences
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = torch_ff.normalize(embeddings, p=2, dim=1)

        return embeddings

    def unique_elem_dict(self, data: pd.DataFrame, column: str) -> dict[str, Any]:
        """
        Create a dictionary with unique elements in a specified column of a DataFrame
        and their corresponding indices in the DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame to process.
        column (str): The name of the column in the DataFrame to extract unique elements from.

        Returns:
        dict: A dictionary where keys are unique elements from the specified column,
        and values are lists of indices in the DataFrame where those elements appear.
        """

        unique_email_dict = {}

        for index, email_list in tqdm(enumerate(data[column].dropna())):
            if email_list:  # Check if the list is not empty
                for email in email_list:
                    if email not in unique_email_dict:
                        unique_email_dict[email] = [index]
                    else:
                        unique_email_dict[email].append(index)
        return unique_email_dict


    def encode_list_of_texts_batched(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        batch_size: int = 256,
        verbose: bool = False
    ) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        embeddings_list = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]

            # Retry smaller batch on memory error
            for attempt in [batch_texts, batch_texts[:batch_size // 2], batch_texts[:batch_size // 4]]:
                try:
                    encoded_input = tokenizer(attempt, padding=True, truncation=True, return_tensors='pt')
                    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

                    with torch.no_grad():
                        model_output = model(**encoded_input, return_dict=True)

                    attention_mask = encoded_input['attention_mask']
                    token_embeddings = model_output.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    normalized = torch_ff.normalize(pooled, p=2, dim=1)

                    embeddings_list.append(normalized.cpu())
                    break  # Success, exit retry loop

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[WARN] OOM at batch {i // batch_size + 1}, retrying smaller batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        if embeddings_list:
            return torch.cat(embeddings_list, dim=0)
        return torch.empty(0)

    def compute_similar_candidates(self, unique_values_sets: list, doc_emb: torch.Tensor, sim_threshold: float=0.9) -> dict:
        """Calculate similar candidates for each element in a list based on their embeddings.

        Parameters:
            unique_values_sets (list): A list of elements for which similar candidates need to be found.
            doc_emb (torch.Tensor): A tensor containing embeddings for all elements.
            sim_threshold: The threshold for considering things to be similar.
        Returns:
            A dictionary where keys are elements from 'unique_values_sets', and values are lists
        of similar candidates based on the similarity score (greater than 0.85).
        """

        similar_candidates = {}

        # Loop through each element in unique_values_sets
        for idx, value in tqdm(enumerate(unique_values_sets)):
            # Get the embedding for the current element
            query_emb = doc_emb[idx:idx+1]  # Ensure query_emb is a 2D tensor
            

            # Calculate the similarity scores with all elements in doc_emb
            # Perform matrix multiplication and squeeze any singleton dimensions
            scores = torch.mm(query_emb, doc_emb.transpose(0, 1)).squeeze()

            # Check if scores is a scalar (not iterable) by checking its dimension
            if scores.dim() == 0:
                # If scores is a scalar, convert it to a list with one element
                scores_list = [scores.item()]
            else:
                # If scores is not a scalar, convert the tensor to a list
                scores_list = scores.cpu().tolist()


            # Find similar candidates with a score greater than 0.85
            similar_indices = [i for i, score in enumerate(scores_list) if score > sim_threshold]

            # Exclude the element itself from similar candidates
            similar_indices = [i for i in similar_indices if i != idx]

            # Store the similar candidates in the dictionary
            similar_candidates[value] = [unique_values_sets[i] for i in similar_indices]

        return similar_candidates


    def calculate_cardinality(self, cell_item, unique_object_dict):
        # Function to calculate the cardinality of an email address
        if cell_item in unique_object_dict:
            return len(unique_object_dict[cell_item])
        return 0

    def find_similar_candidates(self, cell_item, similar_candidates):
        if cell_item in similar_candidates:
            return similar_candidates[cell_item]
        return []

    def normalize_fields_sim(self, row):
        return [
            element for element in row if self.calculate_cardinality(element) > 1
        ] + [
            candidate for element in row for candidate in self.find_similar_candidates(element)
        ]

    def normalize_cell_elements(self, row: list[str], use_similarity: bool = False, 
                                similar_candidates: dict[str, Any] = None, 
                                unique_element_dict: dict[str, Any] = None,
                                cardinality_lower_bound = 1,
                                cardinality_ratio = 0.75
                               ) -> list[str]:
        """
        Normalizes cell elements based on the specified criteria.

        Parameters:
            row: The list of strings representing the row to be normalized.
            use_similarity: A flag indicating whether to use similarity criteria for normalization. Default is False.
            similar_candidates: A dictionary containing potential similar candidates for elements in the row.
            unique_element_dict: A dictionary mapping each element to its uniqueness score.
            cardinality_lower_bound: The minimum number of elements within the dataframe that should be similar.
                - We default to 1. 
            cardinality_ratio: If an element/value has a ratio of occurence greater than this number, we drop it.             

        Returns:
            The list of normalized cell elements.
        """

        if unique_element_dict is None:
            raise ValueError("unique_element_dict cannot be None")

        if not row:
            return []  # Return an empty list if row is empty

        # Precompute cardinality values for all elements in the row
        cardinalities = [self.calculate_cardinality(element, unique_element_dict) for element in row]
        max_cardinality = max(cardinalities)
        if use_similarity:
            if similar_candidates is None:
                raise ValueError("similar_candidates cannot be None when use_similarity is True")

            return [
                element for element, cardinality in zip(row, cardinalities) if 
                cardinality > cardinality_lower_bound and cardinality < cardinality_ratio * max_cardinality
            ] + [
                candidate 
                for element in row 
                for candidate in self.find_similar_candidates(element, similar_candidates)
            ]

        return [
            element for element, cardinality in zip(row, cardinalities) if 
            cardinality > cardinality_lower_bound and cardinality < cardinality_ratio * max_cardinality
        ]

    def string_feature_embed_similarity(
        self,
        data: pd.DataFrame,
        column: str,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        similarity_threshold=0.70,
        **kwargs
    ) -> pd.Series:
        """
        Scalable, memory-efficient embedding similarity normalizer with original signature.

        Accepts **kwargs for:
            - chunk_size (default=5000)
            - max_len (default=1024)
            - cardinality_lower_bound (default=1)
            - cardinality_ratio (default=0.75)
            - cache_dir (optional temp path)
        """
        import tempfile
        import shutil

        chunk_size = kwargs.get("chunk_size", 5000)
        max_len = kwargs.get("max_len", 1024)
        cardinality_lower_bound = kwargs.get("cardinality_lower_bound", 1)
        cardinality_ratio = kwargs.get("cardinality_ratio", 0.75)
        cache_dir = kwargs.get("cache_dir") or tempfile.mkdtemp(prefix="embed_sim_")

        # Validate column
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # Extract just the needed column
        subset = data[[column]].dropna().copy()
        subset.to_parquet(os.path.join(cache_dir, f"{column}.parquet"), index=True)
        full_column = pd.read_parquet(os.path.join(cache_dir, f"{column}.parquet"))

        indices = full_column.index.to_list()
        normalized_chunks = []

        for start in range(0, len(full_column), chunk_size):
            end = start + chunk_size
            chunk = full_column.iloc[start:end].copy()

            if chunk.empty:
                normalized_chunks.append(pd.Series([[]] * len(chunk), index=chunk.index))
                continue

            # Flatten to unique values
            flat = list(set(
                i.strip()[:max_len]
                for sublist in chunk[column]
                if isinstance(sublist, list)
                for i in sublist if isinstance(i, str)
            ))

            if not flat:
                normalized_chunks.append(pd.Series([[]] * len(chunk), index=chunk.index))
                continue

            embeddings = self.encode_list_of_texts_batched(flat, tokenizer, model, batch_size=256)
            sim_matrix = torch.mm(embeddings, embeddings.T).cpu()

            similar_candidates = {
                value: [
                    flat[j]
                    for j in (sim_matrix[i] > similarity_threshold).nonzero().flatten().tolist()
                    if j != i
                ]
                for i, value in enumerate(flat)
            }

            unique_dict = self.unique_elem_dict(chunk, column)
            
            normalized = chunk[column].apply(
                self.normalize_cell_elements,
                use_similarity=True,
                similar_candidates=similar_candidates,
                unique_element_dict=unique_dict,
                cardinality_lower_bound=cardinality_lower_bound,
                cardinality_ratio=cardinality_ratio,
            )
            normalized_chunks.append(normalized)

        shutil.rmtree(cache_dir)
        return pd.concat(normalized_chunks)


    def normalize_column_using_popularity(self, data: pd.DataFrame, column: str, **kwargs):
        """Normalize the elements in a specified column of a DataFrame using a custom normalization function.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the column in the DataFrame that contains the elements to be normalized.

        Returns:
            A Pandas Series containing the normalized elements based on the provided normalization function.
        """
        if column not in data:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        unique_elements_obj = self.unique_elem_dict(data, column)
        return data[column].apply(self.normalize_cell_elements, use_similarity=False,unique_element_dict=unique_elements_obj, **kwargs)


    def normalize_column_using_popularity(self, data: pd.DataFrame, column: str, max_workers: int = 8, **kwargs):
        """
        Normalize elements in the column using cardinality-based filtering with parallel processing.
        """
        if column not in data:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        unique_elements_obj = self.unique_elem_dict(data, column)

        def normalize_row(row):
            return self.normalize_cell_elements(
                row,
                use_similarity=False,
                unique_element_dict=unique_elements_obj,
                **kwargs
            )

        # Use ThreadPoolExecutor to apply normalization in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            normalized_data = list(executor.map(normalize_row, data[column]))

        return pd.Series(normalized_data, index=data.index)


    def one_hot_encode_list_column(self, large_dataframe: pd.DataFrame, column_name: str, to_lower=False) -> pd.DataFrame:
        """
        One-hot encode a list-of-strings column, safely handling missing columns.
        Returns an empty DataFrame (with same index) if the column is missing or all values are empty.
        """
        if column_name not in large_dataframe.columns:
            print(f"[WARN] Column '{column_name}' not found — skipping one-hot encoding.")
            return pd.DataFrame(index=large_dataframe.index)

        # Drop rows with NaNs in the target column
        col = large_dataframe[column_name].dropna()

        if col.empty:
            print(f"[WARN] Column '{column_name}' is empty after dropping NaNs.")
            return pd.DataFrame(index=large_dataframe.index)

        # Optional lowercase transform
        if to_lower:
            col = col.apply(lambda x: list({i.lower() for i in x}) if isinstance(x, list) else x)

        # Now fit MultiLabelBinarizer
        try:
            mlb = MultiLabelBinarizer()
            one_hot_encoded = pd.DataFrame(
                mlb.fit_transform(col),
                index=col.index,
                columns=mlb.classes_
            )
            # Merge encoded data back into a full-index DataFrame
            return one_hot_encoded.reindex(index=large_dataframe.index, fill_value=0)
        except Exception as e:
            print(f"[ERROR] Failed one-hot encoding column '{column_name}': {e}")
            return pd.DataFrame(index=large_dataframe.index)

    def merge_and_relabel(self, df1, df2, merge_column, reference_label_col, other_label_col):
        """
        Merges two DataFrames (df1 and df2) on a specified column (merge_column) and generates a new
        DataFrame with relabeled data using either df1 or df2 labels as a reference.

        Parameters:
        - df1, df2 (pd.DataFrame): DataFrames to merge and relabel.
        - merge_column (str): The column name on which to merge the DataFrames.
        - reference_label_col, other_label_col (str): The column names containing the labels to use
          as reference and the other to compare.

        Returns:
        - new_label_assignments_df (pd.DataFrame): DataFrame containing the merged and relabeled data.
        """

        # Merge the DataFrames
        merged_labels = pd.merge(df1, df2, on=merge_column, suffixes=('_ref', '_other'))

        # Initialize seen hashes and new label assignments
        seen_hashes = set()
        new_label_assignments = []

        # Iterate over unique labels in the reference label column
        for label_ref in merged_labels[reference_label_col + '_ref'].unique():

            # If all hashes for this label have been seen, continue
            if set(merged_labels.loc[merged_labels[reference_label_col + '_ref'] == label_ref, merge_column]).issubset(seen_hashes):
                continue

            # Get the relevant subset of hashes for the current label
            ref_hashes = set(merged_labels.loc[merged_labels[reference_label_col + '_ref'] == label_ref, merge_column]) - seen_hashes

            # Check the corresponding other labels for each hash in this subset
            other_labels = merged_labels.loc[merged_labels[merge_column].isin(ref_hashes), other_label_col + '_other'].unique()

            # For each other label, get all hashes and add them with the reference label to new assignments
            for label_other in other_labels:
                new_hashes = set(merged_labels.loc[merged_labels[other_label_col + '_other'] == label_other, merge_column])

                for hash_ in new_hashes:
                    new_label_assignments.append({merge_column: hash_, 'labels': label_ref})

                # Update seen hashes
                seen_hashes.update(new_hashes)

        # Convert the list of dictionaries to a DataFrame
        new_label_assignments_df = pd.DataFrame(new_label_assignments)

        return new_label_assignments_df