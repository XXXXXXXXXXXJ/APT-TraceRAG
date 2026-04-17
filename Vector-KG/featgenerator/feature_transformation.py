import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from tensorflow.keras.models import Model


from .doc_features import DocFeatures
from .exif_feat import ExifFeatures
from .floss_general_feat import FlossFeatures
from .floss_regex import FlossRegexFeatures
from .lief_features import LiefFeatures
from .malcat import MalcatFeatures
from .util import DataProcessor, Util


def get_combined_features(
    floss_features: bool = False,
    exif_features: bool = False,
    malcat_features: bool = False,
    lief_features: bool = False,
    flossregex_features: bool = False,
    exported_functions_features: bool = False,
    configuration_version: bool = False,
    document_features: bool = False,
    exported_functions_similarity: float = 0.8,
):
    """
    Combines features from different sources based on the specified boolean parameters.

    Args:
    - floss_features: Whether to include FlossFeatures.
    - exif_features: Whether to include ExifFeatures.
    - malcat_features: Whether to include MalcatFeatures.
    - lief_features: Whether to include LiefFeatures.
    - flossregex_features: Whether to include FlossRegexFeatures.
    - exported_functions_features: Whether to include ExportedFunctionsFeatures.
    - configuration_version: Whether to include ConfigurationVersion from LiefFeatures.
    - document_features: Whether to include DocumentFeatures.
    - exported_functions_similarity: The similarity threshold between exported functions.

    Returns:
    - A DataFrame containing the combined features.
    """
    # Initialize the main DataFrame with 'hash' column
    combined_df = pd.DataFrame()
    all_results_dict: dict[str, pd.DataFrame] = {
        "floss_features": pd.DataFrame(),
        "exif_features": pd.DataFrame(),
        "malcat_features": pd.DataFrame(),
        "lief_features": pd.DataFrame(),
        "flossregex_features": pd.DataFrame(),
        "exported_functions_features": pd.DataFrame(),
        "configuration_version": pd.DataFrame(),
        "document_features": pd.DataFrame(),
    }
    if lief_features:
        lief_feat = LiefFeatures()
        lcf = lief_feat.get_features()
        lcf["hash"] = lcf["hash"].astype(str)
        combined_df = lcf[["hash"]].copy()
        all_results_dict["lief_features"] = lcf

    if floss_features:
        floss_feat = FlossFeatures()
        fdf = floss_feat.get_features()
        fdf["hash"] = fdf["hash"].astype(str)
        combined_df = (
            combined_df.merge(fdf, how="inner", on="hash")
            if not combined_df.empty
            else fdf
        )
        all_results_dict["floss_features"] = fdf

    if exif_features:
        exif_feat = ExifFeatures()
        exf = exif_feat.get_normalized_features()
        exf["hash"] = exf["hash"].astype(str)
        combined_df = (
            combined_df.merge(exf, how="inner", on="hash")
            if not combined_df.empty
            else exf
        )
        all_results_dict["exif_features"] = exf

    if malcat_features:
        malcat_feat = MalcatFeatures()
        mcf = malcat_feat.get_features()
        mcf["hash"] = mcf["hash"].astype(str)
        combined_df = (
            combined_df.merge(mcf, how="inner", on="hash")
            if not combined_df.empty
            else mcf
        )
        all_results_dict["malcat_features"] = mcf

    if flossregex_features:
        flossregex_feat = FlossRegexFeatures()
        reg = flossregex_feat.get_features()
        # Assuming 'reg' DataFrame contains a 'hash' column
        combined_df = (
            combined_df.merge(reg, how="inner", on="hash")
            if not combined_df.empty
            else reg
        )
        all_results_dict["flossregex_features"] = reg

    if exported_functions_features:
        # Assuming 'exported_functions_reduced' and 'data_processor' are defined elsewhere and relevant
        model, tokenizer = Util().get_sentence_tok_model()
        data_processor = DataProcessor()
        lcf["ExportedFunctions"] = lcf["ExportedFunctions"].apply(
            lambda x: [os.path.basename(i.split(".")[-1]) for i in x]
        )
        exported_functions_reduced = data_processor.string_feature_embed_similarity(
            lcf, "ExportedFunctions", tokenizer, model, exported_functions_similarity
        )
        exported_features = data_processor.one_hot_encode_list_column(
            pd.DataFrame(exported_functions_reduced), "ExportedFunctions", True
        )
        exdf = exported_features.copy()
        exdf["hash"] = lcf["hash"].copy().astype(str)
        combined_df = (
            combined_df.merge(exdf, how="inner", on="hash")
            if not combined_df.empty
            else exdf
        )
        all_results_dict["exported_functions_features"] = exdf

    if lief_features and configuration_version:
        # Additional processing related to LiefFeatures
        version_encoder = LabelEncoder()
        conf_version = pd.DataFrame(
            version_encoder.fit_transform(lcf.ConfigurationVersion),
            columns=["ConfigurationVersion"],
        )
        conf_version["hash"] = lcf["hash"].astype(str).copy()
        combined_df = combined_df.merge(conf_version, how="inner", on="hash")
        all_results_dict["configuration_version"] = conf_version

    if document_features:
        doc_feat = DocFeatures()
        dcf = doc_feat.get_features()
        dcf["hash"] = dcf["hash"].astype(str)
        combined_df = combined_df.merge(dcf, how="inner", on="hash")
    return combined_df, all_results_dict


def prepare_and_encode_features(
    joined_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    adversary_dataset: pd.DataFrame,
    target_column: str = "Campaign_Tag",
    include_embedding: bool = True,
    num_epochs=10,
) -> Tuple[Model, np.ndarray, pd.DataFrame]:
    """
    Merges datasets, optionally includes embedding features, prepares features and target, trains an autoencoder,
    and encodes the features. Optionally merges embedding features after encoding if they were not included initially.

    Args:
    - joined_df: Main DataFrame to be merged with embedding_df if include_embedding is True.
    - embedding_df: DataFrame containing embedding features.
    - adversary_dataset: Dataset containing adversary information including the target column.
    - target_column: The name of the column to be used as the target variable. Defaults to 'Campaign_Tag'.
    - include_embedding: Determines whether embedding_df should be merged into joined_df before encoding.

    Returns:
    - A tuple containing the autoencoder model, encoded features (X_encoded), and the combined features DataFrame.
    """
    if include_embedding:
        joined_inner = joined_df.merge(embedding_df, how="inner", on=["hash"])
    else:
        joined_inner = joined_df.copy()

    # Merge with adversary_dataset
    all_features = joined_inner.merge(
        adversary_dataset[["hash", "Report link", "Normalized_Tag", "Campaign_Tag"]],
        left_on="hash",
        right_on="hash",
    )

    # Prepare features and target
    X = joined_inner.drop(columns=["hash"])
    X.columns = X.columns.astype(str)
    y = all_features[target_column].fillna("-1")

    # Train autoencoder and encode features
    autoencoder, X_encoded = Util().train_autoencoder(X, num_epochs=num_epochs)

    # Prepare combined features DataFrame
    if include_embedding:
        combined_features = pd.DataFrame(X_encoded)
    else:
        # If embedding_df was not included initially, merge it with the encoded features after encoding
        combined_features = pd.concat(
            [embedding_df.reset_index(drop=True), pd.DataFrame(X_encoded)], axis=1
        ).drop(columns=["hash"])
    combined_features.columns = combined_features.columns.astype(str)
    return autoencoder, X_encoded, combined_features, all_features
