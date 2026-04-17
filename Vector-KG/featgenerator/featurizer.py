import pandas as pd
import textstat
from sklearn.compose import ColumnTransformer
# Models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer)


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

class Featurizer():
    def get_features_from_df(self, df, ord_features=[], cat_features=[], numeric_features=[], string_feautures=[]):
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy = 'constant', fill_value=0)),
            ('odi', OrdinalEncoder(sparse=False))])

        numeric_transformer = Pipeline(steps=[
            ('scaler', QuantileTransformer(output_distribution="normal")),
            ('imputer', SimpleImputer(strategy = 'constant', fill_value=0))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('onehot', OneHotEncoder(sparse=False))])


        ShortTextProcessor = Pipeline(steps=[
            ("vect", CountVectorizer(analyzer="char")),
            ("tfidf", TfidfTransformer()),
        ])

        text_transformer = Pipeline(
                steps = [("vect", CountVectorizer(analyzer="char"))
                        ]
        )
        transformers = [
            ("ord", ordinal_transformer, ord_features),
            ("num", numeric_transformer, numeric_features),
            ("cat",categorical_transformer, cat_features)
        ]
        for feat in string_feautures:
            transformers.append((feat, text_transformer, string_feautures[feat]))
        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        
        try:
            X = pd.DataFrame(ct.fit_transform(df),columns=ct.get_feature_names_out())
        except AttributeError:
            X = pd.DataFrame.sparse.from_spmatrix(ct.fit_transform(df),columns=ct.get_feature_names_out())

        return X