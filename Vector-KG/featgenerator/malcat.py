import warnings
import os
import json
from datetime import datetime
import dateparser as dp
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Models
from sklearn.ensemble import IsolationForest

import seaborn as sns
from matplotlib import pyplot as plt
from .config import Config

warnings.filterwarnings("ignore")

class MalcatFeatures():
    def __init__(self):
        conf = Config()
        self.root_dir = conf.get_root_dir()
        self.yara_filename = conf.get_malcat_filename()

    def get_dataset(self,yara_filename):
        hashes_df = []
        yara_keys = set()


        for subdir, dirs, files in os.walk(self.root_dir):
                for file in files:
                    file_hash = os.path.basename(os.path.normpath(subdir))
                    c = {}
                    if file_hash != file:
                        continue
                    yara_file = os.path.join(subdir, yara_filename)
                    try: 
                        if os.path.isfile(yara_file):
                            with open(yara_file) as f:
                                data = json.load(f)
                                for k in data:
                                    if k == 'hash':
                                        c['hash'] = data['hash']
                                        continue
                                    else:
                                        c[k] = 1 if (data[k] == True) else 0
                                    yara_keys.add(k)


                        hashes_df.append(c)   
                        break  
                    except Exception as e:
                        raise e
        for idx,obj in enumerate(hashes_df):
            for key in yara_keys:
                if obj.get(key) == None:
                    hashes_df[idx].setdefault(key, 0)
        return hashes_df

    def get_features(self):
        hashed_obj = self.get_dataset(self.yara_filename)
        if len(hashed_obj) == 0:
            return pd.DataFrame()
        df_features = pd.DataFrame(hashed_obj)
        return df_features