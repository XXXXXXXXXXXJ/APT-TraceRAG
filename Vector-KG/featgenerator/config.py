import configparser
import os
feature_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_CONFIG_FILE = os.path.join(feature_directory, "config\projectconfig.ini")

class Config():
    def __init__(self, config_file = DEFAULT_CONFIG_FILE) -> None:
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_root_dir(self) -> str:
        root_dir = self.config["DEFAULT"]["root_dir"]
        return root_dir
    
    def get_floss_file(self) -> str:
        floss_file = self.config["floss"]["floss_filename"]
        return floss_file
    
    def get_string_model(self) -> str:
        string_model = self.config["models"]["string_model"]
        return string_model
    
    def get_exif_file_name(self) -> str:
        exif_filename = self.config["exif"]["exif_filename"]
        return exif_filename
    
    def get_func_file_name(self) -> str:
        func_filename = self.config["func_feat"]["func_filename"]
        return func_filename
    
    def get_exif_binary(self) -> str:
        exif_binary = self.config["exif"]["exif_binary"]
        return exif_binary
    
    def get_adversary_mapping(self) -> str:
        adversary_mapping = self.config["groundtruth"]["adversary_mapping"]
        return adversary_mapping
    
    def get_lief_filename(self) -> str:
        lief_filename = self.config["lief"]["lief_filename"]
        return lief_filename
    
    def get_malcat_filename(self) -> str:
        malcat_filename = self.config["yara"]["malcat_filename"]
        return malcat_filename
    
    def get_mitre_campaigns(self) -> str:
        mitre_campaigns = self.config["groundtruth"]["mitre_campaigns"]
        return mitre_campaigns
    
    def get_regex_filename(self) -> str:
        regex_filename = self.config["regex"]["regex_filename"]
        return regex_filename
    
    def get_oletool_filename(self) -> str:
        oletool_filename = self.config["oletool"]["oletool_filename"]
        return oletool_filename
    
    def get_censys_filename(self) -> str:
        censys_filename = self.config["censys"]["censys_filename"]
        return censys_filename