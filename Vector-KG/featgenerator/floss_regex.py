import json
import os

import pandas as pd

from .config import Config


class FlossRegexFeatures():
        def __init__(self):
            conf = Config()
            self.root_dir = conf.get_root_dir()
            self.regex_filename = conf.get_regex_filename()
            self.sensitive_unix_directories = ["/proc/cmdline", "/src/syscall", "/etc/system.d", " /etc/rc.",
                             "/etc/init.", "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/gshadow",
                             "/etc/hosts", "/etc/resolv.conf", "/etc/profile",
                             "~/.bash_profile", "~/.bash_login", "~/.profile", "~/.bashrc",
                             "/etc/bash.bashrc", "/etc/profile.d/"]
            self.windows_file = ["cmd.exe","system.dll", "user32.dll", "winsrv.dll", "ntuser.dat", "powershell.exe", 
                                "shell32.dll", "wscript.exe", "lsass.exe", "ftp.exe", "mshta.exe", "bitsadmin.exe", 
                                "regsvr32.exe", "rundll32.exe", "wmic.exe", "netsh.exe",
                                "at.exe", "schtasks.exe", "ipconfig.exe", "net.exe", "tasklist.exe", "reg.exe",
                                "ping.exe", "msiexec.exe", "psexec.exe", "winword.exe", "certutil.exe"]

        def get_dataset(self, root_dir, regex_filename):
            hashes_df = []
            for subdir, dirs, files in os.walk(root_dir):
                if len(files) == 0:
                    continue
                for file in files:
                    file_hash = os.path.basename(os.path.normpath(subdir))
                    if file_hash != file:
                        continue
                    regex_file = os.path.join(subdir, regex_filename)
                    hashed_obj = {
                       "hash": file_hash,
                       "UrlCount":0,
                       "IPCount":0,
                       "SensitiveFilePath":False,
                       "SensitiveFile":False,
                       "IPencodedUrl":False
                    }
                    try:
                        if os.path.isfile(regex_file):
                            with open(regex_file) as f:
                                data = json.load(f)

                            if 'URL'in data[file_hash]:
                                res_1 = list(filter(None, data[file_hash]['URL']))
                                hashed_obj['UrlCount'] = len(res_1)
                                try:
                                    if self.domain_info(res_1):
                                        hashed_obj['IPencodedUrl'] = True
                                except Exception as e:
                                    pass
                                                                                        
                            if 'ipaddress' in data[file_hash]:
                                res_2 = list(filter(None, data[file_hash]['ipaddress']))
                                hashed_obj['IPCount'] = len(res_2)
                                
                            if 'FilePath_1' in data[file_hash]:
                                windows_path = list(filter(None, data[file_hash]['FilePath_1']))
                                #print(self.windows_file)
                                for path in windows_path:
                                    for file in self.windows_file:
                                        if file in path.lower():
                                            hashed_obj['SensitiveFile'] = True
                                
                            if 'FilePath_2' in data[file_hash]:
                                unix_path = list(filter(None, data[file_hash]['FilePath_2']))
                                for path in unix_path:
                                    for dir_i in self.sensitive_unix_directories:
                                        if dir_i in path.lower():
                                            hashed_obj['SensitiveFilePath'] = True   


                        hashes_df.append(hashed_obj)
                        
                        break
                    except Exception as e:
                        break
                        raise e

            return hashes_df


        def get_features(self):
            hashed_obj = self.get_dataset(self.root_dir, self.regex_filename)
            if len(hashed_obj) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(hashed_obj)

            df["SensitiveFilePath"] = df["SensitiveFilePath"].astype(int)
            df["SensitiveFile"] = df["SensitiveFile"].astype(int)
            df["IPencodedUrl"] = df["IPencodedUrl"].astype(int)


            df_features = df
            return df_features
        
