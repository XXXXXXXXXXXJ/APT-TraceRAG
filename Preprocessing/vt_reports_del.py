import os
import json
import re
import logging
from collections import defaultdict
from collections import Counter
import fnmatch
from tqdm import tqdm
from urllib.parse import urlparse

import pymysql
from sshtunnel import SSHTunnelForwarder

from path_var import WORKSPACE, LOG_PATH, MID_PATH
from credits import (
    SSH_HOST_PORT, SSH_USERNAME, SSH_PASSWORD,
    DB_HOST, DB_USR, DB_PSW, DB_NAME
)

# =============== Logging configuration ===============
LOG_FILENAME = 'vt_report_delall.log'
log_filename = os.path.join(WORKSPACE, LOG_PATH, LOG_FILENAME)
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(
    filename=log_filename,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# =============== Constant paths ===============
VT_SRC_DIR = os.path.join(WORKSPACE, 'VT_reports')
VT_DST_DIR = os.path.join(WORKSPACE, 'VT_reports_delall')
os.makedirs(VT_DST_DIR, exist_ok=True)

SHA1_LIST_PATH = os.path.join(WORKSPACE, MID_PATH, 'sha1_list.txt')
os.makedirs(os.path.dirname(SHA1_LIST_PATH), exist_ok=True)
FIELDS_UNION_PATH = os.path.join(WORKSPACE, MID_PATH, 'vt_fields_union_all.json')
os.makedirs(os.path.dirname(FIELDS_UNION_PATH), exist_ok=True)
ORG_UNION_PATH = os.path.join(WORKSPACE, MID_PATH, 'all_ORGs_union.json')
os.makedirs(os.path.dirname(ORG_UNION_PATH), exist_ok=True)
# Temporary alias file
ALIAS_PHRASES_PATH = os.path.join(WORKSPACE, MID_PATH, 'alias_phrases.txt')
os.makedirs(os.path.dirname(ALIAS_PHRASES_PATH), exist_ok=True)

# =============== Redundant fields definition ===============
# Top-level keys to delete
TOPLEVEL_DROP = {
    "response_code", "verbose_msg", "resource", "scan_id", "permalink",
    "first_seen", "last_seen", "times_submitted", "unique_sources",
    "harmless_votes", "malicious_votes", "community_reputation",
    "positives_delta", "total", "tags", "type", "magic", "size", "ITW_urls",
    "scan_date", "submission_names", "positives", "vhash", "ssdeep",
    "authentihash", "sha1", "sha256", "md5"      # If keeping only sha256, remove sha1/md5
}

# =============== Regex and constants ===============
HEX_PATTERN_LENGTH = 70
HEX_PATTERN = re.compile(rf"[0-9A-Fa-f]{{{HEX_PATTERN_LENGTH},}}")
HEX_REPLACEMENT = "[Long continuous sequence of 0-9A-Fa-f with no semantic information]"
ALLOWED_TERMS = set([
    "group", "team", "security", "operation"  # Example list
])

# additional_info subfield blacklist — delete only listed keys, keep the rest
ADDINFO_DROP_BLACKLIST = {
    "exiftool",          # Large color profile/ICC data, weakly relevant
    "trid",              # Duplicates magic/type
    "sigcheck",          # Empty object or duplicates signature_info
    "packers",           # Remove if not classifying packers
    "magic",             # Sometimes duplicated inside additional_info
    "pdf_info_unused",   # Example: legacy keys
    "positives_delta",  # Duplicates scans.*.positives
    "clam-av-pua",  # PUA labels are coarse and duplicate scans.*.result
    "clamav-pua",   # PUA labels are coarse and duplicate scans.*.result
    "command-unpacker", # VT internal hint; sparse and unstable
    "cwsandbox",    # Deprecated sandbox output, noisy
    "deepguard",    # Single-vendor label, duplicates scans
    "f-prot-unpacker",  # Similar info via peid/packers; noisy
    "gandelf",  # Obsolete ELF checker
    "prevx",    # Vendor discontinued; noisy
    "sophos-pua",   # Duplicates scans; low granularity
    "suspicious-insight",   # Single-vendor Symantec score, redundant
    "threatexpert", # Service discontinued
    "swfknife",     # Obsolete tool; Flash largely deprecated
    "trendmicro-housecall-heuristic",   # Single-vendor heuristic, redundant
    "zemana-behaviour",  # Single-vendor label
    "nsrl",  # Only indicates presence in software DB; no attribution value
    "authentihash",  # Duplicates authentihash field
    "execution_parents", # Hash values not interpretable
    "overlay_parents",  # Hash values not interpretable
    "pe-imphash",  # Hash values not interpretable
    "pe_resource_parents",  # Hash values not interpretable
    "pe-resource-detail",  # Redundant PE resource list
    "carbonblack_children", # Hash values not interpretable
    "carbonblack_parents",   # Hash values not interpretable
    "imgjection"  # Image is not readable
    # Add more keys to remove as needed
}
# Support deleting subfields by path pattern using shell-style wildcards
PATH_DROP_PATTERNS = [
    "additional_info.compressedview.beginning",
    "additional_info.compressedview.vhash",
    "additional_info.compressed_parents",
    "additional_info.detailed_email_parents.*.message",
    "additional_info.detailed_email_parents.*.message_id",
    "additional_info.email_parents"
    # 在此添加更多模式，如 "additional_info\.android-behaviour.*"
]

# Keep only detected engines inside scans; drop undetected/empty results
def slim_scans(scans: dict) -> dict:
    return {
        eng: info["result"]
        for eng, info in scans.items()
        if info.get("detected") and info.get("result")
    }

def prune_additional_info(add_info: dict) -> dict:
    """Delete blacklisted subkeys; keep the rest."""
    return {k: v for k, v in add_info.items() if k not in ADDINFO_DROP_BLACKLIST}

# ========= Recursive collection =========
def collect_field_paths(obj, prefix="", sink=None):
    """
    Recursively traverse any JSON object and add full paths of all leaf fields to sink (set).
    - obj : current object (dict / list / scalar)
    - prefix : parent path, e.g. 'additional_info.androguard'
    - sink : set used for de-dup collection
    """
    if sink is None:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            collect_field_paths(v, new_prefix, sink)
    elif isinstance(obj, list):
        # Do not append indices for lists; treat the list itself as a leaf
        sink.add(prefix)
    else:
        sink.add(prefix)        # Scalar: record full path

def extract_outer_class(class_full):
    """
    Extract the outermost class name: strip package and $ suffix.
    """
    base = class_full.split('.')[-1]
    outer = base.split('$')[0]
    return outer

def build_apk_components(androguard_obj):
    components = []
    counts = Counter()
    comp_map = {
        "activity": "Activities",
        "receiver": "Receivers",
        "service": "Services"
    }
    filters = androguard_obj.get("intent-filters", {})
    summary_dict = {}

    for comp_type, comp_field in comp_map.items():
        comp_list = androguard_obj.get(comp_field, [])
        filters_of_type = filters.get(comp_field, {})
        for class_full in comp_list:
            class_short = extract_outer_class(class_full)
            intent = filters_of_type.get(class_full, {})
            action = intent.get("action", [None])
            category = intent.get("category", None)
            key = (
                comp_type,
                class_short,
                tuple(sorted(set(action))) if action else tuple(),
                tuple(sorted(set(category))) if category else tuple(),
            )
            if key in summary_dict:
                summary_dict[key]['variant_count'] += 1
            else:
                comp_entry = {
                    "type": comp_type,
                    "class": class_short,
                    "variant_count": 1
                }
                if action and action[0]:
                    comp_entry["action"] = action[0] if len(action) == 1 else action
                if category:
                    comp_entry["category"] = sorted(list(set(category)))
                summary_dict[key] = comp_entry
            counts[comp_type] += 1

    for comp_type, filters_dict in filters.items():
        if comp_type == "Activities":
            type_name = "activity"
        elif comp_type == "Receivers":
            type_name = "receiver"
        elif comp_type == "Services":
            type_name = "service"
        else:
            continue
        for class_full, intent in filters_dict.items():
            class_short = extract_outer_class(class_full)
            action = intent.get("action", [None])
            category = intent.get("category", None)
            key = (
                type_name,
                class_short,
                tuple(sorted(set(action))) if action else tuple(),
                tuple(sorted(set(category))) if category else tuple(),
            )
            if key in summary_dict:
                continue
            comp_entry = {
                "type": type_name,
                "class": class_short,
                "variant_count": 1
            }
            if action and action[0]:
                comp_entry["action"] = action[0] if len(action) == 1 else action
            if category:
                comp_entry["category"] = sorted(list(set(category)))
            summary_dict[key] = comp_entry
            counts[type_name] += 1

    return {
        "summary": list(summary_dict.values()),
        "counts": dict(counts)
    }

def rebuild_apk_components(androguard_obj: dict) -> dict:
    """
    Aggregate androguard_obj["apk_components"]["summary"] by type,
    and generate parallel actions/categories lists for each class (None if missing).
    Returns a dict like {"activity": {...}, "receiver": {...}, ...}.
    """
    comp = androguard_obj.get("apk_components", {})
    grouped = defaultdict(list)
    # 按 type 分组
    for item in comp.get("summary", []):
        grouped[item["type"]].append(item)

    result = {}
    for t, items in grouped.items():
        classes = []
        variant_counts = []
        actions = []
        categories = []

        for itm in items:
            # class & variant_count
            classes.append(itm["class"])
            variant_counts.append(itm.get("variant_count"))

            # action -> list or None
            act = itm.get("action")
            if act is None:
                actions.append(None)
            else:
                actions.append(act if isinstance(act, list) else [act])

            # category -> list or None
            cat = itm.get("category")
            categories.append(cat if cat else None)

        result[t] = {
            "classes": classes,
            "variant_counts": variant_counts,
            "actions": actions,
            "categories": categories
        }
    return result

def summarize_android_behaviour(obj):
    if not isinstance(obj, dict):
        return obj

    summary = {}

    # Track processed fields
    processed_keys = set()

    # 1. File-related
    def summarize_paths(paths):
        if not isinstance(paths, list): return None
        dirs = set()
        types = set()
        for p in paths:
            if isinstance(p, str):
                # Directory extraction
                if '/' in p:
                    dirs.add(p[:p.rfind('/')+1])
                # File type (extension)
                m = re.search(r'\.([a-zA-Z0-9]+)$', p)
                if m:
                    types.add(m.group(1).lower())
        return {
            "count": len(paths),
            "main_dirs": list(dirs)[:3],
            "main_types": list(types)[:3],
            "examples": [p for p in paths[:2] if isinstance(p, str)]
        }

    # 2. URI/URL related
    def summarize_uris(uris):
        if not isinstance(uris, list): return None
        hosts = set()
        types = set()
        examples = []
        for u in uris:
            if isinstance(u, str):
                try:
                    pr = urlparse(u)
                    if pr.scheme: types.add(pr.scheme)
                    if pr.hostname: hosts.add(pr.hostname)
                    examples.append(u)
                except Exception:
                    pass
        return {
            "count": len(uris),
            "types": list(types)[:3],
            "main_hosts": list(hosts)[:3],
            "examples": examples[:2]
        }

    # 3. contacted_urls (data may be long encrypted strings)
    def summarize_contacted_urls(urls):
        if not isinstance(urls, list): return None
        hosts = set()
        has_encrypted_data = False
        for u in urls:
            if isinstance(u, dict):
                data = u.get('data')
                if data and re.fullmatch(r'[0-9a-fA-F]{16,}', data):
                    has_encrypted_data = True
                url_val = u.get('url')
                if url_val:
                    try:
                        parsed = urlparse(url_val)
                        if parsed.hostname:
                            hosts.add(parsed.hostname)
                    except Exception:
                        pass
        return {
            "count": len(urls),
            "main_hosts": [h for h in hosts if h][:3],
            "has_encrypted_data": has_encrypted_data
        }

    # File-related
    for k in ['accessed_files', 'opened_files', 'deleted_files']:
        if k in obj:
            summary[f"{k}_summary"] = summarize_paths(obj[k])
            processed_keys.add(k)

    # URI/URL
    if 'accessed_uris' in obj:
        summary['accessed_uris_summary'] = summarize_uris(obj['accessed_uris'])
        processed_keys.add('accessed_uris')
    if 'contacted_urls' in obj:
        summary['contacted_urls_summary'] = summarize_contacted_urls(obj['contacted_urls'])
        processed_keys.add('contacted_urls')

    # Permissions/services/receivers/others
    if 'permissions_checked' in obj:
        summary['permissions_checked'] = [
            p.split('.')[-1].split(':')[0].upper()
            for p in obj['permissions_checked'] if isinstance(p, str)
        ][:8]
        processed_keys.add('permissions_checked')
    if 'started_services' in obj:
        summary['started_services_count'] = len(obj['started_services'])
        processed_keys.add('started_services')
    if 'started_receivers' in obj:
        summary['started_receivers'] = [
            s.split('.')[-1].upper() for s in obj['started_receivers'] if isinstance(s, str)
        ][:4]
        processed_keys.add('started_receivers')
    if 'sandbox-version' in obj:
        summary['sandbox-version'] = obj['sandbox-version']
        processed_keys.add('sandbox-version')

    # Copy unknown fields as-is
    for k, v in obj.items():
        if k not in processed_keys:
            summary[k] = v

    return summary

def compress_permissions_minimal(permissions_dict):
    """
    Keep only the risk level for each permission.
    :param permissions_dict: Raw Androguard Permissions field (dict)
    :return: dict {permission: level}
    """
    if not isinstance(permissions_dict, dict):
        return permissions_dict
    return {perm: arr[0] if isinstance(arr, list) and len(arr) > 0 else "" 
            for perm, arr in permissions_dict.items()}

def summarize_pe_resource_filetype(pe_resource_dict):
    """
    Count occurrences of each filetype in pe-resource-list.
    :param pe_resource_dict: dict, key=hash, value=filetype
    :return: dict, key=filetype, value=count
    """
    if not isinstance(pe_resource_dict, dict):
        return pe_resource_dict
    stats = {}
    for v in pe_resource_dict.values():
        if not isinstance(v, str):
            continue
        stats[v] = stats.get(v, 0) + 1
    return stats

def extract_readable_part(func_name):
    """Extract readable function part from mangled names; return None if unreadable."""
    matches = re.findall(r'\?([a-zA-Z0-9_]+)@', func_name)
    if matches:
        return matches[0]
    readable = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,60}', func_name)
    if readable:
        return readable[-1]
    return None

def is_windows_api(func_name):
    """Check if it is a Windows API (avoid compressing it away)."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{2,60}$', func_name))

def compress_module_funcs(mod):
    """
    Compress module function names: keep readable parts, remove unreadable and duplicates.
    param mod: dict, e.g. {dll: [funcs]}
    return: dict
    """
    new_mod = {}
    for dll, funcs in mod.items():
        filtered = []
        for f in funcs:
            if is_windows_api(f):
                filtered.append(f)
            else:
                readable = extract_readable_part(f)
                if readable:
                    filtered.append(readable)
                # Skip unreadable entries
        # De-duplicate and sort
        filtered_set = sorted(set([x for x in filtered if x]))
        if filtered_set:
            new_mod[dll] = filtered_set
    return new_mod

def compress_func_list(funcs):
    """Simplify the exports function name list."""
    filtered = []
    for f in funcs:
        if is_windows_api(f):
            filtered.append(f)
        else:
            readable = extract_readable_part(f)
            if readable and len(f) < 200:
                filtered.append(readable)
            # Skip unreadable entries
    # De-duplicate + sort
    return sorted(set([x for x in filtered if x]))


def replace_obfuscated_in_document_summary_info(doc_sum):
    """
    Replace suspected encrypted/encoded fields under document_summary_info.
    """
    # Heuristic for meaningless encrypted strings: length > 100 and non-normal characters
    OBFUSCATE_THRESHOLD = 100
    def is_obfuscated(val):
        if not isinstance(val, str):
            return False
        # 长度判定
        if len(val) < OBFUSCATE_THRESHOLD:
            return False
        return True
        

    new_doc_sum = {}
    for k, v in doc_sum.items():
        if isinstance(v, str) and is_obfuscated(v):
            new_doc_sum[k] = "Obfuscated/encoded values"
        else:
            new_doc_sum[k] = v
    return new_doc_sum

def summarize_macros(macros):
    patterns, properties, stream_paths, subfilenames, lengths = set(), set(), set(), set(), []
    malicious_count = 0
    for m in macros:
        patterns.update(m.get("patterns", []))
        properties.update(m.get("properties", []))
        stream_paths.add(m.get("stream_path", ""))
        subfilenames.add(m.get("subfilename", ""))
        lengths.append(m.get("length", 0))
        if m.get("patterns") or m.get("properties"):
            malicious_count += 1
    return {
        "macro_count": len(macros),
        "malicious_macro_count": malicious_count,
        "patterns": list(patterns),
        "properties": list(properties),
        "stream_path_samples": list(stream_paths)[:3],
        "subfilename_samples": list(subfilenames)[:3],
        "macro_length_max": max(lengths) if lengths else 0,
        "macro_length_avg": int(sum(lengths) / len(lengths)) if lengths else 0
    }

# Obvious ad/analytics/push-related strings
COMMON_ADV = [
    'ads', 'admob', 'push', 'recommend', 'analytics', 'stat', 'log', 'logger', 'umeng', 'crashlytics'
]

def is_all_digits_or_symbols(s):
    return all(c.isdigit() or not c.isalnum() for c in s)

def is_ascii_letters(s):
    # 至少含有英文字母且无明显乱码
    return bool(re.search(r'[A-Za-z]', s)) and all((32 <= ord(c) < 127) for c in s)

def is_ad_or_analytics(s):
    # 仅限典型广告/统计/push等相关URL或路径
    s_lower = s.lower()
    return any(x in s_lower for x in COMMON_ADV)

# ========= StringsInformation filtering =========
def filter_strings_information(str_list):
    result = []
    for s in str_list:
        # 1. Drop strings of only digits/symbols
        if is_all_digits_or_symbols(s):
            continue
        # 2. Drop non-English/garbled/non-ASCII
        if not is_ascii_letters(s):
            continue
        # 3. Drop too short or too long
        if not (4 <= len(s) <= 100):
            continue
        # 4. Drop ads/analytics/push, etc.
        if is_ad_or_analytics(s):
            continue
        result.append(s)
    return result

# ========= rtfinspect.document_properties de-dup =========
def dedup_objects_with_count(obj_list):
    """
    De-duplicate objects list and add occurrence_num field.
    """
    if not isinstance(obj_list, list):
        return obj_list
    # Use json.dumps for hashable dicts and de-dup (ignore key order)
    obj_json_list = [json.dumps(o, sort_keys=True) for o in obj_list if isinstance(o, dict)]
    count = Counter(obj_json_list)
    unique_objs = []
    for obj_str, num in count.items():
        obj = json.loads(obj_str)
        obj['occurrence_num'] = num
        unique_objs.append(obj)
    return unique_objs


# ========= Recursive sanitization =========
def sanitize_report(obj, alias_phrases: set, sample_id: str, path: str = ""):
    """
    Recursively sanitize the report:
    1) Delete subfields matching PATH_DROP_PATTERNS
    2) Replace long hex strings with [ENCRYPTED_DATA]
    3) Delete any field containing org aliases or real org names
    """
    # 1. If path matches a delete pattern, drop the whole field
    for patt in PATH_DROP_PATTERNS:
        if fnmatch.fnmatch(path, patt):
            # logging.info(f"[{sample_id}] Removed path '{path}' matching pattern '{patt}'")
            return None

    # Handle additional_info.compressedview.children => compressedview.filenames
    if path == "additional_info.compressedview.children" and isinstance(obj, list):
        filenames = [child.get("filename") for child in obj if isinstance(child, dict) and "filename" in child]
        # Return to upper layer as {"filenames": [...]}
        # logging.info(f"[{sample_id}] Rewrite 'additional_info.compressedview.children' to 'filenames'")
        return {"filenames": filenames}
    
    # Handle additional_info.officecheck.document_summary_info
    if path == "additional_info.officecheck.document_summary_info" and isinstance(obj, dict):
        return replace_obfuscated_in_document_summary_info(obj)
    
    # Check and handle additional_info.openxmlchecker.ole.macros
    if path == "additional_info.openxmlchecker.ole.macros" and isinstance(obj, list):
        return summarize_macros(obj)

    # Handle additional_info.imports
    if path == "additional_info.imports" and isinstance(obj, dict):
        return compress_module_funcs(obj)
    
    # Handle additional_info.exports
    if path == "additional_info.exports" and isinstance(obj, list):
        return compress_func_list(obj)
    
    # Handle additional_info.android-behaviour
    if path == "additional_info.android-behaviour" and isinstance(obj, dict):
        return summarize_android_behaviour(obj)
    
    # New: summarize pe-resource-list
    if path == "additional_info.pe-resource-list" and isinstance(obj, dict):
        return summarize_pe_resource_filetype(obj)

    # Handle rtfinspect.document_properties.objects
    if path == "additional_info.rtfinspect.document_properties.objects" and isinstance(obj, list):
        # Recursively sanitize all items first
        cleaned_list = []
        for idx, item in enumerate(obj):
            cleaned = sanitize_report(item, alias_phrases, sample_id, f"{path}[{idx}]")
            if cleaned is not None:
                cleaned_list.append(cleaned)
        return dedup_objects_with_count(cleaned_list)

    # New: handle additional_info.androguard
    if path == "additional_info.androguard" and isinstance(obj, dict):
        new_dict = dict(obj)
        try:
            tmp = build_apk_components(obj)
            new_dict["apk_components"] = rebuild_apk_components({"apk_components": tmp})
        except Exception as e:
            logging.warning(f"[{sample_id}] build_apk_components failed: {e}")
        # Handle Permissions here
        if "Permissions" in new_dict:
            new_dict["Permissions"] = compress_permissions_minimal(new_dict["Permissions"])
        if "StringsInformation" in new_dict:
            new_dict["StringsInformation"] = filter_strings_information(new_dict["StringsInformation"])
        # Delete unnecessary raw fields
        for k in ["Activities", "Receivers", "Services", "intent-filters"]:
            if k in new_dict:
                del new_dict[k]
        return new_dict
    
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            sanitized_v = sanitize_report(v, alias_phrases, sample_id, new_path)
            if sanitized_v is None:
                # logging.info(f"[{sample_id}] Removed field at '{new_path}'")
                continue
            new_dict[k] = sanitized_v
        return new_dict
    elif isinstance(obj, list):
        new_list = []
        for idx, item in enumerate(obj):
            new_path = f"{path}.*"
            sanitized_item = sanitize_report(item, alias_phrases, sample_id, new_path)
            if sanitized_item is None:
                # logging.info(f"[{sample_id}] Removed list element at '{new_path}'")
                continue
            new_list.append(sanitized_item)
        return new_list
    elif isinstance(obj, str):
        # Replace hex segments
        def replace_hex(m):
            logging.info(f"[{sample_id}] Replaced encrypted hex segment at '{path}'")
            return HEX_REPLACEMENT
        new_str = HEX_PATTERN.sub(replace_hex, obj)
        # If whole string is replaced
        if new_str == '':
            new_str = HEX_REPLACEMENT
        lower_val = obj.lower()
        # Delete fields containing org aliases or real org names
        for phrase in alias_phrases:
            if phrase in lower_val and 'scans' in path:
                logging.info(f"[{sample_id}] Deleted value at '{path}' containing alias/group '{phrase}'")
                return None
        return new_str
    else:
        return obj
    
# =============== Database connection ===============
def connect_to_database(
    ssh_host_port, ssh_username, ssh_password,
    remote_bind_address, db_user, db_password, db_name
):
    """Connect to the database."""
    ssh_host, ssh_port = ssh_host_port.split(':')
    server = SSHTunnelForwarder(
        (ssh_host, int(ssh_port)),
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_bind_address=remote_bind_address
    )
    server.start()
    db = pymysql.connect(
        host='127.0.0.1',
        port=server.local_bind_port,
        user=db_user,
        password=db_password,
        database=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return db, server

# =============== Main flow ===============
def main():
    logging.info("Start processing VT reports: read DB, map org aliases, slim fields, and sanitize...")
    all_fields_union = set()
    # itw_nonnull_count = 0

    try:
        
        # 1. Get sha1 list and OrgLabel from the database
        db, server = connect_to_database(
            SSH_HOST_PORT, SSH_USERNAME, SSH_PASSWORD,
            (DB_HOST, 3306), DB_USR, DB_PSW, DB_NAME
        )
        cursor = db.cursor()
        cursor.execute("SELECT sha1, OrgLabel FROM STAT_SampleInfo;")
        records = cursor.fetchall()
        sha1_orglabel = {row['sha1'].lower(): (row.get('OrgLabel') or '') for row in records}
        sha1_list = list(sha1_orglabel.keys())
        # Extract unique OrgLabel names
        orglabels = {lbl.strip() for lbl in sha1_orglabel.values() if lbl.strip()}
        orglabels_lower = {lbl.lower() for lbl in orglabels}


        # 2. Read alias mappings for OrgLabel entries from JSONL
        alias_file = os.path.join(WORKSPACE, 'aptnamemapping', 'rewrite_noid_apts_info_20250314_patched3.jsonl')
        simple_alias_list = []
        alias_mapping = {}
        if os.path.exists(alias_file):
            with open(alias_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        name = entry.get('name', '').strip().lower()
                        # Only handle OrgLabel names that appear in DB
                        if name in orglabels_lower:
                            raw_alias = entry.get('aliases')
                            # If aliases is a list
                            if isinstance(raw_alias, list):
                                aliases = [a.strip().lower() for a in raw_alias if isinstance(a, str)]
                            # If aliases is a string
                            elif isinstance(raw_alias, str):
                                aliases = [raw_alias.strip().lower()]
                            # null or other cases
                            else:
                                aliases = []
                            simple_alias_list.append({'name': name, 'aliases': aliases})
                            alias_mapping[name] = aliases
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse alias line: {line}")
        
        # Write alias_phrases to a temporary file
        # with open(ALIAS_PHRASES_PATH, 'w', encoding='utf-8') as f:
        #     for phrase in sorted(alias_phrases):
        #         f.write(f"{phrase}\n")
        # logging.info(f"Alias phrases saved to {ALIAS_PHRASES_PATH}")

        # Write sha1/name/aliases to JSON
        output = []
        for sha1, org in sha1_orglabel.items():
            key = org.strip().lower()
            name = org.strip()
            aliases = alias_mapping.get(key, [])
            output.append({
                'sha1': sha1,
                'name': name,
                'aliases': aliases
            })
        with open(ORG_UNION_PATH, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logging.info("Wrote all_orgs_union.json (contains sha1/name/aliases)")

        # Process each report
        for idx, sha1 in enumerate(tqdm(sha1_list, desc="Processing VT reports"), start=1):
            src_path = os.path.join(VT_SRC_DIR, f"{sha1}.json")
            dst_path = os.path.join(VT_DST_DIR, f"{sha1}.json")

            if not os.path.exists(src_path):
                logging.warning(f"[{idx}/{len(sha1_list)}] Report file not found: {src_path}")
                continue

            try:
                with open(src_path, "r", encoding="utf-8") as fp:
                    report = json.load(fp)
            except Exception as e:
                logging.error(f"Failed to read {src_path}: {e}")
                continue

            # Collect field union (already have full field list; not needed)
            # collect_field_paths(report, sink=all_fields_union)
            # itw_urls = report.get("ITW_urls")
            # if isinstance(itw_urls, list) and len(itw_urls) > 0:
            #     itw_nonnull_count += 1


            # Slim top-level fields
            slim_report = {k: v for k, v in report.items() if k not in TOPLEVEL_DROP}
            # Special handling for scans
            if 'scans' in slim_report:
                slim_report['scans'] = slim_scans(slim_report['scans'])

            # Special handling for additional_info
            if 'additional_info' in slim_report:
                slim_report['additional_info'] = prune_additional_info(slim_report['additional_info'])

            # ====== Build alias_phrases dynamically from current OrgLabel and aliases ======
            orglabel = sha1_orglabel.get(sha1, '').strip()
            orglabel_lower = orglabel.lower()
            aliases = alias_mapping.get(orglabel_lower, [])
            alias_phrases = set([orglabel_lower] + aliases)
            alias_phrases = {a for a in alias_phrases if len(a) >= 4 and not a.isdigit()}
            alias_phrases -= ALLOWED_TERMS

            # Sanitize: replace encrypted hex and remove org info
            sanitized = sanitize_report(slim_report, alias_phrases, sha1)

            # Write result
            try:
                json_str = json.dumps(sanitized, ensure_ascii=False, indent=2)
                with open(dst_path, "wb") as f:
                    f.write(json_str.encode("utf-8", "surrogatepass"))
            except Exception as dump_err:
                logging.error(f"Write failed {dst_path}: {dump_err}")
                continue

            logging.info(f"[{idx}/{len(sha1_list)}] Completed: {sha1}")

        # Write field union to file (not needed since full field list is available)
        # with open(FIELDS_UNION_PATH, "w", encoding="utf-8") as fp:
        #     json.dump(sorted(all_fields_union), fp, indent=2, ensure_ascii=False)
        # logging.info(f"Field union saved to {FIELDS_UNION_PATH}")
        # logging.info(f"Top-level ITW_urls not None count: {itw_nonnull_count}")
        # print(f"Top-level ITW_urls not None count: {itw_nonnull_count} ")


    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
    finally:
        # Clean up resources
        if 'cursor' in locals():
            cursor.close()
        if 'db' in locals():
            db.close()
        if 'server' in locals():
            server.stop()
        logging.info("VT report slimming workflow finished.")

if __name__ == "__main__":
    main()
