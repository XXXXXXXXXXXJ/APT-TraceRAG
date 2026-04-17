#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean Benchmark_1000/{sha1}.json  MalFamily_ClarAVy & MalFamily_TagClass 。

write：
python clean_mal_tags.py --bench ./Executable/Benchmark_1000_all --orgs ./all_ORGs_union.json --out ./Executable/Benchmark_1000_all_cleaned

python clean_mal_tags.py --bench ./Executable/R_all --orgs ./all_ORGs_union.json --out ./Executable/R_all_cleaned

"""

import os
import re
import json
import argparse
import logging
from typing import Dict, Any, List, Optional

# Optional import of path_var (aligns with your project)
try:
    from path_var import WORKSPACE, LOG_PATH
except Exception:
    WORKSPACE = "."
    LOG_PATH = "."

# ===== Logging configuration (per your request) =====
LOG_FILENAME = 'clean_mal_tags.log'
log_dir = os.path.join(WORKSPACE, LOG_PATH)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, LOG_FILENAME)
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
# =================================

def load_org_index_file(orgs_json_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Read a single JSON file (list) and return sha1 -> {"name": str, "aliases": List[str]}.
    """
    with open(orgs_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, list):
        raise ValueError("all_ORGs_union must be a list.")
    for item in data:
        if not isinstance(item, dict):
            continue
        sha1 = (item.get("sha1") or "").lower()
        if not sha1:
            continue
        name = item.get("name") or ""
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        index[sha1] = {"name": name, "aliases": [a for a in aliases if isinstance(a, str)]}
    return index

def _casefold(s: str) -> str:
    return s.casefold()

def clean_malfamily_claravy(val: Any, sha1: str) -> Any:
    """
    Remove GRP:xxxxxx|XXX segments from MalFamily_ClarAVy and log removals.
    """
    if not isinstance(val, str):
        return val

    parts = [p.strip() for p in val.split(",")]
    removed_segments = []
    kept = []
    for p in parts:
        if re.search(r"\bGRP\s*:", p, flags=re.IGNORECASE):
            removed_segments.append(p)
        else:
            kept.append(p)

    if removed_segments:
        logger.info(f"[{sha1}] ClarAVy: removed GRP segments -> {removed_segments}")

    cleaned = ",".join([p for p in kept if p != ""])
    cleaned = re.sub(r"\s*,\s*", ",", cleaned).strip(" ,")
    return cleaned

def _delete_keys_case_insensitive(d: Dict[str, Any], keys_to_remove: List[str]) -> (Dict[str, Any], List[str]):
    """
    Delete entries from dict d whose keys match keys_to_remove (case-insensitive).
    Returns (new dict, list of removed original keys).
    """
    remove_set = set(_casefold(k) for k in keys_to_remove if isinstance(k, str) and k)
    newd = {}
    removed_keys = []
    for k, v in d.items():
        if _casefold(str(k)) in remove_set:
            removed_keys.append(k)
            continue
        newd[k] = v
    return newd, removed_keys

def _find_words_in_text(text: str, words: List[str]) -> List[str]:
    """
    Find matched words in plain text (case-insensitive, whole-word when possible).
    Returns matched words from the input list.
    """
    hits = []
    for w in words:
        if not isinstance(w, str) or not w:
            continue
        if re.fullmatch(r"\w+", w, flags=re.UNICODE):
            pattern = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        if pattern.search(text):
            hits.append(w)
    return hits

def _strip_words_case_insensitive(text: str, words: List[str]) -> str:
    """
    Remove words from text (case-insensitive, whole-word when possible).
    """
    out = text
    for w in words:
        if not isinstance(w, str) or not w:
            continue
        if re.fullmatch(r"\w+", w, flags=re.UNICODE):
            pattern = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        out = pattern.sub("", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s*,\s*,+", ",", out).strip(" ,")
    return out

def clean_malfamily_tagclass(val: Any, removable_words: List[str], sha1: str) -> Any:
    """
    Clean MalFamily_TagClass and log removals:
    - Dict or "JSON-in-string": delete matching keys and log removed keys.
    - Non-JSON string: log matched words, then remove them.
    """
    if isinstance(val, dict):
        cleaned_dict, removed_keys = _delete_keys_case_insensitive(val, removable_words)
        if removed_keys:
            logger.info(f"[{sha1}] TagClass(dict): removed keys -> {removed_keys}")
        return cleaned_dict

    if isinstance(val, str):
        s = val.strip()
        # If it looks like a JSON object
        if s.startswith("{") and s.endswith("}"):
            parsed: Optional[Dict[str, Any]] = None
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                cleaned_dict, removed_keys = _delete_keys_case_insensitive(parsed, removable_words)
                if removed_keys:
                    logger.info(f"[{sha1}] TagClass(jsonstr): removed keys -> {removed_keys}")
                return json.dumps(cleaned_dict, ensure_ascii=False, separators=(", ", ": "))
        # Fall back to plain-string handling: log matches, then remove
        hits = _find_words_in_text(s, removable_words)
        if hits:
            logger.info(f"[{sha1}] TagClass(str): matched words -> {hits}")
        return _strip_words_case_insensitive(s, removable_words)

    # Other types are ignored
    return val

def process_one_file(fp_in: str, fp_out: str, org_info: Dict[str, Any], sha1: str) -> bool:
    try:
        with open(fp_in, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Read failed: {fp_in} - {e}")
        return False

    # Delete attack_ids/attck_ids
    keys_to_delete = {"attack_ids", "attck_ids", "pstree", "signature", "network"}
    if isinstance(data, dict):
        removed = []
        for k in list(data.keys()):
            if k.lower() in keys_to_delete:
                removed.append(k)
                data.pop(k, None)
        if removed:
            logger.info(f"[{sha1}] Removed top-level keys -> {removed}")

    # ClarAVy
    if "MalFamily_ClarAVy" in data:
        data["MalFamily_ClarAVy"] = clean_malfamily_claravy(data["MalFamily_ClarAVy"], sha1)

    # TagClass
    name = (org_info.get("name") or "") if org_info else ""
    aliases = (org_info.get("aliases") or []) if org_info else []
    removable_words: List[str] = []
    if name:
        removable_words.append(name)
    if isinstance(aliases, list):
        removable_words.extend([a for a in aliases if isinstance(a, str)])

    if "MalFamily_TagClass" in data and removable_words:
        data["MalFamily_TagClass"] = clean_malfamily_tagclass(data["MalFamily_TagClass"], removable_words, sha1)

    # Write output
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    try:
        with open(fp_out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[WARN] Write failed: {fp_out} - {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Clean MalFamily_ClarAVy / MalFamily_TagClass for Benchmark_1000 (single JSON org index)")
    ap.add_argument("--bench", type=str, default="./Benchmark_1000_cleaned_network", help="Directory containing {sha1}.json")
    ap.add_argument("--orgs", type=str, default="./all_ORGs_union.json", help="all_ORGs_union single JSON file (list)")
    ap.add_argument("--out", type=str, default="./Benchmark_1000_cleaned_network_deltpp", help="Output directory (default: new dir)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite in place (mutually exclusive with --out, takes precedence)")
    args = ap.parse_args()

    bench_dir = args.bench
    if not os.path.isdir(bench_dir):
        raise SystemExit(f"Benchmark directory does not exist: {bench_dir}")
    if not os.path.isfile(args.orgs):
        raise SystemExit(f"all_ORGs_union file does not exist: {args.orgs}")

    # Load ORG index (single file)
    print("[INFO] Loading all_ORGs_union (single JSON file)...")
    org_index = load_org_index_file(args.orgs)
    if not org_index:
        print("[WARN] all_ORGs_union is empty: only ClarAVy cleaning will run.")

    out_dir = bench_dir if args.inplace else args.out
    if not args.inplace:
        os.makedirs(out_dir, exist_ok=True)

    total, ok = 0, 0
    for fn in os.listdir(bench_dir):
        if not fn.lower().endswith(".json"):
            continue
        sha1 = os.path.splitext(fn)[0].lower()
        fp_in = os.path.join(bench_dir, fn)
        fp_out = os.path.join(out_dir, fn)
        org_info = org_index.get(sha1, {})
        success = process_one_file(fp_in, fp_out, org_info, sha1)
        total += 1
        ok += 1 if success else 0

    print(f"[DONE] Completed: {ok}/{total} files succeeded")
    if args.inplace:
        print(f"[NOTE] Overwritten in place: {bench_dir}")
    else:
        print(f"[NOTE] Written to: {out_dir}")
    print(f"[LOG] Log file: {log_filename}")

if __name__ == "__main__":
    main()
