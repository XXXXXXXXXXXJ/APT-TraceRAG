#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VT flattening

Overview (single file runnable):
1) Flattening: use only VirusTotal report JSON (fields may be missing) to produce normalized text.
   - Field-to-text mapping:
       family -> "FAMILY: <name>"
       malcatYara -> "YARA: <rule>"
       exports -> "PE.EXPORT: <func>"
       IOCs -> "IOC.<TYPE>: <val>" (limit count)
       tools/packer/lang -> "TOOL: <tool>", "PACKER: <packer>", "LANG: <lang>"
       summary/desc -> "DESC: <text>"
       behavior (CMD, Process, Network, IDS)
   - Normalization: lowercase, drop www prefix, full-width to half-width, de-duplicate.
   - Long-text chunking: tokenizer-based 512–768 tokens (default 640), overlap≈120; MaxP aggregation.

Examples:
    python vt_flatten.py flatten --input_dir ./Executable/R_all --output_flat_docs ./Executable/R_Index_flat_docs_all.jsonl --output_chunks ./Executable/R_Index_chunks_all.jsonl --org_map ./all_ORGs_union.json --default_org unknown --max_tokens 2048 --overlap 256
    python vt_flatten.py flatten  --input_dir ./Executable/Benchmark_1000_all --output_flat_docs ./Executable/Benchmark_1000_flat_docs_all.jsonl --output_chunks ./Executable/Benchmark_1000_chunks_all.jsonl --org_map ./all_ORGs_union.json --default_org unknown --max_tokens 2048 --overlap 256
Notes:
- org_labels.tsv (optional): two columns separated by \t: doc_id\torg_label (gold). You can also fill via --default_org.
- JSON input: one VT JSON per file; filename (without extension) is used as doc_id by default.
- This script is offline; models must be available locally or cached. First run may need network to download weights.
"""

from __future__ import annotations
import os
import re
import sys
import json
import math
import glob
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional
from collections import Counter
import ujson as uj
from tqdm import tqdm
import logging
import glob


try:
    from path_var import WORKSPACE, LOG_PATH
except Exception:
    WORKSPACE = "."
    LOG_PATH = "."

LOG_FILENAME = 'vt_flatten.log'
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

# -----------------------------
# Configuration
# -----------------------------
# Truncate IOC lists that may contain thousands of items.
# In retrieval scenarios, BGE-M3 was trained with long-text data.
MAX_IOC_PER_TYPE = 512

# -----------------------------
# Tokenizer (robust fallback)
# -----------------------------
try:
    from transformers import AutoTokenizer
    # Try to load a local model; adjust the path or ensure network access if missing
    _TOK = AutoTokenizer.from_pretrained("./models/bert-base-multilingual-cased")
    print(f"[INFO] Loaded tokenizer vocab size: {_TOK.vocab_size}, fast={getattr(_TOK, 'is_fast', False)}")

    def count_tokens(text: str) -> int:
        return len(_TOK.encode(text, add_special_tokens=False))

    def split_by_tokens(text: str, max_len: int, overlap: int) -> List[str]:
        step = max(1, max_len - overlap)
        if getattr(_TOK, "is_fast", False):
            enc = _TOK(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False
            )   
            offsets = enc["offset_mapping"]
            N = len(offsets)
            chunks = []
            i = 0
            while i < N:
                j = min(i + max_len, N)
                char_start = offsets[i][0]
                char_end   = offsets[j - 1][1]
                # Try to snap to newline
                snap_window = 200
                s = max(char_start, char_end - snap_window)
                k = text.rfind("\n", s, char_end)
                if k != -1 and k > char_start + 32:
                    char_end = k
                pieces = text[char_start:char_end]
                chunks.append(pieces)
                i = min(i + step, N)
            return chunks

        ids = _TOK.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, len(ids), step):
            piece_ids = ids[start:start + max_len]
            if not piece_ids:
                break
            s = _TOK.decode(piece_ids)
            # basic cleanup
            s = re.sub(r"\s*::\s*", "::", s)
            s = re.sub(r"\s*\.\s*", ".",  s)
            s = re.sub(r"\s*_\s*", "_",  s)
            s = re.sub(r"\s{2,}", " ", s).strip()
            chunks.append(s)
        return chunks
    
except Exception:
    _WORD_RE = re.compile(r"\w+|\S", re.UNICODE)
    def split_by_tokens(text: str, max_len: int, overlap: int) -> List[str]:
        words = _WORD_RE.findall(text)
        approx_tok_per_word = 1/1.3
        max_words = int(max_len / approx_tok_per_word)
        overlap_words = int(overlap / approx_tok_per_word)
        chunks = []
        step = max(1, max_words - overlap_words)
        for i in range(0, len(words), step):
            piece = words[i:i+max_words]
            if not piece:
                break
            chunks.append(" ".join(piece))
        return chunks

# -----------------------------
# Normalization utilities
# -----------------------------
WWW_PREFIX = re.compile(r"^www\.", re.I)

def zenkaku_to_hankaku(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def norm_ioc(s: str) -> str:
    if not s: return ""
    s = zenkaku_to_hankaku(str(s).strip())
    
    s = s.lower()
    s = WWW_PREFIX.sub("", s)
    s = s.rstrip(".")
    return s

def uniq_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# -----------------------------
# VT -> Flat text mapping
# -----------------------------
@dataclass
class FlatDoc:
    doc_id: str
    sha1: str
    org: str
    flat_text: str
    chunks: List[str]

class VTFlattener:
    def __init__(self, max_tokens:int=2048, overlap:int=256):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def _emit(self, tag: str, val: Optional[str], buf: List[str]):
        if not val:
            return
        v = str(val).strip()
        if v:
            buf.append(f"{tag}: {v}")

    def _process_list(self, tag_prefix: str, data_list: List[Any], buf: List[str], limit: int = MAX_IOC_PER_TYPE):
   
        if not isinstance(data_list, list): return
        
       
        clean_items = []
        for item in data_list:
            if isinstance(item, (str, int, float)):
                clean_items.append(norm_ioc(str(item)))
            elif isinstance(item, dict):
                
                v = " ".join([str(x) for x in item.values() if x])
                clean_items.append(norm_ioc(v))
        
        clean_items = uniq_preserve(clean_items)
        
        for i, val in enumerate(clean_items):
            if i >= limit: break
            self._emit(tag_prefix, val, buf)

    def flatten_one(self, vt: Dict[str, Any], doc_id: str, sha1: str, org: str = "unknown") -> FlatDoc:
        lines: List[str] = []

        # 1. Family (TagClass / ClarAVy)
        fams: List[str] = []
        m1 = vt.get("MalFamily_ClarAVy")
        if isinstance(m1, str):
            for m in re.findall(r"FAM:([\w.-]+)", m1, flags=re.I):
                fams.append(norm_ioc(m))
        m2 = vt.get("MalFamily_TagClass")
        if isinstance(m2, str):
            try:
                d = json.loads(m2)
                fams.extend([norm_ioc(k) for k in d.keys()])
            except Exception:
                pass
        fams = uniq_preserve(fams)
        for f in fams:
            self._emit("FAMILY", f, lines)

        # 2. malcatYara (New)
        
        yara_data = vt.get("malcatYara") or {}
        if isinstance(yara_data, dict):
            for rule_name, is_hit in yara_data.items():
                if is_hit:
                    self._emit("YARA", norm_ioc(rule_name), lines)

        addi = vt.get("additional_info", {}) or {}

        # 3. Basic Metadata
        self._emit("FILE.TYPE", norm_ioc(str(vt.get("FileType_VT") or "")), lines)
        fsz = vt.get("FileSize")
        if isinstance(fsz, (int, float)): self._emit("FILE.SIZE", f"{int(fsz)}b", lines)
        
        ct = vt.get("Compile_time") or (addi.get("pe-timestamp") if isinstance(addi, dict) else None)
        if ct: self._emit("FILE.COMPILE_TIME", str(ct), lines)
        
        if addi.get("pe-machine-type"): self._emit("PE.MACHINE", norm_ioc(str(addi.get("pe-machine-type"))), lines)
        if addi.get("pe-entry-point"):  self._emit("PE.ENTRY",   norm_ioc(str(addi.get("pe-entry-point"))), lines)

        # 4. Exports (New)
        exports = addi.get("exports") or []
        if isinstance(exports, list):
            
            exports = uniq_preserve([norm_ioc(x) for x in exports])
            for i, exp in enumerate(exports):
                if i >= MAX_IOC_PER_TYPE * 2: 
                    break
                self._emit("PE.EXPORT", exp, lines)

        # 5. PE Sections
        secs = addi.get("sections") or {}
        if isinstance(secs, dict):
            for i in range(0, 8): # Keep first 8 sections
                pass 
        
        if isinstance(secs, list):
            for i, arr in enumerate(secs):
                if i >= 8: break
                if isinstance(arr, list) and len(arr) >= 4:
                    # [name, virt_addr, raw_size, entropy, ...]
                    try:
                        name = str(arr[0])
                        size = int(arr[2])
                        entro = float(arr[4]) if len(arr) > 4 else 0.0 
                        self._emit("PE.SECTION", f"{i} {norm_ioc(name)} sz={size} ent={round(entro,1)}", lines)
                    except: pass

        # 6. Imports
        imports = addi.get("imports") or {}
        if isinstance(imports, dict):
            for dll, apis in imports.items():
                if not isinstance(apis, list): continue
                dll_l = str(dll).lower()
                
                api_list = uniq_preserve([norm_ioc(a) for a in apis if isinstance(a, str)])
                for idx, api in enumerate(api_list):
                    if idx >= 50: break 
                    self._emit("PE.IMPORT", f"{dll_l}::{api}", lines)

        # 7. IOCs (New & Large Data Handling)
       
        iocs_data = vt.get("IOCs") or {}
        if isinstance(iocs_data, dict):
            for ioc_type, vals in iocs_data.items():
                if not isinstance(vals, list): continue
                
                type_tag = norm_ioc(ioc_type).upper() 
               
                clean_vals = uniq_preserve([norm_ioc(str(v)) for v in vals])
                
               
                count = 0
                for v in clean_vals:
                    if count >= MAX_IOC_PER_TYPE:
                        break
                    self._emit(f"IOC.{type_tag}", v, lines)
                    count += 1

        # 9. Tools / Packer / Lang (Heuristic)
        if vt.get("Packer"): self._emit("PACKER", norm_ioc(str(vt["Packer"])), lines)
        
        # 10. Summary / Desc
        for k in ("summary", "desc", "description"):
            if vt.get(k):
                self._emit("DESC", vt.get(k), lines)
                break
        # -------------------------------------------------------
        # 11. Dynamic Behaviour (Enhanced Section)
        # -------------------------------------------------------
        behav_src = vt
        if "command_executions" not in vt and isinstance(vt.get("data"), dict):
            behav_src = vt.get("data")
         
            if isinstance(behav_src.get("data"), dict): 
                behav_src = behav_src.get("data")
        
        # 11.1 Command Executions
     
        cmds = behav_src.get("command_executions") or behav_src.get("behavior", {}).get("command_executions")
        self._process_list("BEHAV.CMD", cmds, lines)

        # 11.2 Processes
        p_inj = behav_src.get("processes_injected") or behav_src.get("behavior", {}).get("processes_injected")
        self._process_list("BEHAV.PROINJECT", p_inj, lines)
        
        p_kill = behav_src.get("processes_killed") or behav_src.get("behavior", {}).get("processes_killed")
        self._process_list("BEHAV.PROKILL", p_kill, lines)

        # 11.3 Invokes
        invokes = behav_src.get("invokes")
        self._process_list("BEHAV.INVOKE", invokes, lines)

        # 11.4 System Changes
        svcs = behav_src.get("services_opened") or behav_src.get("services_created")
        self._process_list("BEHAV.SERVICE", svcs, lines)
        
        mutex = behav_src.get("mutexes_created")
        self._process_list("BEHAV.MUTEX", mutex, lines)

        # 11.5 Network
        # DNS
        dns = behav_src.get("dns_lookups")
        if isinstance(dns, list):
            clean_dns = []
            for item in dns:
                if isinstance(item, dict):
                   
                    h = item.get("hostname")
                    if h: clean_dns.append(h)
                elif isinstance(item, str):
                    clean_dns.append(item)
            self._process_list("NET.DNS", clean_dns, lines)

        # JA3 (Raw list)
        ja3s = behav_src.get("ja3_digests")
        self._process_list("NET.JA3", ja3s, lines)

        # TLS (Structured)
        tls_list = behav_src.get("tls")
        if isinstance(tls_list, list):
            for t in tls_list:
                if not isinstance(t, dict): continue
               
                if t.get("ja3"): self._emit("NET.JA3", t["ja3"], lines)
                if t.get("issuer_cn"): self._emit("NET.CERT.ISSUER", norm_ioc(t["issuer_cn"]), lines)
                if t.get("subject_cn"): self._emit("NET.CERT.SUBJECT", norm_ioc(t["subject_cn"]), lines)

        # 11.6 Security Alerts (IDS / SIGs) (High Value)
        # IDS Alerts
        ids = behav_src.get("ids_alerts")
        if isinstance(ids, list):
            for alert in ids:
                if not isinstance(alert, dict): continue
                msg = alert.get("rule_msg")
                
                if msg: self._emit("BEHAV.IDS_ALERT", norm_ioc(msg), lines)

        # Signature Matches (High Severity)
        sigs = behav_src.get("signature_matches")
        if isinstance(sigs, list):
            for s in sigs:
                if not isinstance(s, dict): continue
                
                txt = s.get("description") or s.get("name")
                if txt: self._emit("BEHAV.SIG_MATCH", norm_ioc(txt), lines)

       
        text = "\n".join(lines)
        text = zenkaku_to_hankaku(text).lower()

       
        chunks = split_by_tokens(text, self.max_tokens, self.overlap)
        return FlatDoc(doc_id=doc_id, sha1=sha1, org=org, flat_text=text, chunks=chunks)

# -----------------------------
# IO helpers
# -----------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
        try:
            return uj.loads(txt)
        except Exception:
            return json.loads(txt)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(uj.dumps(r, ensure_ascii=False) + "\n")

def load_org_map_json(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for item in data:
        if not isinstance(item, dict): continue
        sha1 = (item.get("sha1") or "").lower()
        if not sha1: continue
        name = item.get("name") or "unknown"
        aliases = item.get("aliases") or []
        aliases_norm = uniq_preserve([norm_ioc(str(a)) for a in aliases if a])
        out[sha1] = {"name": name, "aliases": aliases_norm}
    return out

_SHA1_RE_DEFAULT = re.compile(r'([A-Fa-f0-9]{40})')
def sha1_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = _SHA1_RE_DEFAULT.search(name)
    return (m.group(1).lower() if m else "")

def write_flat_docs_jsonl(path: str, docs: Iterable[FlatDoc]):
    rows = []
    for d in docs:
        rows.append({
            "doc_id": d.doc_id,
            "sha1": d.sha1,
            "n_chars": len(d.flat_text),
            "n_chunks": len(d.chunks),
            "flat_text": d.flat_text
        })
    write_jsonl(path, rows)

def write_chunks_jsonl(path: str, docs: Iterable[FlatDoc]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for d in docs:
            for i, ch in enumerate(d.chunks):
                row = {
                    "doc_id": d.doc_id,
                    "sha1": d.sha1,
                    "chunk_id": i,
                    "text": ch
                }
                f.write(uj.dumps(row, ensure_ascii=False) + "\n")

# -----------------------------
# CLI Commands
# -----------------------------
def cmd_flatten(args):
    flattener = VTFlattener(max_tokens=args.max_tokens, overlap=args.overlap)
    org_map = load_org_map_json(args.org_map)
    docs: List[FlatDoc] = []
    
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.json')))
    print(f"[INFO] Found {len(files)} files in {args.input_dir}")

    for fp in tqdm(files, desc='flatten'):
        try:
            vt = read_json(fp)
        except Exception as e:
            
            tqdm.write(f"[WARN] Skipping corrupted file: {os.path.basename(fp)} Error: {e}")
            logging.error(f"Corrupted JSON {fp}: {e}")
        did = os.path.splitext(os.path.basename(fp))[0]
        sha1 = sha1_from_filename(fp)
        org_entry = org_map.get(sha1, None)
        org_name = org_entry.get("name", args.default_org) if org_entry else args.default_org
        
        doc = flattener.flatten_one(vt, doc_id=did, sha1=sha1, org=org_name)
        docs.append(doc)

    write_flat_docs_jsonl(args.output_flat_docs, docs)
    write_chunks_jsonl(args.output_chunks, docs)
    print(f"[OK] Wrote {len(docs)} docs")
    print(f"  - flat docs -> {args.output_flat_docs}")
    print(f"  - chunks    -> {args.output_chunks}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='VT Flatten (Updated 2026)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_f = sub.add_parser('flatten', help='Flatten VT JSON')
    ap_f.add_argument('--input_dir', required=True)
    ap_f.add_argument('--output_flat_docs', required=True)
    ap_f.add_argument('--output_chunks', required=True)
    ap_f.add_argument('--org_map', default=None)
    ap_f.add_argument('--default_org', default='unknown')
    ap_f.add_argument('--max_tokens', type=int, default=640)
    ap_f.add_argument('--overlap', type=int, default=120)
    ap_f.set_defaults(func=cmd_flatten)

    args = ap.parse_args()
    args.func(args)