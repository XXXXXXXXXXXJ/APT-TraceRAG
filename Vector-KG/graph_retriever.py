#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List, Set, Tuple

import pandas as pd
from neo4j import GraphDatabase
import logging
#from path_var import WORKSPACE, LOG_PATH
#
#LOG_FILENAME = 'graph_retriever.log'
#log_dir = os.path.join(WORKSPACE, LOG_PATH)
#if not os.path.exists(log_dir):
#    os.makedirs(log_dir, exist_ok=True)
#log_filename = os.path.join(log_dir, LOG_FILENAME)
#
#logging.basicConfig(
#    filename=log_filename,
#    filemode='a',
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s',
#    encoding='utf-8'
#)



# ============================================================
# Utilities
# ============================================================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any, compact: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            # separators=(',', ':') removes spaces around key:value and item,item
            json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# BenchmarkSet.csv -> map any hash to sha1
# ============================================================
def build_hash_to_sha1_map(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    mp: Dict[str, str] = {}
    for _, row in df.iterrows():
        sha1 = row.get("sha1", "").lower()
        if not sha1:
            continue
        for col in ("md5", "sha1", "sha256", "sha512"):
            h = row.get(col, "").lower()
            if h:
                mp[h] = sha1
    return mp


# ============================================================
# 1 Normalize vector_store_benchmark/samples.json
#    —— add sha1 for each sample (in-memory only)
# ============================================================
def normalize_benchmark_samples(
    samples_json_path: str,
    hash2sha1: Dict[str, str]
) -> Dict[str, dict]:
    samples = load_json(samples_json_path)
    out: Dict[str, dict] = {}

    for s in samples:
        raw_hash = s.get("hash", "").lower()
        sha1 = hash2sha1.get(raw_hash)
        if not sha1:
            continue
        s2 = dict(s)
        s2["sha1"] = sha1
        out[sha1] = s2

    return out


# ============================================================
# 2 Extract local features for query from vector_store_benchmark
#    —— same schema as kg_builder.py (no StringGroup)
# ============================================================
def load_local_features_from_vector_store_benchmark(
    vs_benchmark_dir: str,
    samples_by_sha1: Dict[str, dict],
    hash2sha1: Dict[str, str],
) -> Dict[str, Dict[str, Set[str]]]:
     
    feats: Dict[str, Dict[str, Set[str]]] = {}

    def ensure(sha1: str):
        if sha1 not in feats:
            feats[sha1] = {
                "MalcatFeature": set(),
                "ExifTag": set(),
                "IoCNorm": set(),
                "Function": set(),
                "Mutex": set(),
            }
        return feats[sha1]

    # Initialize (ensure queries have empty sets even if no edges)
    for sha1 in samples_by_sha1:
        ensure(sha1)

    # --- malcat_edges.json ---
    path = os.path.join(vs_benchmark_dir, "malcat_edges.json")
    if os.path.exists(path):
        for e in load_json(path):
            sha1 = hash2sha1.get(e.get("hash", "").lower())
            if not sha1:
                continue
            fname = e.get("feature_name")
            if fname:
                ensure(sha1)["MalcatFeature"].add(fname)

    # --- exif_edges.json ---
    path = os.path.join(vs_benchmark_dir, "exif_edges.json")
    if os.path.exists(path):
        for e in load_json(path):
            sha1 = hash2sha1.get(e.get("hash", "").lower())
            if not sha1:
                continue
            tn = e.get("tag_name")
            nv = e.get("normalized_value")
            if tn is not None and nv is not None:
                ensure(sha1)["ExifTag"].add(f"exif:{tn}:{nv}")

    # --- ioc_edges.json ---
    path = os.path.join(vs_benchmark_dir, "ioc_edges.json")
    if os.path.exists(path):
        for e in load_json(path):
            sha1 = hash2sha1.get(e.get("hash", "").lower())
            if not sha1:
                continue
            it = e.get("ioc_type")
            nv = e.get("normalized_value")
            if it is not None and nv is not None:
                ensure(sha1)["IoCNorm"].add(f"ioc:{it}:{nv}")

    # --- func_edges.json ---
    path = os.path.join(vs_benchmark_dir, "func_edges.json")
    if os.path.exists(path):
        for e in load_json(path):
            sha1 = hash2sha1.get(e.get("hash", "").lower())
            if not sha1:
                continue
            t = e.get("type")
            n = e.get("name")
            if t is not None and n is not None:
                ensure(sha1)["Function"].add(f"{t}:{n}")

    # --- mutex_edges.json ---
    path = os.path.join(vs_benchmark_dir, "mutex_edges.json")
    if os.path.exists(path):
        for e in load_json(path):
            sha1 = hash2sha1.get(e.get("hash", "").lower())
            if not sha1:
                continue
            t = e.get("type")
            n = e.get("name")
            if t is not None and n is not None:
                ensure(sha1)["Mutex"].add(f"{t}:{n}")

    return feats

def log_if_unique_feature(
    query_sha1: str,
    side: str,          # "query_demo" or "query_match"
    kind: str,          # MalcatFeature / ExifTag / IoCNorm
    node: dict
):
    """
    If org_stats contains only one org, treat it as a unique feature and log it.
    """
    org_stats = node.get("org_stats", [])
    if len(org_stats) == 1:
        # org_stats looks like [{"apt38": 7}]
        org_name = next(iter(org_stats[0].keys()))
        logging.info(
            "[UNIQUE_GENERIC] query=%s side=%s kind=%s node=%s org=%s",
            query_sha1, side, kind, node.get("id"), org_name
        )


# ============================================================
# Neo4j query layer (demo / match only)
# ============================================================
class KGQuery:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def run(self, q, **kw):
        with self.driver.session() as s:
            return list(s.run(q, **kw))

    def get_generic(self, sha1: str, kind: str) -> Set[str]:
        if kind == "MalcatFeature":
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_MALCAT]->(f) RETURN collect(DISTINCT f.name) AS x"
        elif kind == "ExifTag":
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_EXIF]->(f) RETURN collect(DISTINCT f.id) AS x"
        elif kind == "Function":
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_FUNCTION]->(f) RETURN collect(DISTINCT f.id) AS x"
        elif kind == "Mutex":
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_MUTEX]->(f) RETURN collect(DISTINCT f.id) AS x"
        elif kind == "StringGroup":
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_STRING_GROUP]->(f) RETURN collect(DISTINCT f.group_id) AS x"
        else:  # IoCNorm
            q = "MATCH (s:Sample {sha1:$h})-[:HAS_IOC]->(f) RETURN collect(DISTINCT f.id) AS x"

        r = self.run(q, h=sha1)
        return set(r[0]["x"] or []) if r else set()

    def node_org_stats(self, kind: str, nid: str) -> List[Dict[str, int]]:
        if kind == "MalcatFeature":
            q = """
            MATCH (f:MalcatFeature {name:$n})<-[:HAS_MALCAT]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        elif kind == "ExifTag":
            q = """
            MATCH (f:ExifTag {id:$n})<-[:HAS_EXIF]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        elif kind == "Function":
            q = """
            MATCH (f:Function {id:$n})<-[:HAS_FUNCTION]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        elif kind == "Mutex":
            q = """
            MATCH (f:Mutex {id:$n})<-[:HAS_MUTEX]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        elif kind == "StringGroup":
            q = """
            MATCH (f:StringGroup {group_id:$n})<-[:HAS_STRING_GROUP]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        else:
            q = """
            MATCH (f:IoCNorm {id:$n})<-[:HAS_IOC]-(s)-[:BELONGS_TO_ORG]->(o)
            RETURN o.name AS o, count(DISTINCT s) AS c
            """
        return [{r["o"]: r["c"]} for r in self.run(q, n=nid)]

    def feature_graph_idf_map(self, kind: str, ids: List[str]) -> Dict[str, float]:
        if not ids:
            return {}
        if kind == "MalcatFeature":
            label = "MalcatFeature"
            id_prop = "name"
        elif kind == "ExifTag":
            label = "ExifTag"
            id_prop = "id"
        elif kind == "Function":
            label = "Function"
            id_prop = "id"
        elif kind == "Mutex":
            label = "Mutex"
            id_prop = "id"
        elif kind == "StringGroup":
            label = "StringGroup"
            id_prop = "group_id"
        else:
            label = "IoCNorm"
            id_prop = "id"
        q = f"""
        UNWIND $ids AS fid
        MATCH (f:{label} {{{id_prop}: fid}})
        RETURN fid AS id, coalesce(f.graph_idf, 0.0) AS graph_idf
        """
        return {r["id"]: float(r["graph_idf"]) for r in self.run(q, ids=ids)}


# ============================================================
# Main flow
# ============================================================
GENERIC_KINDS = ("MalcatFeature", "ExifTag", "IoCNorm", "Function", "Mutex", "StringGroup")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j_user", default="neo4j")
    ap.add_argument("--neo4j_password", default="12345678")
    ap.add_argument("--benchmark_csv", default="./samples/BenchmarkSet.csv")
    ap.add_argument("--vs_benchmark", default="./vector_store_benchmark")
    ap.add_argument("--out_dir", default="./commonality")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # hash -> sha1
    hash2sha1 = build_hash_to_sha1_map(args.benchmark_csv)

    # Normalize samples.json
    samples_by_sha1 = normalize_benchmark_samples(
        os.path.join(args.vs_benchmark, "samples.json"),
        hash2sha1
    )

    # Local features for query
    local_feats = load_local_features_from_vector_store_benchmark(
        args.vs_benchmark,
        samples_by_sha1,
        hash2sha1
    )

    kg = KGQuery(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        for query_sha1 in samples_by_sha1.keys():
            q_feats = local_feats.get(query_sha1, {
                "MalcatFeature": set(),
                "ExifTag": set(),
                "IoCNorm": set(),
                "Function": set(),
                "Mutex": set(),
            })

            # Graph-IDF: top10 features by weight for this query
            idf_scores: Dict[Tuple[str, str], float] = {}
            for k in GENERIC_KINDS:
                ids = list(q_feats.get(k, set()))
                idf_map = kg.feature_graph_idf_map(k, ids)
                for fid, score in idf_map.items():
                    idf_scores[(k, fid)] = score

            top_graph_idf = [] 
            for idx, ((k, fid), _score) in enumerate(
                sorted(idf_scores.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:10],
                start=1,
            ):
                org_stats = kg.node_org_stats(k, fid)
                org_map = {}
                for d in org_stats:
                    for org, cnt in d.items():
                        org_map[org] = cnt
                top_graph_idf.append([fid, org_map])

            out = {
                "q": query_sha1,       # shorten "query_sha1" to "q"
                "t": top_graph_idf,    # shorten "top_graph_idf" to "t"
            }
            dump_json(os.path.join(args.out_dir, f"{query_sha1}.json"), out, compact=True)


    finally:
        kg.close()


if __name__ == "__main__":
    main()
