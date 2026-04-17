import os
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Set, Any
from neo4j import GraphDatabase
from tqdm import tqdm
from path_var import WORKSPACE, LOG_PATH

# --------------------------- 日志配置 ---------------------------
LOG_FILENAME = 'graph_reranker_dump.log'
log_dir = os.path.join(WORKSPACE, LOG_PATH)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, LOG_FILENAME),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# --------------------------- 工具函数 ---------------------------
def load_json(path: str) -> Any:
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def build_hash_to_sha1_map(csv_path: str) -> Dict[str, str]:
    if not os.path.exists(csv_path): return {}
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    mp = {}
    for _, row in df.iterrows():
        sha1 = row.get("sha1", "").lower().strip()
        if not sha1: continue
        for col in ("md5", "sha1", "sha256", "sha512"):
            h = row.get(col, "").lower().strip()
            if h: mp[h] = sha1
    return mp

def parse_candidates(doc_json_path, vec_json_path, hash_map):
    """
    解析并合并两个来源的候选集，返回 {query_sha1: [unique_candidate_sha1s]}
    """
    merged_data = {} # type: Dict[str, Set[str]]

    # 1. 解析 Document Retrieval (Retrieval-Augmented-Few-shot_all_all.json)
    # 格式: {"query_hash": {"demo1": "hash", ...}}
    doc_data = load_json(doc_json_path)
    for q_hash, demos in doc_data.items():
        q_sha1 = hash_map.get(q_hash.lower(), q_hash.lower())
        if q_sha1 not in merged_data: merged_data[q_sha1] = set()
        
        for k, v in demos.items():
            if k.startswith("demo") and not k.endswith("_output"):
                c_sha1 = hash_map.get(v.lower(), v.lower())
                if c_sha1 and c_sha1 != q_sha1:
                    merged_data[q_sha1].add(c_sha1)

    # 2. 解析 Vector Retrieval (sample_attribution_top20_rewrite.json)
    # 格式: [{"hash": "query_hash", "matches": [{"match_hash": "..."}]}]
    vec_data = load_json(vec_json_path)
    if isinstance(vec_data, list):
        for item in vec_data:
            q_hash = item.get("hash")
            if not q_hash: continue
            q_sha1 = hash_map.get(q_hash.lower(), q_hash.lower())
            
            if q_sha1 not in merged_data: merged_data[q_sha1] = set()
            
            for match in item.get("matches", []):
                m_hash = match.get("match_hash")
                if m_hash:
                    c_sha1 = hash_map.get(m_hash.lower(), m_hash.lower())
                    if c_sha1 and c_sha1 != q_sha1:
                        merged_data[q_sha1].add(c_sha1)

    # 转换 Set 为 List
    return {k: list(v) for k, v in merged_data.items()}

# --------------------------- 核心类 ---------------------------
class GraphDataDumper:
    def __init__(self, uri, auth, benchmark_vs_path, hash_map):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.vs_path = benchmark_vs_path
        self.hash_map = hash_map
        self.local_features_cache = {}
        self.avg_lengths = {} 
        
        # ID 生成逻辑 (必须与 kg_builder 一致)
        self.feature_configs = [
            {"file": "malcat_edges.json", "type": "MalcatFeature", "id_gen": lambda e: e.get("feature_name")},
            {"file": "exif_edges.json", "type": "ExifTag", "id_gen": lambda e: f"exif:{e['tag_name']}:{e['normalized_value']}"},
            {"file": "ioc_edges.json", "type": "IoCNorm", "id_gen": lambda e: f"ioc:{e['ioc_type']}:{e['normalized_value']}"},
            {"file": "string_edges.json", "type": "StringGroup", "id_gen": lambda e: e.get("group_id")},
            {"file": "func_edges.json", "type": "Function", "id_gen": lambda e: f"{e['type']}:{e['name']}"},
            {"file": "mutex_edges.json", "type": "Mutex", "id_gen": lambda e: f"{e['type']}:{e['name']}"}
        ]
        
        self._preload_local_features()
        self._compute_global_avg_lengths()

    def close(self):
        self.driver.close()

    def _preload_local_features(self):
        """加载 Query (本地测试集) 的特征"""
        logging.info("Preloading local features from vector_store_benchmark...")
        for config in self.feature_configs:
            path = os.path.join(self.vs_path, config["file"])
            data = load_json(path)
            if not data: continue
            for item in data:
                raw_h = item.get("hash", "").lower()
                sha1 = self.hash_map.get(raw_h, raw_h)
                if not sha1: continue
                feat_id = config["id_gen"](item)
                if not feat_id: continue
                
                if sha1 not in self.local_features_cache:
                    self.local_features_cache[sha1] = {}
                # 记录特征 ID 及其类型 (因为 Query 不在 Neo4j，我们只能根据来源文件知道类型)
                self.local_features_cache[sha1][feat_id] = config["type"]

    def _compute_global_avg_lengths(self):
        """计算全图的平均特征长度 (avgdl)"""
        logging.info("Computing global average lengths from Neo4j...")
        # 默认值
        self.avg_lengths = {
            "MalcatFeature": 20.0, "ExifTag": 5.0, "IoCNorm": 2.0,
            "StringGroup": 50.0, "Function": 150.0, "Mutex": 1.5
        }
        try:
            with self.driver.session() as session:
                for label in self.avg_lengths.keys():
                    rel_type = "HAS_" + label.upper()
                    if label == "StringGroup": rel_type = "HAS_STRING_GROUP"
                    if label == "MalcatFeature": rel_type = "HAS_MALCAT"
                    
                    # 抽样 2000 个样本计算平均值
                    query = f"""
                    MATCH (s:Sample) WITH s LIMIT 2000
                    MATCH (s)-[:{rel_type}]->(f)
                    RETURN count(f) as cnt
                    """
                    res = session.run(query)
                    counts = [r["cnt"] for r in res]
                    if counts:
                        self.avg_lengths[label] = sum(counts) / len(counts)
            logging.info(f"Avg Lengths: {self.avg_lengths}")
        except Exception as e:
            logging.error(f"Error computing avg lengths: {e}")

    def fetch_candidates_data(self, all_candidates: Set[str]) -> Dict[str, Dict]:
        """
        批量获取所有涉及到的 Candidates 的特征和长度信息
        返回: { sha1: { "features": {id: {type, idf}}, "lengths": {type: count} } }
        """
        logging.info(f"Fetching data for {len(all_candidates)} unique candidates...")
        result_map = {sha1: {"features": {}, "lengths": {}} for sha1 in all_candidates}
        
        # 1. 获取特征详情 (IDF, Type)
        query_feats = """
        UNWIND $candidates AS sha1
        MATCH (s:Sample {sha1: sha1})-[r]->(f)
        WHERE f.graph_idf IS NOT NULL
        RETURN s.sha1 AS sha1, head(labels(f)) AS label, 
               coalesce(f.id, f.name, f.group_id) AS feat_id, f.graph_idf AS idf
        """
        
        # 2. 获取样本长度
        query_lengths = """
        UNWIND $candidates AS sha1
        MATCH (s:Sample {sha1: sha1})-[r]->(f)
        RETURN s.sha1 AS sha1, head(labels(f)) AS label, count(f) AS len
        """

        cand_list = list(all_candidates)
        batch_size = 1000
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(cand_list), batch_size), desc="Querying Neo4j"):
                batch = cand_list[i : i + batch_size]
                
                # Fetch Features
                for r in session.run(query_feats, candidates=batch):
                    c_sha1 = r["sha1"]
                    if c_sha1 in result_map:
                        result_map[c_sha1]["features"][r["feat_id"]] = {
                            "type": r["label"], "idf": float(r["idf"])
                        }
                
                # Fetch Lengths
                for r in session.run(query_lengths, candidates=batch):
                    c_sha1 = r["sha1"]
                    if c_sha1 in result_map:
                        result_map[c_sha1]["lengths"][r["label"]] = int(r["len"])
                        
        return result_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="12345678")
    parser.add_argument("--benchmark_dir", default="./vector_store_benchmark")
    parser.add_argument("--benchmark_csv", default="./samples/BenchmarkSet.csv")
    parser.add_argument("--doc_json", default="./Retrieval-Augmented-Few-shot_all_all.json")
    parser.add_argument("--vec_json", default="./sample_attribution_top20.json")
    parser.add_argument("--output_dump", default="tuning_data.json")
    args = parser.parse_args()

    # 1. 准备 Hash 映射
    hash_map = build_hash_to_sha1_map(args.benchmark_csv)

    # 2. 解析并合并候选集
    logging.info("Parsing input JSONs...")
    # merged_inputs: { query_sha1: [cand1, cand2, ...] }
    merged_inputs = parse_candidates(args.doc_json, args.vec_json, hash_map)
    # print(f"Parsed {len(merged_inputs)} queries with candidates from input JSONs.")
    
    # 3. 收集所有需要查询的 Candidate SHA1 (去重)
    all_unique_candidates = set()
    for cands in merged_inputs.values():
        all_unique_candidates.update(cands)
    # print(f"Total unique candidates to fetch from Neo4j: {len(all_unique_candidates)}")
    
    # 4. 初始化 Dumper
    dumper = GraphDataDumper(args.uri, (args.user, args.password), args.benchmark_dir, hash_map)
    
    try:
        # 5. 从 Neo4j 获取所有 Candidate 的数据
        candidates_data_cache = dumper.fetch_candidates_data(all_unique_candidates)
        
        # 6. 组装最终的 Dump 数据结构
        # 结构:
        # {
        #   "avg_lengths": {...},
        #   "queries": { q_sha1: { "local_feats": {fid: type} }, ... },
        #   "candidates_cache": { c_sha1: { "features": ..., "lengths": ... } },
        #   "tasks": [ { "query_sha1": ..., "candidates": [...] } ]
        # }
        
        dump_output = {
            "avg_lengths": dumper.avg_lengths,
            "queries": {},
            "candidates_cache": candidates_data_cache,
            "tasks": []
        }
        
        # 填充 Queries 本地特征 和 Tasks
        for q_sha1, cands in merged_inputs.items():
            if q_sha1 in dumper.local_features_cache:
                dump_output["queries"][q_sha1] = {
                    "local_feats": dumper.local_features_cache[q_sha1]
                }
                dump_output["tasks"].append({
                    "query_sha1": q_sha1,
                    "candidates": cands
                })
            else:
                logging.warning(f"Query {q_sha1} has no local features in vector_store, skipping.")

        # 7. 保存到磁盘
        logging.info(f"Saving dump to {args.output_dump}...")
        with open(args.output_dump, "w", encoding="utf-8") as f:
            json.dump(dump_output, f, ensure_ascii=False)
            
        logging.info("Data dump completed successfully.")

    finally:
        dumper.close()

if __name__ == "__main__":
    main()