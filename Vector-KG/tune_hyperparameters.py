import json
import itertools
import argparse
import pandas as pd
from tqdm import tqdm
import os

# ================= 配置区域 =================
# 定义超参数搜索空间
SEARCH_SPACE = {
    "Function": {"b": [0.6, 0.75, 0.85, 0.95], "weight": [0.1, 0.2, 0.3, 0.5]},
    "StringGroup": {"b": [0.4, 0.6, 0.8], "weight": [0.5, 0.8, 1.0]},
    # 其他类型固定
    "MalcatFeature": {"b": 0.6, "weight": 0.5},
    "ExifTag":       {"b": 0.3, "weight": 1.5},
    "IoCNorm":       {"b": 0.05, "weight": 2.0},
    "Mutex":         {"b": 0.05, "weight": 2.0}
}
K1 = 1.2

# ================= 核心函数 =================

def load_all_tags(bench_csv, r_index_csv):
    """
    加载测试集(BenchmarkSet)和知识库(R_IndexSet)的所有标签。
    返回: {sha1: normalized_tag}
    """
    print("Loading Tags from CSVs...")
    all_tags = {}
    
    csv_files = [bench_csv, r_index_csv]
    
    for path in csv_files:
        if not os.path.exists(path):
            print(f"Warning: CSV not found: {path}")
            continue
            
        try:
            df = pd.read_csv(path, dtype=str, encoding='utf-8').fillna("")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, encoding='gbk').fillna("")
            
        count = 0
        for _, row in df.iterrows():
            tag = row.get("Normalized_Tag", "").strip().lower()
            if not tag: continue
            
            # 建立多重哈希映射，确保能查到
            for col in ["sha1", "md5", "sha256", "sha512"]:
                h = row.get(col, "").strip().lower()
                if h:
                    all_tags[h] = tag
            count += 1
        print(f"  Loaded {count} rows from {os.path.basename(path)}")

    print(f"Total unique hash-tag entries: {len(all_tags)}")
    return all_tags

def bm25_score(q_feats, c_data, params, avg_lengths):
    """ 计算单对 Query-Candidate 的分数 """
    c_features = c_data["features"]
    c_lengths = c_data["lengths"]
    common_ids = set(q_feats.keys()) & set(c_features.keys())
    
    total_score = 0.0
    for fid in common_ids:
        feat_info = c_features[fid]
        f_type = feat_info["type"]
        idf = feat_info["idf"]
        
        p = params.get(f_type, {"b": 0.5, "weight": 0.5})
        b = p["b"]
        weight = p["weight"]
        doc_len = c_lengths.get(f_type, 0)
        avg_len = avg_lengths.get(f_type, 1.0) or 1.0
        
        numerator = idf * (K1 + 1)
        denominator = 1 + K1 * (1 - b + b * (doc_len / avg_len))
        score = weight * (numerator / denominator)
        total_score += score
    return total_score

def eval_one_config(config_pack):
    """ 评估一个配置的 Top-1 准确率 """
    params, tasks, queries_data, candidates_cache, avg_lengths, gt_map = config_pack
    
    correct_top1 = 0
    total = 0
    
    for task in tasks:
        q_sha1 = task["query_sha1"]
        target_tag = gt_map.get(q_sha1)
        if not target_tag: continue # 只有 GT 中有的 Query 才参与评估
        
        q_feats = queries_data.get(q_sha1, {}).get("local_feats", {})
        candidates = task["candidates"]
        
        scores = []
        for c_sha1 in candidates:
            c_data = candidates_cache.get(c_sha1)
            if not c_data: continue
            s = bm25_score(q_feats, c_data, params, avg_lengths)
            scores.append((c_sha1, s))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            top1_hash = scores[0][0]
            top1_tag = gt_map.get(top1_hash, "UNKNOWN")
            if top1_tag == target_tag:
                correct_top1 += 1
        total += 1
        
    acc1 = correct_top1 / total if total else 0
    return acc1, params

def generate_final_json(best_params, tasks, queries_data, candidates_cache, avg_lengths, gt_map, output_file):
    """ 使用最佳参数生成 final_RAG.json """
    print(f"\nGenerating {output_file} with best parameters...")
    
    final_output = {}
    
    for task in tqdm(tasks, desc="Final Reranking"):
        q_sha1 = task["query_sha1"]
        q_feats = queries_data.get(q_sha1, {}).get("local_feats", {})
        candidates = task["candidates"]
        
        scores = []
        for c_sha1 in candidates:
            c_data = candidates_cache.get(c_sha1)
            if not c_data: continue
            s = bm25_score(q_feats, c_data, best_params, avg_lengths)
            scores.append((c_sha1, s))
            
        # 降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 取 Top 2
        entry = {}
        for rank, (c_hash, score) in enumerate(scores[:2]):
            idx = rank + 1
            entry[f"demo{idx}"] = c_hash
            # 从 R_IndexSet (gt_map) 获取标签，如果没有则 UNKNOWN
            entry[f"demo{idx}_output"] = gt_map.get(c_hash, "UNKNOWN")
            # 可选：如果你想保存分数用于调试
            # entry[f"demo{idx}_score"] = score 
            
        final_output[q_sha1] = entry
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"Saved final results to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", default="tuning_data.json")
    parser.add_argument("--benchmark_csv", default="./samples/BenchmarkSet.csv")
    parser.add_argument("--r_index_csv", default="./samples/R_IndexSet.csv")
    parser.add_argument("--output_json", default="final_RAG.json")
    parser.add_argument("--params_path", default="./best_bm25_params.json", help="Path to best_bm25_params.json")
    args = parser.parse_args()

    # 1. 加载 Dump 数据
    print("Loading tuning_data.json...")
    try:
        with open(args.dump_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: tuning_data.json not found.")
        return

    avg_lengths = data["avg_lengths"]
    queries_data = data["queries"]
    candidates_cache = data["candidates_cache"]
    tasks = data["tasks"]
    print(f"{len(tasks)} tasks loaded from dump.")
    
    # 2. 加载全量标签 (Query + Candidate)
    gt_map = load_all_tags(args.benchmark_csv, args.r_index_csv)
    
    # 3. 过滤有效评估任务 (仅用于调参，生成时会对所有任务生成)
    eval_tasks = [t for t in tasks if t["query_sha1"] in gt_map]
    print(f"Tasks usable for accuracy evaluation: {len(eval_tasks)} / {len(tasks)}")

    # 4. 生成参数组合
    configs = []
    func_opts = list(itertools.product(SEARCH_SPACE["Function"]["b"], SEARCH_SPACE["Function"]["weight"]))
    str_opts = list(itertools.product(SEARCH_SPACE["StringGroup"]["b"], SEARCH_SPACE["StringGroup"]["weight"]))
    
    for (fb, fw) in func_opts:
        for (sb, sw) in str_opts:
            p = {
                "Function": {"b": fb, "weight": fw},
                "StringGroup": {"b": sb, "weight": sw},
                "MalcatFeature": SEARCH_SPACE["MalcatFeature"],
                "ExifTag": SEARCH_SPACE["ExifTag"],
                "IoCNorm": SEARCH_SPACE["IoCNorm"],
                "Mutex": SEARCH_SPACE["Mutex"]
            }
            configs.append(p)

    with open(args.params_path, "r", encoding="utf-8") as f:
        best_config = json.load(f)

    # 5. 生成最终 JSON
    if best_config:
        generate_final_json(
            best_config, tasks, queries_data, candidates_cache, 
            avg_lengths, gt_map, args.output_json
        )

if __name__ == "__main__":
    main()