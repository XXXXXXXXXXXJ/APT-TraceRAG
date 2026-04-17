"""
nohup python retrieval_pipeline.py > retrieval_pipeline.log 2>&1 &

Three-stage retrieval → fusion → rerank (BM25 + bge-m3 + RRF + bge-reranker-v2-m3)
Adapted for the updated vt_flatten.py (long text + behavioral features)

Inputs:
    ./Executable/R_Index_flat_docs_all.jsonl
    ./Executable/R_Index_chunks_all.jsonl
    ./Executable/Benchmark_1000_flat_docs_all.jsonl
    ./Executable/Benchmark_1000_chunks_all.jsonl
    ./all_ORGs_union.json

Outputs:
    ./Executable/Retrieval-Augmented-Few-shot_all.json (Top2)
    ./Executable/Retrieval-Augmented-Few-shot_all_all.json (Top20)
"""

import os
import re
import json
import ujson
import time
import math
import faiss
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rank_bm25 import BM25Okapi
from openai import OpenAI
import requests
import logging

# ------------------------
# Configuration
# ------------------------

OPENAI_BASE_URL = "http://127.0.0.1:9997/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "no empty")
EMBED_MODEL = "bge-m3"
RERANK_MODEL = "bge-reranker-v2-m3"

PATH_INDEX_DOCS = "./Executable/R_Index_flat_docs_all.jsonl"
PATH_INDEX_CHUNKS = "./Executable/R_Index_chunks_all.jsonl"
PATH_BENCH_DOCS = "./Executable/Benchmark_1000_flat_docs_all.jsonl"
PATH_BENCH_CHUNKS = "./Executable/Benchmark_1000_chunks_all.jsonl"
PATH_ORG_UNION = "./all_ORGs_union.json"
PATH_OUTPUT_TOP2 = "./Executable/Retrieval-Augmented-Few-shot_all.json"
PATH_OUTPUT_TOP20 = "./Executable/Retrieval-Augmented-Few-shot_all_all.json"
PATH_FAISS_INDEX = "./Executable/chunk_vectors_all.faiss"

TOPK_CHUNK_SPARSE = 300
TOPK_CHUNK_DENSE = 300
RRF_K = 60
TOPK_DOC_CANDIDATE = 80
TOPK_FINAL = 20
MAX_CHUNKS_PER_DOC_FOR_RERANK = 3
EMBED_BATCH = 16
# ------------------------
# Utilities
# ------------------------

try:
    from path_var import WORKSPACE, LOG_PATH
except Exception:
    WORKSPACE = "."
    LOG_PATH = "."

LOG_FILENAME = 'retrieval_pipeline_final.log'
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

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield ujson.loads(line)

def simple_tokenize(text: str):
    return re.split(r"\s+", text.lower().strip())

def build_query_text(flat_text: str,
                     max_imports: int = 50,
                     max_iocs: int = 50, 
                     max_lines: int = 600,
                     head_chars: int = 4000) -> str:
    """
    Build a compact query text.
    Priority: core metadata (Family, Yara, Type) > exports > imports (partial) > IOCs (partial) > head snippet.
    """
    lines = flat_text.splitlines()

    # 1. High-value core lines
    core_lines = []
    # 2. Imports
    import_lines = []
    # 3. IOCs (hashes, paths, URLs, etc.)
    ioc_lines = []
    behav_lines = []
    net_lines = []
    
    # Core field prefixes
    core_prefixes = (
        "family:", "yara:", "file.type:", "file.size:", "file.compile_time:",
        "lang:", "pe.machine:", "pe.entry:", "pe.export:", "pe.section:"
    )
    prefixes_behav = ("behav.cmd:", "behav.inject:", "behav.kill:", "behav.mutex:", "behav.service:", "behav.invoke:")
    prefixes_net = ("net.ja3:", "net.cert.issuer:", "net.cert.subject:")

    for ln in lines:
        ln_strip = ln.strip()
        if not ln_strip: continue
        
        if ln_strip.startswith(core_prefixes):
            core_lines.append(ln_strip)
        elif ln_strip.startswith("pe.import:"):
            if len(import_lines) < max_imports:
                import_lines.append(ln_strip)
        elif ln_strip.startswith("ioc."):
            if len(ioc_lines) < max_iocs:
                ioc_lines.append(ln_strip)
        elif ln_strip.startswith(prefixes_behav):
            behav_lines.append(ln_strip)
        elif ln_strip.startswith(prefixes_net):
            net_lines.append(ln_strip)

    head = flat_text[:head_chars]

    pieces = []
    if core_lines:
        pieces.append("# CORE\n" + "\n".join(core_lines))
    if ioc_lines:
        pieces.append("# IOCs (partial)\n" + "\n".join(ioc_lines))
    if import_lines:
        pieces.append("# IMPORTS (partial)\n" + "\n".join(import_lines))
    if behav_lines:
        pieces.append("# BEHAVIOR\n" + "\n".join(behav_lines))
    if net_lines:
        pieces.append("# NETWORK_FINGERPRINT\n" + "\n".join(net_lines))
    
    # Fallback to the head snippet
    pieces.append("# SNIPPET\n" + head)

    compact = "\n".join(pieces)
    
    # Hard cap on total lines
    c_lines = compact.splitlines()
    if len(c_lines) > max_lines:
        compact = "\n".join(c_lines[:max_lines])

    return compact

def _looks_cjk(s: str) -> bool:
    return re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', s) is not None

def truncate_for_model(text: str, max_tokens_budget: int = 6000) -> str:
    if not text: return text
    # Rough estimate: CJK 1 char = 1 token, English 4 chars = 1 token
    factor = 1 if _looks_cjk(text) else 4
    max_chars = max_tokens_budget * factor
    return text[:max_chars]

def hard_clip_8192(text: str) -> str:
    return text[:8000] # Leave some buffer

def build_bm25(corpus_texts):
    tokenized = [simple_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized), tokenized

def bm25_search_doclevel(bm25, tokenized_corpus, chunk_meta, q_text, top_chunk_k=300, top_doc_k=80):
    q_tokens = simple_tokenize(q_text)
    scores = bm25.get_scores(q_tokens)
    k = min(top_chunk_k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    
    doc_best = {}
    doc_top_chunks = defaultdict(list)
    for idx in top_idx:
        doc_id, chunk_id, sha1 = chunk_meta[idx]
        sc = float(scores[idx])
        if (doc_id not in doc_best) or (sc > doc_best[doc_id][0]):
            doc_best[doc_id] = (sc, chunk_id)
        doc_top_chunks[doc_id].append((sc, chunk_id))
        
    ranked = sorted(
        [{"doc_id": d, "score": s_c[0], "best_chunk_id": s_c[1]} for d, s_c in doc_best.items()],
        key=lambda x: x["score"], reverse=True
    )[:top_doc_k]
    
    for it in ranked:
        doc_id = it["doc_id"]
        it["_top_chunks"] = sorted(doc_top_chunks[doc_id], key=lambda x: x[0], reverse=True)[:MAX_CHUNKS_PER_DOC_FOR_RERANK]
    return ranked

def l2_normalize(mat: np.ndarray):
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def openai_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def embed_texts(texts):
    client = openai_client()
    vecs = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding chunks", unit="batch"):
        batch = texts[i:i+EMBED_BATCH]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
            vecs.append(emb)
        except Exception as e:
            print(f"[ERR] Embed batch failed: {e}")
            # fallback zero vectors
            vecs.append(np.zeros((len(batch), 1024), dtype=np.float32))
            
    mat = np.vstack(vecs)
    return l2_normalize(mat)

def build_faiss_index(embeds: np.ndarray):
    d = embeds.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeds)
    return index

def dense_search_doclevel(index_faiss, q_text, chunk_texts, chunk_meta, top_chunk_k=300, top_doc_k=80):
    client = openai_client()
    # Compact the query text
    q_proc = build_query_text(q_text, max_imports=30, max_iocs=30) 
    q_proc = truncate_for_model(q_proc, 6000)
    q_proc = hard_clip_8192(q_proc)
    
    q_emb_resp = client.embeddings.create(model=EMBED_MODEL, input=[q_proc])
    q_emb = np.array(q_emb_resp.data[0].embedding, dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    k = min(top_chunk_k, len(chunk_texts))
    D, I = index_faiss.search(q_emb.reshape(1, -1), k)
    idxs = I[0]
    dists = D[0]

    doc_best = {}
    doc_top_chunks = defaultdict(list)
    for score, idx in zip(dists, idxs):
        if idx == -1: continue
        doc_id, chunk_id, sha1 = chunk_meta[idx]
        sc = float(score)
        if (doc_id not in doc_best) or (sc > doc_best[doc_id][0]):
            doc_best[doc_id] = (sc, chunk_id)
        doc_top_chunks[doc_id].append((sc, chunk_id))

    ranked = sorted(
        [{"doc_id": d, "score": s_c[0], "best_chunk_id": s_c[1]} for d, s_c in doc_best.items()],
        key=lambda x: x["score"], reverse=True
    )[:top_doc_k]
    for it in ranked:
        doc_id = it["doc_id"]
        it["_top_chunks"] = sorted(doc_top_chunks[doc_id], key=lambda x: x[0], reverse=True)[:MAX_CHUNKS_PER_DOC_FOR_RERANK]
    return ranked

def rrf_fuse(bm25_docs, dense_docs, k=60, keep=80):
    ranks_bm25 = {d["doc_id"]: r+1 for r, d in enumerate(bm25_docs)}
    ranks_dense = {d["doc_id"]: r+1 for r, d in enumerate(dense_docs)}
    
    top_chunks_bm25 = {d["doc_id"]: d.get("_top_chunks", []) for d in bm25_docs}
    top_chunks_dense = {d["doc_id"]: d.get("_top_chunks", []) for d in dense_docs}

    all_ids = set(ranks_bm25) | set(ranks_dense)
    fused = []
    for did in all_ids:
        r1 = ranks_bm25.get(did, math.inf)
        r2 = ranks_dense.get(did, math.inf)
        score = (1.0 / (k + r1)) + (1.0 / (k + r2))
        fused.append({"doc_id": did, "rrf": score})
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    fused = fused[:keep]

    for it in fused:
        did = it["doc_id"]
        cand = []
        seen = set()
        for src_list in (top_chunks_bm25.get(did, []), top_chunks_dense.get(did, [])):
            for sc, cid in src_list:
                if cid not in seen:
                    cand.append((sc, cid))
                    seen.add(cid)
        cand = sorted(cand, key=lambda x: x[0], reverse=True)[:MAX_CHUNKS_PER_DOC_FOR_RERANK]
        it["_top_chunks"] = cand
    return fused

def call_rerank(query, documents):
    url = f"{OPENAI_BASE_URL.rstrip('/')}/rerank"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": RERANK_MODEL, "query": query, "documents": documents}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Rerank Error {resp.status_code}: {resp.text}")
    data = resp.json()
    
    # Standardize result format (some APIs return 'results', some 'scores')
    if "results" in data:
        results = sorted(data["results"], key=lambda x: x.get("index", 0))
        return [r.get("relevance_score", 0.0) for r in results]
    elif "scores" in data:
        return data["scores"]
    else:
        # Fallback for simple list response
        return data if isinstance(data, list) else []

def build_org_lookup(path):
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    mp = {}
    for it in arr:
        sha1 = it.get("sha1")
        name = it.get("name")
        if sha1 and name:
            mp[sha1.lower()] = name
    return mp


def main():
    print("=== Loading Chunks ===")
    chunk_texts, chunk_meta = [], []
    chunk_store = {}
    docid_to_sha1 = {}
    
    for obj in load_jsonl(PATH_INDEX_CHUNKS):
        did, sha1 = obj["doc_id"], obj.get("sha1", "")
        cid = int(obj["chunk_id"])
        txt = obj["text"]
        chunk_texts.append(txt)
        chunk_meta.append((did, cid, sha1))
        chunk_store[(did, cid)] = txt
        if did not in docid_to_sha1 and sha1:
            docid_to_sha1[did] = sha1
    print(f"Total Chunks: {len(chunk_texts)}")

    print("=== Building BM25 ===")
    bm25, tokenized_corpus = build_bm25(chunk_texts)

    print("=== Building Dense Index ===")

    if os.path.exists(PATH_FAISS_INDEX):
        print(f"Loading FAISS from {PATH_FAISS_INDEX} ...")
        index_faiss = faiss.read_index(PATH_FAISS_INDEX)
    else:
        embeds = embed_texts(chunk_texts)
        index_faiss = build_faiss_index(embeds)
        faiss.write_index(index_faiss, PATH_FAISS_INDEX)

    print("=== Loading ORG Map ===")
    sha1_to_org = build_org_lookup(PATH_ORG_UNION)

    print("=== Processing Benchmark Docs ===")
    bench_docs = list(load_jsonl(PATH_BENCH_DOCS))
    
    out_map = {}
    out_map_all = {}
    
    for idx, qdoc in enumerate(bench_docs, 1):
        q_doc_id = qdoc["doc_id"]
        q_raw_text = qdoc.get("flat_text") or qdoc.get("text") or ""
        logging.info(f"Query {idx}/{len(bench_docs)}: {q_doc_id}")

        # 1. Sparse Search
        bm25_res = bm25_search_doclevel(bm25, tokenized_corpus, chunk_meta, q_raw_text, 
                                        TOPK_CHUNK_SPARSE, TOPK_DOC_CANDIDATE)
        
        # 2. Dense Search
        dense_res = dense_search_doclevel(index_faiss, q_raw_text, chunk_texts, chunk_meta,
                                          TOPK_CHUNK_DENSE, TOPK_DOC_CANDIDATE)
        
        # 3. Fusion
        fused = rrf_fuse(bm25_res, dense_res, k=RRF_K, keep=TOPK_DOC_CANDIDATE)
        
        # 4. Rerank
        rerank_cands = []
        for item in fused:
            doc_id = item["doc_id"]
            # Retrieve actual texts for top chunks
            docs_text = []
            cids = []
            for sc, cid in item.get("_top_chunks", []):
                key = (doc_id, cid)
                if key in chunk_store:
                    docs_text.append(chunk_store[key])
                    cids.append(cid)
            
            if not docs_text: continue
            
            try:
                # Build safe query for reranker
                safe_q = build_query_text(q_raw_text, max_imports=30, max_iocs=20)
                safe_q = truncate_for_model(safe_q, 6000)
                
                scores = call_rerank(safe_q, docs_text)
                
                # MaxP Strategy
                best_score = max(scores)
                best_chunk_idx = int(np.argmax(scores))
                rerank_cands.append({
                    "doc_id": doc_id,
                    "score": best_score,
                    "best_chunk_id": cids[best_chunk_idx]
                })
            except Exception as e:
                logger.error(f"Rerank failed for {doc_id}: {e}")

        # 5. Final Top 20
        rerank_cands.sort(key=lambda x: x["score"], reverse=True)
        top20 = rerank_cands[:TOPK_FINAL]

        # Output formatting
        entry_all = {}
        for rank, it in enumerate(top20, 1):
            s = docid_to_sha1.get(it["doc_id"], "").lower()
            org = sha1_to_org.get(s, "Unknown")
            entry_all[f"demo{rank}"] = s
            entry_all[f"demo{rank}_output"] = org
        out_map_all[q_doc_id] = entry_all
        
        # Top 2 specific
        entry_2 = {}
        if len(top20) >= 1:
            entry_2["demo1"] = entry_all["demo1"]
            entry_2["demo1_output"] = entry_all["demo1_output"]
        if len(top20) >= 2:
            entry_2["demo2"] = entry_all["demo2"]
            entry_2["demo2_output"] = entry_all["demo2_output"]
        out_map[q_doc_id] = entry_2

    with open(PATH_OUTPUT_TOP2, "w", encoding="utf-8") as f:
        json.dump(out_map, f, ensure_ascii=False, indent=2)
    with open(PATH_OUTPUT_TOP20, "w", encoding="utf-8") as f:
        json.dump(out_map_all, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
