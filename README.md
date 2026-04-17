# APT-TraceRAG

> **Short abstract.** APT attribution in practice often begins with noisy, semi-structured first-hand malware evidence rather than curated CTI prose. This repository accompanies our paper on **APT-TrailBench**, a long-tail benchmark built from hybrid standardized malware reports, and **APT-TraceRAG**, a graph-enhanced dual-view retrieval framework for evidence-grounded analyst lead generation. Under a known-group, forced top-1 protocol, the framework retrieves historical exemplars, aggregates graph-backed feature concentration signals, and returns inspectable attribution leads for analyst review.

APT-TraceRAG is the public research codebase accompanying our study of evidence-grounded APT attribution assistance from standardized first-hand malware reports. The repository releases the code path, benchmark scripts, prompts, split definitions, and supporting artifacts used for the paper's reproducible evaluation setting.

## Overview

Our study focuses on a deliberately narrow and operational benchmark setting: given a **hybrid standardized malware report**, a method must predict **one most likely known group** under a shared historical label space and, when possible, surface supporting evidence. We position this task as **analyst lead generation**, not autonomous accusation.

The repository accompanies two paper artifacts:

- **APT-TrailBench**: a long-tail benchmark with 13,206 samples from 281 APT groups.
- **APT-TraceRAG**: a graph-enhanced dual-view retrieval framework that combines behavioral semantics, technical fingerprints, and historical context to produce inspectable analyst-facing attribution leads.

## What This Repository Provides

- the unified end-to-end launcher used for the public APT-TraceRAG pipeline,
- preprocessing scripts for constructing hybrid standardized malware reports from VirusTotal file and behavior reports,
- document retrieval code for historical exemplar retrieval,
- vector and graph retrieval code for feature-grounded evidence construction,
- fusion and reranking scripts for final prediction generation,
- benchmark-related scripts, prompts, and reproducibility artifacts used in the reported protocol.

## Research Scope

This repository is designed for the paper's benchmarkable proxy of attribution assistance:

- **Input**: a hybrid standardized malware report derived from semi-structured sandbox outputs plus lightweight static features.
- **Output**: one predicted APT group from a known historical label space.
- **Protocol**: known-group, forced top-1 attribution.
- **Goal**: evidence-backed analyst lead generation over a noisy search space.

Accordingly, the repository should be understood as a research artifact for **evidence-grounded attribution assistance** over APT-TrailBench rather than a production deployment of automated attribution.

## Preprocessing

The `Preprocessing/` directory contains the script collection used to preprocess VirusTotal reports and assemble the benchmark-facing report artifacts.

In the paper, this stage corresponds to **hybrid standardized malware report construction**. Starting from VirusTotal file and behavior reports queried by executable hash, the preprocessing pipeline converts noisy first-hand malware evidence into a compact, attribution-oriented representation that can support both classical baselines and APT-TraceRAG. The construction is designed to reduce context length and suppress raw-field noise while preserving attribution-relevant evidence.

At a high level, the preprocessing stage supports three transformations described in the paper:

- **File-report sanitization**: removes raw hashes, explicit APT-group label fields, and analyst-curated ATT&CK abstractions while retaining attribution-relevant build artifacts and other semantically dense report fields.
- **Dynamic-behavior refinement**: extracts and normalizes behavior-report content into a smaller schema better suited to retrieval and downstream attribution modeling.
- **Lightweight local static supplementation**: augments cloud-native reports with lightweight static residues, including FLOSS- and YARA-derived signals that may not be preserved in the original cloud report view.

The same directory also contains scripts related to **auxiliary family label construction**. Following the paper, normalized malware-family metadata derived from tools such as ClarAVy and TagClass are treated as auxiliary report fields rather than as ground-truth APT-group labels. Raw antivirus-engine detections are removed from the final standardized reports, while the normalized family fields are retained as realistic but potentially shortcut-prone metadata for later analysis.

Current scripts under `Preprocessing/` include:

- `Preprocessing/vt_reports_del.py`: report cleaning and field-pruning logic for VirusTotal-derived artifacts.
- `Preprocessing/extract_behaviour.py`: extraction and normalization utilities for dynamic behavior reports.
- `Preprocessing/clean_mal_tags.py`: malware-family tag cleaning and normalization utilities.
- `Preprocessing/combine_data.py`: helper script for merging preprocessed artifacts into benchmark-facing inputs.

## Repository Layout

- `APT-TraceRAG.py`: unified launcher for the end-to-end pipeline.
- `Preprocessing/`: scripts for VirusTotal report preprocessing, auxiliary family-label cleaning, and benchmark-facing artifact assembly.
- `apt_tracerag/`: orchestration wrappers around the document and vector/KG stages.
- `Document/`: hybrid report flattening plus retrieval and reranking code.
- `Vector-KG/`: feature extraction, vector retrieval, graph retrieval, fusion, and hyperparameter tuning.
- `Vector-KG/samples/`: example CSVs and sample vector-store artifacts used by the pipeline.

A minimal code-path view is:

```text
APT-TraceRAG/
|- APT-TraceRAG.py                 # end-to-end launcher
|- Preprocessing/                 # VT report preprocessing and benchmark assembly
|- apt_tracerag/                  # orchestration wrappers
|- Document/
|  |- vt_flatten.py               # hybrid report flattening
|  `- retrieval_pipeline.py       # document retrieval and reranking
`- Vector-KG/
   |- featuresPrepare.py          # vector-store construction
   |- sampleAttribution.py        # vector retrieval for candidate groups
   |- graph_retriever.py          # graph-backed evidence retrieval
   |- graph_reranker_data_dump.py # fusion data preparation
   `- tune_hyperparameters.py     # final reranking and output generation
```

## Open Science

To support reproducibility and follow-on research, we adopt an open-science release strategy within the limits of platform policy and malware safety.

### Data availability

We will release **APT-TrailBench** metadata upon publication in a **hydration-based** form. To comply with VirusTotal policy, the release includes:

- sample hashes,
- label mappings,
- the hybrid standardized malware report field definition,
- split metadata, and
- full evaluation outputs.

We do **not** redistribute raw report contents or platform-derived feature values. Researchers can reconstruct the benchmark through their own authorized access using the released hashes and schema, subject to the continued availability of the underlying VirusTotal reports.

### Artifact availability

During double-blind review, we provide an anonymous artifact repository containing the source code for **APT-TraceRAG**, ThreatKG construction, retrieval, graph reasoning, benchmark scripts, prompts, split definitions, and reproducibility artifacts.

### Archival release

To preserve anonymity during review, we omit permanent DOI-based or share-token-based archival links from the submission. The camera-ready version will replace the anonymous review link with a permanent archival link for the released schema and result artifacts.

### Reproducibility

Our experiments use publicly available open-source LLMs and standard libraries. The paper and appendices provide prompts, hyperparameters, and implementation details needed to reproduce the reported protocol and code path, while exact benchmark re-hydration remains contingent on authorized platform access.

## Expected Workspace Layout

The current launcher resolves several defaults relative to the repository's parent directory. By default it expects a workspace of the following form:

```text
<workspace>/
|- APT-TraceRAG/
|  |- APT-TraceRAG.py
|  |- Document/
|  |- Vector-KG/
|  `- apt_tracerag/
|- Executable/
|  |- Benchmark_1000_all/
|  |- APT-ClarityExec_1000/
|  |- BenchmarkSet.csv
|  |- R_IndexSet.csv
|  |- R_Index_flat_docs_all.jsonl
|  `- R_Index_chunks_all.jsonl
`- all_ORGs_union.json
```

If your local layout differs, pass the corresponding paths explicitly on the command line.

During execution, the unified launcher also updates `Vector-KG/config/projectconfig.ini` so that the vector/KG stage points to the active `sample_root_dir` and `benchmark_csv`.

## Environment and Dependencies

### Core requirements

- Python 3.10+
- a local or remote **OpenAI-compatible API endpoint** for embedding and reranking requests,
- a running **Neo4j** instance for graph retrieval and graph-backed evidence aggregation.

### Python packages

Install the dependencies required by the stages you run. In our environment, the code depends on packages such as:

```bash
openai numpy pandas tqdm requests ujson faiss-cpu rank-bm25 neo4j transformers sentence-transformers torch scikit-learn scipy matplotlib seaborn nltk dateparser textstat bitcoinlib
```

Depending on your feature-generation setup, you may also need additional packages referenced under `Vector-KG/featgenerator/`.

### Service assumptions in the current code

The present implementation assumes:

- `OPENAI_API_KEY` is set in the environment,
- an OpenAI-compatible endpoint is available at `http://127.0.0.1:9997/v1`,
- the embedding model name is `bge-m3`,
- the reranker model name is `bge-reranker-v2-m3`,
- Neo4j is reachable at `bolt://localhost:7687` unless overridden.

These defaults may need to be adapted for other deployment environments.

## Running the Pipeline

### Quick start

1. Create and activate a Python environment.
2. Install the dependencies needed by the stages you intend to run.
3. Prepare the hydrated benchmark inputs and the reference index files.
4. Start the embedding/reranking service expected by `Document/retrieval_pipeline.py`.
5. Start Neo4j and load the graph resources needed by the graph retrieval stage.
6. Run the unified launcher:

```bash
python APT-TraceRAG.py
```

To list all runtime options:

```bash
python APT-TraceRAG.py --help
```

### Example invocation

```bash
python APT-TraceRAG.py --document-input-dir <path-to-benchmark-vt-jsons> --sample-root-dir <path-to-hydrated-sample-features> --benchmark-csv <path-to-BenchmarkSet.csv> --r-index-csv <path-to-R_IndexSet.csv> --org-map <path-to-all_ORGs_union.json> --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password <password>
```

## Pipeline Outputs

A successful end-to-end run typically produces the following outputs.

### Document retrieval stage

- flattened benchmark documents,
- chunked benchmark documents,
- top-2 document retrieval results,
- top-20 document retrieval results,
- a FAISS index for chunk retrieval.

### Vector/KG stage

- vector-store artifacts for benchmark samples,
- vector-based sample attribution results,
- graph commonality outputs,
- fusion tuning data,
- final fused predictions in `final_RAG.json`.

The unified launcher prints the key output paths after execution. Under the default workspace layout, the main outputs are typically written to:

- `Executable/Retrieval-Augmented-Few-shot_all.json`
- `Executable/Retrieval-Augmented-Few-shot_all_all.json`
- `Executable/vector_store_benchmark/`
- `Vector-KG/sample_attribution_top20.json`
- `Vector-KG/commonality/`
- `Vector-KG/tuning_data.json`
- `Vector-KG/final_RAG.json`

## Notes and Limitations

- This is a research codebase intended for **reproducible experimentation**, not a production attribution system.
- The repository targets the paper's **known-group, forced top-1** benchmark protocol.
- Exact benchmark re-hydration depends on authorized access to the underlying platform and on the continued availability of the underlying reports.
- Some scripts retain legacy path assumptions and may require environment-specific path or service adjustments.
- Because attribution is a high-risk task, the system should be used as **analyst assistance** rather than as an autonomous decision maker.

## Citation

If you use APT-TrailBench or APT-TraceRAG in academic work, please cite the accompanying paper.

## License

Please add the final project license before public release.
