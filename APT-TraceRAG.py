from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from apt_tracerag.paths import ProjectPaths
from apt_tracerag.runner import APTTraceRAGRunner


def build_parser(defaults: ProjectPaths) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APT-TraceRAG unified launcher")
    parser.add_argument(
        "--document-input-dir",
        default=str(defaults.executable_dir / "Benchmark_1000_all"),
        help="VT JSON directory for the document retrieval stage.",
    )
    parser.add_argument(
        "--sample-root-dir",
        default=str(defaults.executable_dir / "APT-ClarityExec_1000"),
        help="Sample directory used by the vector and graph stages.",
    )
    parser.add_argument(
        "--benchmark-csv",
        default=str(defaults.executable_dir / "BenchmarkSet.csv"),
        help="Ground-truth CSV used by the benchmark pipeline.",
    )
    parser.add_argument(
        "--r-index-csv",
        default=str(defaults.executable_dir / "R_IndexSet.csv"),
        help="Reference-index CSV used by the fusion reranker.",
    )
    parser.add_argument(
        "--org-map",
        default=str(defaults.org_map_path),
        help="Organization mapping JSON for the document retrieval stage.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI for graph retrieval.",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j user name.",
    )
    parser.add_argument(
        "--neo4j-password",
        default="12345678",
        help="Neo4j password.",
    )
    parser.add_argument(
        "--doc-top20-output",
        default=str(defaults.executable_dir / "Retrieval-Augmented-Few-shot_all_all.json"),
        help="Top-20 document retrieval output JSON.",
    )
    parser.add_argument(
        "--vector-store-dir",
        default=str(defaults.executable_dir / "vector_store_benchmark"),
        help="Vector store output directory.",
    )
    parser.add_argument(
        "--commonality-dir",
        default=str(defaults.vector_kg_dir / "commonality"),
        help="Graph commonality output directory.",
    )
    parser.add_argument(
        "--final-rag-output",
        default=str(defaults.vector_kg_dir / "final_RAG.json"),
        help="Final fusion result JSON.",
    )
    return parser


def main() -> None:
    defaults = ProjectPaths.discover()
    args = build_parser(defaults).parse_args()
    runner = APTTraceRAGRunner(defaults)
    result = runner.run(
        document_input_dir=Path(args.document_input_dir),
        sample_root_dir=Path(args.sample_root_dir),
        benchmark_csv=Path(args.benchmark_csv),
        r_index_csv=Path(args.r_index_csv),
        org_map=Path(args.org_map),
        doc_top20_output=Path(args.doc_top20_output),
        vector_store_dir=Path(args.vector_store_dir),
        commonality_dir=Path(args.commonality_dir),
        final_rag_output=Path(args.final_rag_output),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )

    print("APT-TraceRAG pipeline finished.")
    print(f"Document Top20: {result.document.top20}")
    print(f"Graph commonality: {result.vector_kg.commonality_dir}")
    print(f"Final RAG: {result.vector_kg.final_rag}")


if __name__ == "__main__":
    main()
