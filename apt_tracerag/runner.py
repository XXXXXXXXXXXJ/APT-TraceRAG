from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .document_pipeline import DocumentOutputs, DocumentPipeline
from .paths import ProjectPaths
from .vector_kg_pipeline import VectorKGOutputs, VectorKGPipeline


@dataclass(frozen=True)
class PipelineResult:
    document: DocumentOutputs
    vector_kg: VectorKGOutputs


class APTTraceRAGRunner:
    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or ProjectPaths.discover()
        self.document_pipeline = DocumentPipeline(self.paths)
        self.vector_kg_pipeline = VectorKGPipeline(self.paths)

    def run(
        self,
        document_input_dir: Path,
        sample_root_dir: Path,
        benchmark_csv: Path,
        r_index_csv: Path,
        org_map: Path | None = None,
        flat_docs_output: Path | None = None,
        chunks_output: Path | None = None,
        doc_top2_output: Path | None = None,
        doc_top20_output: Path | None = None,
        faiss_index_output: Path | None = None,
        vector_store_dir: Path | None = None,
        sample_attribution_output: Path | None = None,
        commonality_dir: Path | None = None,
        tuning_data_output: Path | None = None,
        final_rag_output: Path | None = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "12345678",
    ) -> PipelineResult:
        executable_dir = self.paths.executable_dir
        vector_kg_dir = self.paths.vector_kg_dir

        document_outputs = self.document_pipeline.run(
            input_dir=Path(document_input_dir),
            flat_docs=Path(flat_docs_output or (executable_dir / "Benchmark_1000_flat_docs_all.jsonl")),
            chunks=Path(chunks_output or (executable_dir / "Benchmark_1000_chunks_all.jsonl")),
            top2_output=Path(doc_top2_output or (executable_dir / "Retrieval-Augmented-Few-shot_all.json")),
            top20_output=Path(doc_top20_output or (executable_dir / "Retrieval-Augmented-Few-shot_all_all.json")),
            faiss_index=Path(faiss_index_output or (executable_dir / "chunk_vectors_all.faiss")),
            index_docs=executable_dir / "R_Index_flat_docs_all.jsonl",
            index_chunks=executable_dir / "R_Index_chunks_all.jsonl",
            org_map=Path(org_map or self.paths.org_map_path),
        )

        vector_outputs = self.vector_kg_pipeline.run(
            sample_root_dir=Path(sample_root_dir),
            benchmark_csv=Path(benchmark_csv),
            vector_store_dir=Path(vector_store_dir or (executable_dir / "vector_store_benchmark")),
            sample_attribution_output=Path(sample_attribution_output or (vector_kg_dir / "sample_attribution_top20.json")),
            commonality_dir=Path(commonality_dir or (vector_kg_dir / "commonality")),
            tuning_data_output=Path(tuning_data_output or (vector_kg_dir / "tuning_data.json")),
            final_rag_output=Path(final_rag_output or (vector_kg_dir / "final_RAG.json")),
            doc_top20_json=document_outputs.top20,
            r_index_csv=Path(r_index_csv),
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )

        return PipelineResult(document=document_outputs, vector_kg=vector_outputs)
