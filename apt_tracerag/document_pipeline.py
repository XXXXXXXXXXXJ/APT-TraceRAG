from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .legacy import load_module, pushd
from .paths import ProjectPaths


@dataclass(frozen=True)
class DocumentOutputs:
    flat_docs: Path
    chunks: Path
    top2: Path
    top20: Path
    faiss_index: Path


class DocumentPipeline:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def run(
        self,
        input_dir: Path,
        flat_docs: Path,
        chunks: Path,
        top2_output: Path,
        top20_output: Path,
        faiss_index: Path,
        index_docs: Path,
        index_chunks: Path,
        org_map: Path,
    ) -> DocumentOutputs:
        input_dir = Path(input_dir).resolve()
        flat_docs = Path(flat_docs).resolve()
        chunks = Path(chunks).resolve()
        top2_output = Path(top2_output).resolve()
        top20_output = Path(top20_output).resolve()
        faiss_index = Path(faiss_index).resolve()
        index_docs = Path(index_docs).resolve()
        index_chunks = Path(index_chunks).resolve()
        org_map = Path(org_map).resolve()

        flat_docs.parent.mkdir(parents=True, exist_ok=True)
        chunks.parent.mkdir(parents=True, exist_ok=True)
        top2_output.parent.mkdir(parents=True, exist_ok=True)
        top20_output.parent.mkdir(parents=True, exist_ok=True)
        faiss_index.parent.mkdir(parents=True, exist_ok=True)

        vt_flatten = load_module(
            "_apttracerag_vt_flatten",
            self.paths.document_dir / "vt_flatten.py",
            extra_sys_paths=[self.paths.document_dir],
        )
        flatten_args = argparse.Namespace(
            input_dir=str(input_dir),
            output_flat_docs=str(flat_docs),
            output_chunks=str(chunks),
            org_map=str(org_map),
            default_org="unknown",
            max_tokens=2048,
            overlap=256,
        )
        with pushd(self.paths.experiments_dir):
            vt_flatten.cmd_flatten(flatten_args)

        retrieval_pipeline = load_module(
            "_apttracerag_retrieval_pipeline",
            self.paths.document_dir / "retrieval_pipeline.py",
            extra_sys_paths=[self.paths.document_dir],
        )
        retrieval_pipeline.PATH_INDEX_DOCS = str(index_docs)
        retrieval_pipeline.PATH_INDEX_CHUNKS = str(index_chunks)
        retrieval_pipeline.PATH_BENCH_DOCS = str(flat_docs)
        retrieval_pipeline.PATH_BENCH_CHUNKS = str(chunks)
        retrieval_pipeline.PATH_ORG_UNION = str(org_map)
        retrieval_pipeline.PATH_OUTPUT_TOP2 = str(top2_output)
        retrieval_pipeline.PATH_OUTPUT_TOP20 = str(top20_output)
        retrieval_pipeline.PATH_FAISS_INDEX = str(faiss_index)

        with pushd(self.paths.experiments_dir):
            retrieval_pipeline.main()

        return DocumentOutputs(
            flat_docs=flat_docs,
            chunks=chunks,
            top2=top2_output,
            top20=top20_output,
            faiss_index=faiss_index,
        )
