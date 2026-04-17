from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .legacy import load_module, pushd, temporary_argv, update_ini_values
from .paths import ProjectPaths


@dataclass(frozen=True)
class VectorKGOutputs:
    vector_store_dir: Path
    sample_attribution: Path
    commonality_dir: Path
    tuning_data: Path
    final_rag: Path


class VectorKGPipeline:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def run(
        self,
        sample_root_dir: Path,
        benchmark_csv: Path,
        vector_store_dir: Path,
        sample_attribution_output: Path,
        commonality_dir: Path,
        tuning_data_output: Path,
        final_rag_output: Path,
        doc_top20_json: Path,
        r_index_csv: Path,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ) -> VectorKGOutputs:
        sample_root_dir = Path(sample_root_dir).resolve()
        benchmark_csv = Path(benchmark_csv).resolve()
        vector_store_dir = Path(vector_store_dir).resolve()
        sample_attribution_output = Path(sample_attribution_output).resolve()
        commonality_dir = Path(commonality_dir).resolve()
        tuning_data_output = Path(tuning_data_output).resolve()
        final_rag_output = Path(final_rag_output).resolve()
        doc_top20_json = Path(doc_top20_json).resolve()
        r_index_csv = Path(r_index_csv).resolve()

        vector_store_dir.mkdir(parents=True, exist_ok=True)
        sample_attribution_output.parent.mkdir(parents=True, exist_ok=True)
        commonality_dir.mkdir(parents=True, exist_ok=True)
        tuning_data_output.parent.mkdir(parents=True, exist_ok=True)
        final_rag_output.parent.mkdir(parents=True, exist_ok=True)

        update_ini_values(
            self.paths.vector_config_path,
            {
                ("DEFAULT", "root_dir"): str(sample_root_dir),
                ("groundtruth", "adversary_mapping"): str(benchmark_csv),
            },
        )

        extra_sys_paths = [self.paths.vector_kg_dir]
        with pushd(self.paths.vector_kg_dir):
            features_prepare = load_module(
                "_apttracerag_features_prepare",
                self.paths.vector_kg_dir / "featuresPrepare.py",
                extra_sys_paths=extra_sys_paths,
            )
            features_prepare.FeaturesPreparer().run(str(vector_store_dir))

            sample_attribution = load_module(
                "_apttracerag_sample_attribution",
                self.paths.vector_kg_dir / "sampleAttribution.py",
                extra_sys_paths=extra_sys_paths,
            )
            sample_attribution.SampleAttributor(vector_store_dir=str(vector_store_dir)).run(
                input_dir=str(sample_root_dir),
                output_file=str(sample_attribution_output),
            )

            graph_retriever = load_module(
                "_apttracerag_graph_retriever",
                self.paths.vector_kg_dir / "graph_retriever.py",
                extra_sys_paths=extra_sys_paths,
            )
            with temporary_argv(
                [
                    "graph_retriever.py",
                    "--neo4j_uri",
                    neo4j_uri,
                    "--neo4j_user",
                    neo4j_user,
                    "--neo4j_password",
                    neo4j_password,
                    "--benchmark_csv",
                    str(benchmark_csv),
                    "--vs_benchmark",
                    str(vector_store_dir),
                    "--out_dir",
                    str(commonality_dir),
                ]
            ):
                graph_retriever.main()

            graph_dump = load_module(
                "_apttracerag_graph_dump",
                self.paths.vector_kg_dir / "graph_reranker_data_dump.py",
                extra_sys_paths=extra_sys_paths,
            )
            with temporary_argv(
                [
                    "graph_reranker_data_dump.py",
                    "--uri",
                    neo4j_uri,
                    "--user",
                    neo4j_user,
                    "--password",
                    neo4j_password,
                    "--benchmark_dir",
                    str(vector_store_dir),
                    "--benchmark_csv",
                    str(benchmark_csv),
                    "--doc_json",
                    str(doc_top20_json),
                    "--vec_json",
                    str(sample_attribution_output),
                    "--output_dump",
                    str(tuning_data_output),
                ]
            ):
                graph_dump.main()

            tune_hyperparameters = load_module(
                "_apttracerag_tune_hyperparameters",
                self.paths.vector_kg_dir / "tune_hyperparameters.py",
                extra_sys_paths=extra_sys_paths,
            )
            with temporary_argv(
                [
                    "tune_hyperparameters.py",
                    "--dump_file",
                    str(tuning_data_output),
                    "--benchmark_csv",
                    str(benchmark_csv),
                    "--r_index_csv",
                    str(r_index_csv),
                    "--output_json",
                    str(final_rag_output),
                    "--params_path",
                    str(self.paths.vector_kg_dir / "best_bm25_params.json"),
                ]
            ):
                tune_hyperparameters.main()

        return VectorKGOutputs(
            vector_store_dir=vector_store_dir,
            sample_attribution=sample_attribution_output,
            commonality_dir=commonality_dir,
            tuning_data=tuning_data_output,
            final_rag=final_rag_output,
        )
