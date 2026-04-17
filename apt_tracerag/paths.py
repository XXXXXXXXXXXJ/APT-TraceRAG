from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    apt_rag_dir: Path
    experiments_dir: Path
    document_dir: Path
    vector_kg_dir: Path
    executable_dir: Path
    org_map_path: Path
    vector_config_path: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        apt_rag_dir = Path(__file__).resolve().parents[1]
        experiments_dir = apt_rag_dir.parent
        return cls(
            apt_rag_dir=apt_rag_dir,
            experiments_dir=experiments_dir,
            document_dir=apt_rag_dir / "Document",
            vector_kg_dir=apt_rag_dir / "Vector-KG",
            executable_dir=experiments_dir / "Executable",
            org_map_path=experiments_dir / "all_ORGs_union.json",
            vector_config_path=apt_rag_dir / "Vector-KG" / "config" / "projectconfig.ini",
        )
