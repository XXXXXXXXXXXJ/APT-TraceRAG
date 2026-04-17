from __future__ import annotations

import configparser
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator, Sequence


@contextmanager
def pushd(target_dir: Path) -> Iterator[None]:
    previous_dir = Path.cwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


@contextmanager
def prepend_sys_path(paths: Iterable[Path]) -> Iterator[None]:
    injected = [str(Path(path).resolve()) for path in paths]
    previous_sys_path = list(sys.path)
    for path in reversed(injected):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = previous_sys_path


@contextmanager
def temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    previous_argv = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = previous_argv


def load_module(module_name: str, module_path: Path, extra_sys_paths: Iterable[Path] | None = None) -> ModuleType:
    module_path = Path(module_path).resolve()
    extra_sys_paths = list(extra_sys_paths or [])
    with prepend_sys_path(extra_sys_paths):
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def update_ini_values(config_path: Path, updates: dict[tuple[str, str], str]) -> None:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    for (section, option), value in updates.items():
        if section not in parser:
            parser[section] = {}
        parser[section][option] = value
    with open(config_path, "w", encoding="utf-8") as handle:
        parser.write(handle)
