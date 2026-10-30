# -*- coding: utf-8 -*-
"""Utilities for loading external official/public baseline code.

Why external loading instead of copying simplified code?
-------------------------------------------------------
For paper-grade baselines, we should preserve the source implementation's
core backbone as much as possible. This module loads a model file from a
checked-out public repository and exposes it through a thin ECG wrapper.

Repository discovery order for a baseline named, e.g., ``patchtst``:
  1. ECG_BASELINE_PATCHTST_ROOT=/absolute/path/to/repo-or-subdir
  2. OFFICIAL_BASELINES_ROOT=/path containing repo folders
  3. ./external_baselines/<candidate repo names>

No fallback implementation is used. Missing sources raise ExternalRepoError.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence


class ExternalRepoError(RuntimeError):
    """Raised when an official/public baseline repository is unavailable."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "external_baselines"


BASELINE_SOURCE_SPECS = {
    "medformer": {
        "env": "ECG_BASELINE_MEDFORMER_ROOT",
        "repo_names": ["Medformer", "medformer", "DL4mHealth_Medformer"],
        "import_root_candidates": ["."],
        "model_file_candidates": ["models/Medformer.py"],
        "url": "https://github.com/DL4mHealth/Medformer",
        "paper_name": "Medformer",
    },
    "patchtst": {
        "env": "ECG_BASELINE_PATCHTST_ROOT",
        "repo_names": ["PatchTST", "patchtst", "yuqinie98_PatchTST"],
        "import_root_candidates": ["PatchTST_supervised", "."],
        "model_file_candidates": [
            "PatchTST_supervised/models/PatchTST.py",
            "models/PatchTST.py",
        ],
        "url": "https://github.com/yuqinie98/PatchTST",
        "paper_name": "PatchTST",
    },
    "itransformer": {
        "env": "ECG_BASELINE_ITRANSFORMER_ROOT",
        "repo_names": ["iTransformer", "itransformer", "thuml_iTransformer"],
        "import_root_candidates": ["."],
        "model_file_candidates": ["model/iTransformer.py", "models/iTransformer.py"],
        "url": "https://github.com/thuml/iTransformer",
        "paper_name": "iTransformer",
    },
    "fedformer": {
        "env": "ECG_BASELINE_FEDFORMER_ROOT",
        "repo_names": ["FEDformer", "fedformer", "MAZiqing_FEDformer"],
        "import_root_candidates": ["."],
        "model_file_candidates": ["models/FEDformer.py"],
        "url": "https://github.com/MAZiqing/FEDformer",
        "paper_name": "FEDformer",
    },

    "fouriergnn": {
        "env": "ECG_BASELINE_FOURIERGNN_ROOT",
        "repo_names": ["FourierGNN", "fouriergnn", "aikunyi_FourierGNN"],
        "import_root_candidates": ["."],
        "model_file_candidates": ["model/FourierGNN.py"],
        "url": "https://github.com/aikunyi/FourierGNN",
        "paper_name": "FourierGNN",
    },
    "medgnn": {
        "env": "ECG_BASELINE_MEDGNN_ROOT",
        "repo_names": ["MedGNN", "medgnn", "aikunyi_MedGNN"],
        "import_root_candidates": ["."],
        "model_file_candidates": ["models/MedGNN.py"],
        "url": "https://github.com/aikunyi/MedGNN",
        "paper_name": "MedGNN",
    },
}


def _existing_dirs(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_dir()]


def official_external_root() -> Path:
    return Path(os.environ.get("OFFICIAL_BASELINES_ROOT", DEFAULT_EXTERNAL_ROOT)).expanduser().resolve()


def candidate_repo_roots(name: str) -> list[Path]:
    if name not in BASELINE_SOURCE_SPECS:
        raise KeyError(f"Unknown external baseline source spec: {name}")
    spec = BASELINE_SOURCE_SPECS[name]
    candidates: list[Path] = []
    env_value = os.environ.get(spec["env"])
    if env_value:
        candidates.append(Path(env_value).expanduser().resolve())
    root = official_external_root()
    candidates.extend([(root / repo_name).resolve() for repo_name in spec["repo_names"]])
    return candidates


def resolve_repo_root(name: str) -> Path:
    candidates = candidate_repo_roots(name)
    hits = _existing_dirs(candidates)
    if hits:
        return hits[0]
    spec = BASELINE_SOURCE_SPECS[name]
    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise ExternalRepoError(
        f"Missing public source repository for {spec['paper_name']}.\n"
        f"Expected one of:\n{candidate_text}\n\n"
        f"Quick fix:\n"
        f"  mkdir -p {official_external_root()}\n"
        f"  cd {official_external_root()}\n"
        f"  git clone {spec['url']} {spec['repo_names'][0]}\n\n"
        f"Or set {spec['env']}=/absolute/path/to/the/repository."
    )


def _find_existing_child(root: Path, candidates: Sequence[str], expected_kind: str) -> Path:
    for rel in candidates:
        p = (root / rel).resolve()
        if expected_kind == "file" and p.exists() and p.is_file():
            return p
        if expected_kind == "dir" and p.exists() and p.is_dir():
            return p
    raise ExternalRepoError(
        f"Repository found at {root}, but expected {expected_kind} not found. Tried:\n"
        + "\n".join(f"  - {root / rel}" for rel in candidates)
    )


def resolve_model_file(name: str, root: Path | None = None) -> Path:
    root = resolve_repo_root(name) if root is None else root
    return _find_existing_child(root, BASELINE_SOURCE_SPECS[name]["model_file_candidates"], "file")


def resolve_import_root(name: str, root: Path | None = None) -> Path:
    root = resolve_repo_root(name) if root is None else root
    return _find_existing_child(root, BASELINE_SOURCE_SPECS[name]["import_root_candidates"], "dir")


EXTERNAL_TOP_LEVEL_MODULES = (
    # Common package names used by time-series repos.  They often collide with
    # this project (`utils`, `models`) or with each other (`layers`).
    "layers",
    "utils",
    "models",
    "data_provider",
    "exp",
)


def _module_matches_top_level(mod_name: str, top_name: str) -> bool:
    return mod_name == top_name or mod_name.startswith(top_name + ".")


def purge_external_top_level_modules(top_levels: Sequence[str] = EXTERNAL_TOP_LEVEL_MODULES) -> None:
    """Remove cached top-level modules before importing an external repo.

    Many public time-series repositories use absolute imports such as
    `from layers...` or `from utils.masking...`.  This project also has a
    top-level `utils` package, so Python may otherwise resolve
    `utils.masking` to the project package and fail.

    We purge these names immediately before importing a public repository and
    temporarily put that repository at the front of `sys.path`.  Project modules
    are restored after the import by `external_import_context`.
    """
    for mod_name in list(sys.modules.keys()):
        if any(_module_matches_top_level(mod_name, top) for top in top_levels):
            sys.modules.pop(mod_name, None)


@contextmanager
def prepend_sys_path(path: Path):
    path_str = str(path)
    old_path = list(sys.path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path[:] = old_path


@contextmanager
def external_import_context(import_root: Path, top_levels: Sequence[str] = EXTERNAL_TOP_LEVEL_MODULES):
    """Import a public repository without leaking/consuming local packages.

    The key issue this fixes is Medformer importing `utils.masking`: without
    isolation, Python can see this project's `utils` package first and raise
    `ModuleNotFoundError: No module named 'utils.masking'`.
    """
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if any(_module_matches_top_level(name, top) for top in top_levels)
    }
    try:
        purge_external_top_level_modules(top_levels)
        with prepend_sys_path(import_root):
            yield
    finally:
        # Remove any external repo packages imported under collision-prone names.
        purge_external_top_level_modules(top_levels)
        # Restore project modules exactly as they were before the external import.
        sys.modules.update(saved_modules)


def load_external_module(name: str, module_label: str | None = None) -> ModuleType:
    """Load a model file from an external official/public repository.

    The returned module is isolated under a private module name such as
    `_ecg_official_medformer_model`, while its internal imports can resolve
    through the repository import root temporarily prepended to sys.path.
    """
    root = resolve_repo_root(name)
    model_file = resolve_model_file(name, root)
    import_root = resolve_import_root(name, root)
    module_name = module_label or f"_ecg_official_{name}_model"
    spec = importlib.util.spec_from_file_location(module_name, str(model_file))
    if spec is None or spec.loader is None:
        raise ExternalRepoError(f"Could not load Python module for {name}: {model_file}")
    module = importlib.util.module_from_spec(spec)
    with external_import_context(import_root):
        spec.loader.exec_module(module)
    return module


def describe_available_sources() -> list[dict[str, str]]:
    rows = []
    for name, spec in BASELINE_SOURCE_SPECS.items():
        status = "missing"
        root_text = ""
        try:
            root = resolve_repo_root(name)
            model_file = resolve_model_file(name, root)
            import_root = resolve_import_root(name, root)
            status = "ok"
            root_text = f"root={root}; model={model_file.relative_to(root)}; import_root={import_root.relative_to(root)}"
        except Exception as exc:  # diagnostic only
            root_text = str(exc).splitlines()[0]
        rows.append({
            "name": name,
            "paper_name": spec["paper_name"],
            "status": status,
            "details": root_text,
            "url": spec["url"],
        })
    return rows
