# models/baseline_registry.py
"""Baseline name registry and paper-reporting provenance.

This file exists to prevent a serious paper-writing mistake: an internal
implementation must not be reported as an official/reproduced SOTA baseline.

Definitions used here:
  - paper_safe: OK to report with the name below in a paper table.
  - internal: project-local diagnostic/control baseline only; do NOT report as
    an official method such as PatchTST/FourierGNN.
  - deprecated_alias: kept only so old checkpoints and old commands still load.

When adding an external SOTA baseline, add it only after either:
  (1) importing/adapting the authors' official code, or
  (2) faithfully re-implementing the method from the paper and documenting the
      source paper, commit/repo, and allowed modifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import warnings


@dataclass(frozen=True)
class BaselineMeta:
    name: str
    report_name: str
    category: str  # paper_safe / internal / pretrain_only / alias
    description: str
    paper_usage: str
    canonical: str | None = None


BASELINE_REGISTRY: Dict[str, BaselineMeta] = {
    # Paper-safe generic baselines implemented inside this project.
    "resnet1d": BaselineMeta(
        name="resnet1d",
        report_name="ResNet-1D",
        category="paper_safe",
        description="Standard supervised 1D residual CNN baseline implemented in this project.",
        paper_usage="Can be reported as a generic ResNet-1D baseline, not as an official reproduction of a specific ECG ResNet paper unless separately documented.",
    ),
    "transformer1d": BaselineMeta(
        name="transformer1d",
        report_name="Transformer-1D",
        category="paper_safe",
        description="Standard supervised Transformer encoder over ECG patches implemented in this project.",
        paper_usage="Can be reported as a generic Transformer-1D baseline, not as Medformer/PatchTST/iTransformer.",
    ),

    # Internal controls. Kept for engineering comparisons, not official SOTA tables.
    "internal_fourier_graph1d": BaselineMeta(
        name="internal_fourier_graph1d",
        report_name="Internal Fourier-Graph Control",
        category="internal",
        description="Project-local Fourier-domain lead-mixing control model.",
        paper_usage="Do not report as FourierGNN. Use only as an internal control unless rewritten from official/faithful FourierGNN code.",
    ),
    "internal_patch_transformer1d": BaselineMeta(
        name="internal_patch_transformer1d",
        report_name="Internal Patch-Transformer Control",
        category="internal",
        description="Project-local channel-independent patch Transformer control model.",
        paper_usage="Do not report as PatchTST. Use only as an internal control unless official/faithful PatchTST is integrated.",
    ),


    # External official/public-source baselines. These names construct thin ECG
    # adapters around checked-out public repositories. They are paper-eligible
    # only when SOURCE.md records the repo/commit and the repository is present.
    "official_medformer": BaselineMeta(
        name="official_medformer",
        report_name="Medformer",
        category="external_official",
        description="Thin ECG adapter around the public Medformer implementation.",
        paper_usage="Can be reported as Medformer only after documenting the public repo/commit and confirming only input/output/loss adaptation was made.",
    ),
    "official_patchtst": BaselineMeta(
        name="official_patchtst",
        report_name="PatchTST",
        category="external_official",
        description="Thin ECG adapter around the public official PatchTST implementation.",
        paper_usage="Can be reported as PatchTST only after documenting the public repo/commit and classification-head adaptation.",
    ),
    "official_itransformer": BaselineMeta(
        name="official_itransformer",
        report_name="iTransformer",
        category="external_official",
        description="Thin ECG adapter around the public official iTransformer implementation.",
        paper_usage="Can be reported as iTransformer only after documenting the public repo/commit and classification-head adaptation.",
    ),
    "official_fedformer": BaselineMeta(
        name="official_fedformer",
        report_name="FEDformer",
        category="external_official",
        description="Thin ECG adapter around the public FEDformer implementation.",
        paper_usage="Can be reported as FEDformer only after documenting the public repo/commit and classification-head adaptation.",
    ),

    "official_fouriergnn": BaselineMeta(
        name="official_fouriergnn",
        report_name="FourierGNN",
        category="external_official",
        description="Thin ECG adapter around the public official FourierGNN implementation.",
        paper_usage="Can be reported as FourierGNN only after documenting the public repo/commit and classification-head adaptation.",
    ),
    "official_medgnn": BaselineMeta(
        name="official_medgnn",
        report_name="MedGNN",
        category="external_official",
        description="Thin ECG adapter around the public official MedGNN implementation.",
        paper_usage="Can be reported as MedGNN only after documenting the public repo/commit and classification-head adaptation.",
    ),

    "tgllnet": BaselineMeta(
        name="tgllnet",
        report_name="TGLLNet",
        category="paper_safe",
        description="Self-contained implementation of TGLLNet: task-guided learnable lead-correlation graph network for 12-lead ECG classification.",
        paper_usage="Can be reported as a TGLLNet reimplementation baseline. Document that the released TGLLnet archive lacked the core Mymodel.py, so the RPG/TCC/learnable-adjacency modules were implemented from the paper description.",
    ),

    # Pretraining-only baseline. Not a direct supervised classifier.
    "simclr1d": BaselineMeta(
        name="simclr1d",
        report_name="SimCLR-1D pretraining encoder",
        category="pretrain_only",
        description="Self-supervised contrastive encoder; requires a separate linear-probing/fine-tuning protocol for a fair classification baseline.",
        paper_usage="Do not put direct BCE classification scores into the main supervised baseline table without linear probing/fine-tuning protocol.",
    ),

    # Backward-compatible aliases from old code/checkpoints. They are intentionally
    # not paper-safe because the names look like official methods.
    "fouriergnn1d": BaselineMeta(
        name="fouriergnn1d",
        report_name="DEPRECATED alias for internal_fourier_graph1d",
        category="alias",
        description="Backward-compatible alias; not official FourierGNN.",
        paper_usage="Deprecated. Use internal_fourier_graph1d in commands, and do not report as FourierGNN.",
        canonical="internal_fourier_graph1d",
    ),
    "patchtst1d": BaselineMeta(
        name="patchtst1d",
        report_name="DEPRECATED alias for internal_patch_transformer1d",
        category="alias",
        description="Backward-compatible alias; not official PatchTST.",
        paper_usage="Deprecated. Use internal_patch_transformer1d in commands, and do not report as PatchTST.",
        canonical="internal_patch_transformer1d",
    ),
}


def canonical_name(name: str) -> str:
    meta = BASELINE_REGISTRY.get(name)
    if meta is not None and meta.category == "alias" and meta.canonical:
        return meta.canonical
    return name


def warn_if_not_paper_safe(name: str) -> None:
    meta = BASELINE_REGISTRY.get(name)
    if meta is None:
        return
    if meta.category == "alias":
        warnings.warn(
            f"Baseline name '{name}' is a deprecated alias for '{meta.canonical}'. "
            f"It is NOT an official {name.replace('1d','')} implementation. {meta.paper_usage}",
            stacklevel=3,
        )
    elif meta.category in {"internal", "pretrain_only"}:
        warnings.warn(
            f"Baseline '{name}' is marked as {meta.category}. {meta.paper_usage}",
            stacklevel=3,
        )


def is_baseline_name(name: str) -> bool:
    return name in BASELINE_REGISTRY


def implemented_baseline_names(include_aliases: bool = True) -> List[str]:
    names = []
    for name, meta in BASELINE_REGISTRY.items():
        if include_aliases or meta.category != "alias":
            names.append(name)
    return sorted(names)


def paper_safe_baseline_names() -> List[str]:
    return sorted([n for n, m in BASELINE_REGISTRY.items() if m.category == "paper_safe"])


def print_baseline_table(names: Iterable[str] | None = None) -> None:
    if names is None:
        names = implemented_baseline_names(include_aliases=True)
    print("\nImplemented baseline registry:")
    print("-" * 96)
    print(f"{'name':30s} {'category':14s} {'report_name':30s} paper usage")
    print("-" * 96)
    for name in names:
        meta = BASELINE_REGISTRY.get(name)
        if meta is None:
            continue
        print(f"{name:30s} {meta.category:14s} {meta.report_name:30s} {meta.paper_usage}")
    print("-" * 96)
