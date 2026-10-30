# models/model_factory.py
"""Centralized model construction for train/eval/smoke-test scripts.

Important baseline policy
-------------------------
Only generic baselines implemented by this project are treated as paper-safe
names here (e.g. ResNet-1D, Transformer-1D). Project-local internal controls
are intentionally registered under `internal_*` names. Backward-compatible
aliases such as `patchtst1d` and `fouriergnn1d` are still accepted so old
checkpoints can be evaluated, but warnings are emitted and they must not be
reported as official PatchTST/FourierGNN baselines.
"""

from __future__ import annotations

import torch

from models.dual_model import DualFreqDGNN, OldDualFreqDGNN
from models.baselines import ResNet1D, SimCLR1D, Transformer1D, FourierGNN1D, PatchTST1D
from models.tgllnet import TGLLNetForECG
from models.official_baselines import (
    OfficialMedformerForECG,
    OfficialPatchTSTForECG,
    OfficialITransformerForECG,
    OfficialFEDformerForECG,
    OfficialFourierGNNForECG,
    OfficialMedGNNForECG,
)
from models.baseline_registry import (
    implemented_baseline_names,
    paper_safe_baseline_names,
    canonical_name,
    warn_if_not_paper_safe,
)

PAPER_SAFE_BASELINES = set(paper_safe_baseline_names())
IMPLEMENTED_BASELINES = set(implemented_baseline_names(include_aliases=True))
SUPERVISED_BASELINES = {
    "resnet1d",
    "transformer1d",
    "internal_fourier_graph1d",
    "internal_patch_transformer1d",
    # external official/public-source adapters
    "official_medformer",
    "official_patchtst",
    "official_itransformer",
    "official_fedformer",
    "official_fouriergnn",
    "official_medgnn",
    "tgllnet",
    # deprecated aliases kept for old commands/checkpoints
    "fouriergnn1d",
    "patchtst1d",
}
ALL_BASELINES = SUPERVISED_BASELINES | {"simclr1d"}


def _cfg_value(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


def _build_baseline(ablation: str, num_classes: int):
    """Build project-implemented baselines.

    The canonical names `internal_fourier_graph1d` and
    `internal_patch_transformer1d` are internal controls, not official SOTA
    implementations. The old aliases `fouriergnn1d` and `patchtst1d` map to
    those same models for checkpoint compatibility.
    """
    warn_if_not_paper_safe(ablation)
    name = canonical_name(ablation)

    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes)
    if name == "simclr1d":
        return SimCLR1D(num_classes=num_classes)
    if name == "transformer1d":
        return Transformer1D(num_classes=num_classes)
    if name == "internal_fourier_graph1d":
        return FourierGNN1D(num_classes=num_classes)
    if name == "internal_patch_transformer1d":
        return PatchTST1D(num_classes=num_classes)
    if name == "official_medformer":
        return OfficialMedformerForECG(num_classes=num_classes)
    if name == "official_patchtst":
        return OfficialPatchTSTForECG(num_classes=num_classes)
    if name == "official_itransformer":
        return OfficialITransformerForECG(num_classes=num_classes)
    if name == "official_fedformer":
        return OfficialFEDformerForECG(num_classes=num_classes)
    if name == "official_fouriergnn":
        return OfficialFourierGNNForECG(num_classes=num_classes)
    if name == "official_medgnn":
        return OfficialMedGNNForECG(num_classes=num_classes)
    if name == "tgllnet":
        return TGLLNetForECG(num_classes=num_classes)

    raise ValueError(
        f"Unknown baseline '{ablation}'. Use python -m scripts.diagnostics.smoke_test_models --list_models "
        "to view registered baselines."
    )


def build_model_by_name(args, cfg, num_classes: int, device: torch.device | str | None = None):
    """Build a model from the shared CLI/config namespace.

    Expected args fields: ablation, task, aux_multilabel_lambda,
    aux_multilabel_head, aff_top_ratio, aff_component. Missing fields are
    filled with safe defaults so the function can also be used by smoke tests.
    """
    ablation = _cfg_value(args, "ablation", "full")
    task = _cfg_value(args, "task", "multilabel")
    aux_lambda = float(_cfg_value(args, "aux_multilabel_lambda", 0.0))
    aux_head = _cfg_value(args, "aux_multilabel_head", "dual")
    aff_top_ratio = float(_cfg_value(args, "aff_top_ratio", 0.25))
    aff_component = _cfg_value(args, "aff_component", "dominant")

    num_nodes = int(_cfg_value(cfg.model, "num_nodes", 12))
    hidden_dim = int(_cfg_value(cfg.model, "hidden_dim", 64))
    cfg_edge_dropout = float(_cfg_value(cfg.model, "edge_dropout", 0.0))
    edge_dropout = 0.0 if ablation == "wo_edgedrop" else cfg_edge_dropout
    use_aux_head = (task == "single_label" and aux_lambda > 0 and aux_head == "dual")

    if ablation in IMPLEMENTED_BASELINES:
        model = _build_baseline(ablation, num_classes=num_classes)
    elif ablation.startswith("old_dual"):
        model = OldDualFreqDGNN(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            edge_dropout=edge_dropout,
            ablation=ablation,
            use_aux_head=use_aux_head,
        )
    else:
        model = DualFreqDGNN(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            edge_dropout=edge_dropout,
            ablation=ablation,
            use_resnet_fusion=ablation in ["resnet_fusion", "resnet_fusion_wo_cl"],
            use_aux_head=use_aux_head,
            aff_top_ratio=aff_top_ratio,
            aff_component=aff_component,
        )

    return model.to(device) if device is not None else model
