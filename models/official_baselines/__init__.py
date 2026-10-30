"""Adapters for external official/public baseline implementations.

The classes in this package do not silently fall back to project-local
look-alike models. If the required public repository is not present, model
construction raises an actionable error. This keeps paper baseline provenance
clean.
"""

from .adapters import (
    OfficialMedformerForECG,
    OfficialPatchTSTForECG,
    OfficialITransformerForECG,
    OfficialFEDformerForECG,
    OfficialFourierGNNForECG,
    OfficialMedGNNForECG,
)

__all__ = [
    "OfficialMedformerForECG",
    "OfficialPatchTSTForECG",
    "OfficialITransformerForECG",
    "OfficialFEDformerForECG",
    "OfficialFourierGNNForECG",
    "OfficialMedGNNForECG",
]
