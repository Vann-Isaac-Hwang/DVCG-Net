# -*- coding: utf-8 -*-
"""Thin ECG wrappers around external official/public baseline implementations.

These wrappers intentionally perform only task adaptation:
  - x_hf, x_lf -> reconstructed ECG x = x_hf + x_lf
  - [B, 12, T] -> external repositories' usual [B, T, C]
  - external backbone output -> 5 raw logits for BCEWithLogitsLoss

They do not silently substitute project-local style models. If the public source
repo is absent, construction raises ExternalRepoError with git-clone commands.
"""

from __future__ import annotations

from types import SimpleNamespace
from contextlib import contextmanager
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_external_module


def _to_btc(x_hf: torch.Tensor, x_lf: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
    """Reconstruct ECG and return [B, T, C]."""
    x = x_hf + x_lf  # [B, C, T]
    if seq_len is not None and x.shape[-1] != seq_len:
        x = F.interpolate(x, size=seq_len, mode="linear", align_corners=False)
    return x.transpose(1, 2).contiguous()


def _to_btc_standardized(
    x_hf: torch.Tensor,
    x_lf: torch.Tensor,
    seq_len: int | None = None,
    eps: float = 1e-5,
    clip_value: float | None = 8.0,
) -> torch.Tensor:
    """Reconstruct ECG, per-sample/per-lead standardize, and return [B, T, C].

    Medformer and MedGNN official classification backbones do not contain RevIN
    or the iTransformer-style input normalization. PTB-XL amplitude scaling can
    therefore make their first projection/FFT layers poorly conditioned. This is
    an input-adapter normalization only: it does not replace or simplify the
    public-source backbone.
    """
    x = x_hf + x_lf  # [B, C, T]
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if seq_len is not None and x.shape[-1] != seq_len:
        x = F.interpolate(x, size=seq_len, mode="linear", align_corners=False)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    x = (x - mean) / std
    if clip_value is not None:
        x = x.clamp(-clip_value, clip_value)
    return x.transpose(1, 2).contiguous()


def _patch_fedformer_frequency_modes_deterministic(module=None) -> bool:
    """Force the public FEDformer Fourier modules to use deterministic low modes.

    Some released FEDformer revisions sample Fourier indices with NumPy/Python
    random functions and keep those indices as plain Python lists rather than
    state_dict buffers. That makes train-time and eval-time model construction
    choose different frequency bases for the same checkpoint.

    The project loader intentionally removes collision-prone external modules
    from ``sys.modules`` after importing them, so patching only ``sys.modules``
    is not robust. This function therefore also patches the ``__globals__`` of
    the imported FEDformer FourierBlock/FourierCrossAttention classes. Those
    globals are what their ``__init__`` methods consult when they call
    ``get_frequency_modes`` during ``module.Model(configs)`` construction.
    """
    patched = False

    def deterministic_get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
        max_modes = max(1, int(seq_len) // 2)
        m = min(int(modes), max_modes)
        return list(range(m))

    # Patch any still-visible external FourierCorrelation module.
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.endswith("FourierCorrelation") and hasattr(mod, "get_frequency_modes"):
            mod.get_frequency_modes = deterministic_get_frequency_modes
            patched = True

    # Patch the class globals that FEDformer.py imported from FourierCorrelation.
    if module is not None:
        for cls_name in ("FourierBlock", "FourierCrossAttention"):
            cls = getattr(module, cls_name, None)
            init = getattr(cls, "__init__", None)
            glob = getattr(init, "__globals__", None)
            if isinstance(glob, dict):
                glob["get_frequency_modes"] = deterministic_get_frequency_modes
                patched = True

    return patched


@contextmanager
def _ignore_official_fouriergnn_hardcoded_cuda_to():
    """Ignore FourierGNN's hard-coded ``self.to('cuda:0')`` during construction.

    The public FourierGNN release calls ``self.to('cuda:0')`` inside ``FGN.__init__``.
    That is an environment/device placement detail, not part of the backbone.  The
    project-level caller moves the completed wrapper to the requested device after
    construction, so this context prevents CPU smoke tests and non-cuda:0 jobs from
    failing while preserving the released FGN layers/forward computation.
    """
    original_to = nn.Module.to

    def patched_to(module_self, *args, **kwargs):
        if args and str(args[0]) == "cuda:0" and not kwargs:
            return module_self
        return original_to(module_self, *args, **kwargs)

    nn.Module.to = patched_to
    try:
        yield
    finally:
        nn.Module.to = original_to


class OfficialMedformerForECG(nn.Module):
    """Medformer official/public implementation adapted to PTB-XL multi-label ECG.

    Source repo expected: https://github.com/DL4mHealth/Medformer
    Core model file loaded: models/Medformer.py

    P6h fix: use the public run.py classification-scale defaults more closely
    (patch_len_list=2,4,8; e_layers=2) and normalize ECG input in the adapter.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        d_model: int = 256,
        d_ff: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        patch_len_list: str = "2,4,8",
        augmentations: str = "none",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        module = load_external_module("medformer")
        configs = SimpleNamespace(
            task_name="classification",
            pred_len=0,
            output_attention=False,
            enc_in=num_leads,
            single_channel=False,
            patch_len_list=patch_len_list,
            seq_len=seq_len,
            augmentations=augmentations,
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
            no_inter_attn=False,
            d_ff=d_ff,
            activation="gelu",
            e_layers=e_layers,
            num_class=num_classes,
        )
        self.model = module.Model(configs)

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = _to_btc_standardized(x_hf, x_lf, self.seq_len)
        return self.model(x, None, None, None)


class OfficialPatchTSTForECG(nn.Module):
    """PatchTST official implementation adapted as an ECG classification encoder.

    Source repo expected: https://github.com/yuqinie98/PatchTST
    Core model file loaded: PatchTST_supervised/models/PatchTST.py

    The official PatchTST supervised model is a forecasting model. To adapt it
    to classification while preserving the PatchTST backbone, we set its
    `pred_len` to a feature window and replace the final task head with a
    multi-label classifier over the backbone output.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        feature_len: int = 64,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_len = feature_len
        module = load_external_module("patchtst")
        configs = SimpleNamespace(
            enc_in=num_leads,
            seq_len=seq_len,
            pred_len=feature_len,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=dropout,
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=True,
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.backbone = module.Model(configs)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feature_len * num_leads),
            nn.Linear(feature_len * num_leads, num_classes),
        )

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = _to_btc(x_hf, x_lf, self.seq_len)  # [B, T, C]
        features = self.backbone(x)            # official output: [B, feature_len, C]
        return self.classifier(features)


class OfficialITransformerForECG(nn.Module):
    """iTransformer official implementation adapted as an ECG classifier.

    Source repo expected: https://github.com/thuml/iTransformer
    Core model file loaded: model/iTransformer.py

    The official repository is forecasting-oriented. This wrapper preserves the
    inverted embedding and Transformer encoder, while using the classification
    adaptation used by the authors' time-series library: retain all variate
    tokens, apply GELU/dropout, flatten them, and replace the output projector
    with an ECG classification head. Mean-pooling variate tokens is avoided
    because it removes lead-specific information.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        module = load_external_module("itransformer")
        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=1,
            output_attention=False,
            use_norm=True,
            d_model=d_model,
            embed="timeF",
            freq="h",
            dropout=dropout,
            class_strategy="projection",
            factor=1,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            activation="gelu",
        )
        self.model = module.Model(configs)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(num_leads * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_leads * d_model, num_classes),
        )

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = _to_btc(x_hf, x_lf, self.seq_len)  # [B, T, C]
        # Match the official forecast normalization before the inverted encoder.
        if getattr(self.model, "use_norm", False):
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        enc_out = self.model.enc_embedding(x, None)  # [B, variates, d_model]
        enc_out, _ = self.model.encoder(enc_out, attn_mask=None)
        return self.classifier(enc_out)


class OfficialFEDformerForECG(nn.Module):
    """FEDformer official implementation adapted to ECG classification.

    Source repo expected: https://github.com/MAZiqing/FEDformer
    Core model file loaded: models/FEDformer.py

    The original model is forecasting-oriented. We preserve the FEDformer
    encoder/decoder and feed the forecast representation into a classification
    head. This is heavier than the other adapters; run a one-epoch smoke test
    before large 5-fold jobs.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        label_len: int | None = None,
        feature_len: int = 32,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        modes: int = 32,
        mode_select: str = "lowest",
        dropout: float = 0.05,
    ):
        super().__init__()
        # The public FEDformer FourierBlock allocates complex weights with a
        # hard-coded head dimension of 8. Keep n_heads=8 to avoid a runtime
        # mismatch inside layers/FourierCorrelation.py.
        if n_heads != 8:
            raise ValueError("Official FEDformer adapter requires n_heads=8 because the public FourierBlock hard-codes 8 heads.")
        self.seq_len = seq_len
        # The public FEDformer implementation constructs the decoder self-attention
        # Fourier modes for `seq_len // 2 + pred_len` (the standard FEDformer
        # forecasting label window). Passing a shorter decoder length makes the
        # pre-sampled Fourier indices exceed the actual rFFT length. Keep the
        # original label-window convention unless a caller explicitly overrides it
        # with the same value.
        self.label_len = seq_len // 2 if label_len is None else int(label_len)
        if self.label_len != seq_len // 2:
            raise ValueError(
                "Official FEDformer adapter requires label_len=seq_len//2 so that "
                "the decoder Fourier-mode indices match the official backbone."
            )
        self.feature_len = feature_len
        self.num_leads = num_leads
        # Use deterministic low-frequency modes rather than the public
        # implementation's random mode sampling. In the upstream FEDformer code
        # the sampled indices are plain Python lists, not parameters/buffers in
        # state_dict. If random indices are used, train-time and eval-time model
        # construction can silently choose different Fourier modes for the same
        # checkpoint. `mode_select != "random"` follows the upstream deterministic
        # branch (lowest frequency modes) and makes checkpoints reproducible.
        if mode_select == "random":
            raise ValueError(
                "Official FEDformer adapter disables random Fourier mode sampling "
                "because upstream mode indices are not stored in state_dict. "
                "Use mode_select='lowest' for train/eval consistency."
            )
        self.mode_select = mode_select
        self.n_heads = 8
        self._deterministic_fedformer_patch_version = "P6g-class-globals-low-modes-nheads8"

        module = load_external_module("fedformer")
        patched = _patch_fedformer_frequency_modes_deterministic(module)
        if not patched:
            raise RuntimeError(
                "Failed to patch FEDformer FourierCorrelation.get_frequency_modes through either "
                "sys.modules or imported class globals. Check whether the external FEDformer "
                "repo structure changed."
            )
        # Keep both common config field spellings. Some public FEDformer forks
        # read configs.mode_select, others read configs.mode_select_method.
        configs = SimpleNamespace(
            version="Fourier",
            mode_select=mode_select,
            mode_select_method=mode_select,
            modes=modes,
            seq_len=seq_len,
            label_len=self.label_len,
            pred_len=feature_len,
            output_attention=False,
            moving_avg=25,
            enc_in=num_leads,
            dec_in=num_leads,
            c_out=num_leads,
            d_model=d_model,
            embed="timeF",
            dropout=dropout,
            freq="h",
            factor=1,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=d_layers,
            activation="gelu",
            # Wavelet-only fields kept for compatibility with official class.
            L=1,
            base="legendre",
            cross_activation="tanh",
            wavelet=0,
        )
        self.model = module.Model(configs)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feature_len * num_leads),
            nn.Linear(feature_len * num_leads, num_classes),
        )

    def _time_marks(self, batch_size: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # FEDformer timeF embedding expects continuous calendar-like features.
        # ECG has no calendar covariates, so we provide zero covariates and keep
        # the value pathway/backbone unchanged.
        return torch.zeros(batch_size, length, 4, device=device, dtype=dtype)

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = _to_btc(x_hf, x_lf, self.seq_len)  # [B, T, C]
        B, _, C = x.shape
        dec_zeros = torch.zeros(B, self.label_len + self.feature_len, C, device=x.device, dtype=x.dtype)
        dec_zeros[:, : self.label_len, :] = x[:, -self.label_len :, :]
        x_mark_enc = self._time_marks(B, self.seq_len, x.device, x.dtype)
        x_mark_dec = self._time_marks(B, self.label_len + self.feature_len, x.device, x.dtype)
        features = self.model(x, x_mark_enc, dec_zeros, x_mark_dec)  # [B, feature_len, C]
        return self.classifier(features)


class OfficialFourierGNNForECG(nn.Module):
    """FourierGNN official implementation adapted as an ECG classifier.

    Source repo expected: https://github.com/aikunyi/FourierGNN
    Core model file loaded: model/FourierGNN.py

    FourierGNN is released for multivariate time-series forecasting.  This
    wrapper preserves the public FGN backbone/Fourier graph operator and adapts
    only the ECG input normalization plus the final forecasting output to a
    multi-label classification head.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        feature_len: int = 32,
        embed_size: int = 128,
        hidden_size: int = 256,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        sparsity_threshold: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_len = feature_len
        self.num_leads = num_leads
        module = load_external_module("fouriergnn")
        if not hasattr(module, "FGN"):
            raise RuntimeError("FourierGNN repository module does not expose class FGN in model/FourierGNN.py")
        with _ignore_official_fouriergnn_hardcoded_cuda_to():
            self.backbone = module.FGN(
                pre_length=feature_len,
                embed_size=embed_size,
                feature_size=num_leads,
                seq_length=seq_len,
                hidden_size=hidden_size,
                hard_thresholding_fraction=hard_thresholding_fraction,
                hidden_size_factor=hidden_size_factor,
                sparsity_threshold=sparsity_threshold,
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_leads * feature_len),
            nn.Dropout(dropout),
            nn.Linear(num_leads * feature_len, num_classes),
        )

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        # FourierGNN works on [B, L, N].  Use the same per-sample/per-lead
        # standardization as Medformer/MedGNN because the FGN spectral MLP has
        # no RevIN-style input scaling in the public code.
        x = _to_btc_standardized(x_hf, x_lf, self.seq_len)
        features = self.backbone(x)  # public FGN output: [B, num_leads, feature_len]
        return self.classifier(features)


class OfficialMedGNNForECG(nn.Module):
    """MedGNN official implementation adapted to PTB-XL multi-label ECG.

    Source repo expected: https://github.com/aikunyi/MedGNN
    Core model file loaded: models/MedGNN.py

    The public MedGNN run script defines the MedGNN-specific defaults
    resolution_list="2,4,6,8" and nodedim=10. Keep these defaults here
    so the adapter remains close to the released configuration; only the ECG
    input interface and final number of classes are task-adapted.

    P6h fix: normalize ECG input in the adapter to stabilize MedGNN's FFT and
    graph layers on PTB-XL amplitudes.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        d_model: int = 256,
        d_ff: int = 512,
        n_heads: int = 8,
        e_layers: int = 4,
        resolution_list: str = "2,4,6,8",
        nodedim: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        module = load_external_module("medgnn")
        configs = SimpleNamespace(
            enc_in=num_leads,
            seq_len=seq_len,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            output_attention=False,
            activation="gelu",
            resolution_list=resolution_list,
            nodedim=nodedim,
            augmentations="none",
            num_class=num_classes,
        )
        self.model = module.Model(configs)

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = _to_btc_standardized(x_hf, x_lf, self.seq_len)
        return self.model(x, None, None, None)
