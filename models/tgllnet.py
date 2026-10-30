# -*- coding: utf-8 -*-
"""TGLLNet baseline for 12-lead ECG multi-label classification.

This is a self-contained implementation of the TGLLNet architecture described in:
"Enhancing Multilabel ECG Classification via Task-Guided Lead Correlations in
Internet of Medical Things" (IEEE Internet of Things Journal, 2025).

The released TGLLnet archive used in this project did not include the core
`Mymodel.py` file, so this module implements the paper-described components
inside the current training/evaluation framework:
  - ETGC-style temporal graph representation: each time step is a 12-node graph.
  - Residual Pyramid GCN (RPG): stacked Chebyshev GCN layers with learnable
    task-guided lead adjacency matrices.
  - Temporal Context Convolution (TCC): multi-scale temporal convolutions plus
    channel and temporal attention.

The wrapper accepts the project-standard forward signature `(x_hf, x_lf)`,
reconstructs ECG as `x_hf + x_lf`, and outputs raw logits for BCEWithLogitsLoss.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding(kernel_size: int) -> int:
    """Padding for odd-sized 1D kernels that preserves sequence length."""
    return int(kernel_size) // 2


class TGLLChebGraphConv(nn.Module):
    """Chebyshev graph convolution with a learnable ECG-lead adjacency matrix.

    Input shape:  [N, C, Fin], where C is the number of ECG leads.
    Output shape: [N, C, Fout].

    A paper-aligned learnable adjacency is constructed as:
        A_hat = clamp(ReLU(A + A^T), 0, 1) + I
    followed by symmetric degree normalization for stable graph filtering.
    """

    def __init__(self, num_leads: int, in_features: int, out_features: int, cheb_order: int = 3):
        super().__init__()
        if cheb_order < 1:
            raise ValueError("cheb_order must be >= 1")
        self.num_leads = int(num_leads)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cheb_order = int(cheb_order)

        # Small random initialization keeps the initial graph close to self-loops
        # after ReLU/clamp, while still allowing task-driven connectivity to grow.
        self.adjacency = nn.Parameter(0.01 * torch.randn(num_leads, num_leads))
        self.weight = nn.Parameter(torch.empty(self.cheb_order, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def normalized_adjacency(self) -> torch.Tensor:
        a = torch.clamp(F.relu(self.adjacency + self.adjacency.transpose(0, 1)), 0.0, 1.0)
        a = a + torch.eye(self.num_leads, dtype=a.dtype, device=a.device)
        deg = a.sum(dim=-1).clamp_min(1e-6)
        d_inv_sqrt = deg.rsqrt()
        return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, Fin]
        a_norm = self.normalized_adjacency()
        supports = []
        t0 = x
        supports.append(t0)
        if self.cheb_order > 1:
            t1 = torch.einsum("ij,njf->nif", a_norm, x)
            supports.append(t1)
            for _ in range(2, self.cheb_order):
                t2 = 2.0 * torch.einsum("ij,njf->nif", a_norm, supports[-1]) - supports[-2]
                supports.append(t2)

        out = x.new_zeros(x.shape[0], self.num_leads, self.out_features)
        for k, support in enumerate(supports):
            out = out + torch.einsum("nci,io->nco", support, self.weight[k])
        out = out + self.bias
        return out


class ResidualPyramidGCN(nn.Module):
    """Residual Pyramid GCN module from TGLLNet.

    For each temporal graph, stacked Chebyshev GCN layers learn multi-level lead
    relationship embeddings. A residual linear branch preserves the unfiltered
    original graph features. The two branches are averaged to form per-time-step
    tokens of dimension `d_model`.
    """

    def __init__(
        self,
        num_leads: int = 12,
        node_features: int = 1,
        gcn_hidden: int = 16,
        d_model: int = 128,
        num_layers: int = 4,
        cheb_order: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_leads = int(num_leads)
        self.node_features = int(node_features)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        layers = []
        in_dim = node_features
        for _ in range(num_layers):
            layers.append(TGLLChebGraphConv(num_leads, in_dim, gcn_hidden, cheb_order=cheb_order))
            in_dim = gcn_hidden
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(num_leads * gcn_hidden),
            nn.Linear(num_leads * gcn_hidden, d_model),
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(num_leads * node_features),
            nn.Linear(num_leads * node_features, d_model),
        )

    def forward(self, graphs: torch.Tensor) -> torch.Tensor:
        # graphs: [B, T, C, 1]
        bsz, seq_len, num_leads, node_features = graphs.shape
        if num_leads != self.num_leads:
            raise ValueError(f"Expected {self.num_leads} leads, got {num_leads}")
        if node_features != self.node_features:
            raise ValueError(f"Expected {self.node_features} node features, got {node_features}")

        x0 = graphs.reshape(bsz * seq_len, num_leads, node_features)
        x = x0
        for layer in self.gcn_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        graph_feat = self.graph_projection(x.reshape(bsz * seq_len, -1))
        residual_feat = self.residual_projection(x0.reshape(bsz * seq_len, -1))
        tokens = 0.5 * (graph_feat + residual_feat)
        return tokens.reshape(bsz, seq_len, self.d_model)

    def lead_adjacency(self) -> torch.Tensor:
        """Return the average learned normalized adjacency across GCN layers."""
        adjs = [layer.normalized_adjacency().detach() for layer in self.gcn_layers]
        return torch.stack(adjs, dim=0).mean(dim=0)


class TGLLChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class TGLLTemporalAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=_same_padding(kernel_size), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class TemporalContextConvolution(nn.Module):
    """TGLLNet temporal context convolution module.

    It stacks several temporal convolutions to capture local context, then uses
    parallel multi-scale convolutions to aggregate short- and long-range ECG
    patterns. Channel and temporal attention refine the representation.
    """

    def __init__(
        self,
        d_model: int = 128,
        basic_kernels: Iterable[int] = (11, 7, 5, 3),
        multiscale_kernels: Iterable[int] = (3, 13, 23, 33),
        dropout: float = 0.1,
    ):
        super().__init__()
        basic_layers = []
        for k in basic_kernels:
            basic_layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size=int(k), padding=_same_padding(int(k)), bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        self.basic = nn.Sequential(*basic_layers)

        self.multiscale = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=int(k), padding=_same_padding(int(k)), bias=False)
            for k in multiscale_kernels
        ])
        self.post = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.channel_attention = TGLLChannelAttention(d_model)
        self.temporal_attention = TGLLTemporalAttention(kernel_size=7)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        residual = tokens.transpose(1, 2).contiguous()  # [B, D, T]
        x = self.basic(residual)
        multi = torch.stack([conv(x) for conv in self.multiscale], dim=0).mean(dim=0)
        x = self.post(multi) + residual
        x = self.channel_attention(x) * x + self.temporal_attention(x) * x + x
        x = x.transpose(1, 2).contiguous()
        return self.norm(x)


class TGLLNetForECG(nn.Module):
    """TGLLNet adapted to the project-standard ECG interface.

    Parameters are intentionally moderate so it can run under the same 5-fold
    PTB-XL training script as the existing baselines. For a closer reproduction
    of the paper, use `d_model=128`, `gcn_layers=4`, and `cheb_order=3`.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_leads: int = 12,
        seq_len: int = 1000,
        d_model: int = 128,
        gcn_hidden: int = 16,
        gcn_layers: int = 4,
        cheb_order: int = 3,
        dropout: float = 0.1,
        standardize_input: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_leads = int(num_leads)
        self.seq_len = int(seq_len)
        self.standardize_input = bool(standardize_input)

        self.rpg = ResidualPyramidGCN(
            num_leads=num_leads,
            node_features=1,
            gcn_hidden=gcn_hidden,
            d_model=d_model,
            num_layers=gcn_layers,
            cheb_order=cheb_order,
            dropout=dropout,
        )
        self.tcc = TemporalContextConvolution(d_model=d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _prepare_ecg(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = x_hf + x_lf  # [B, C, T]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        if self.standardize_input:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            x = ((x - mean) / std).clamp(-8.0, 8.0)
        return x

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor) -> torch.Tensor:
        x = self._prepare_ecg(x_hf, x_lf)     # [B, C, T]
        graphs = x.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, T, C, 1]
        tokens = self.rpg(graphs)             # [B, T, D]
        features = self.tcc(tokens)           # [B, T, D]
        return self.classifier(features.transpose(1, 2).contiguous())

    def lead_adjacency(self) -> torch.Tensor:
        return self.rpg.lead_adjacency()
