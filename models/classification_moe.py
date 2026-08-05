"""Profile-specific RT-DETR classifier with the audited sparse MoE adapter."""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Mynet import RTDETR_L


DATASET_PROFILES: Dict[str, Dict[str, list[str]]] = {
    "medical5": {
        "chest_xray": ["normal", "pneumonia"],
        "eyepacs": ["mild", "moderate", "proliferate_dr", "severe"],
        "ham10000": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "mura": ["negative", "positive"],
        "pcam": ["0", "1"],
    },
    "fetal_planes": {
        "fetal_planes": [
            "Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax",
            "Maternal cervix", "Other",
        ],
    },
    "cbis_ddsm": {"cbis_ddsm": ["benign", "malignant"]},
    "nct_crc_he100k": {
        "nct_crc_he100k": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
    },
}
DATASET_PROFILE = os.environ.get("CLASSIFICATION_DATASET_PROFILE", "medical5").strip()
if DATASET_PROFILE not in DATASET_PROFILES:
    raise RuntimeError(
        f"Unknown CLASSIFICATION_DATASET_PROFILE={DATASET_PROFILE!r}; "
        f"expected one of {sorted(DATASET_PROFILES)}"
    )
DATASET_CLASSES = DATASET_PROFILES[DATASET_PROFILE]
DATASET_NAMES = list(DATASET_CLASSES)
DATASET_TO_ID = {name: idx for idx, name in enumerate(DATASET_NAMES)}


class ClassificationSparseMoE(nn.Module):
    """Top-k sparse residual adapters for RT-DETR feature maps."""

    def __init__(self, channels: int = 2048, num_experts: int = 4, top_k: int = 2,
                 bottleneck: int = 256, gamma_init: float = 1e-3):
        super().__init__()
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(int(channels), self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, bottleneck, 1, bias=False),
                nn.BatchNorm2d(bottleneck),
                nn.SiLU(inplace=True),
                nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False),
                nn.BatchNorm2d(bottleneck),
                nn.SiLU(inplace=True),
                nn.Conv2d(bottleneck, channels, 1, bias=False),
            )
            for _ in range(self.num_experts)
        ])
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self._last_router_probabilities: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        probabilities = self.router(pooled).softmax(dim=-1)
        self._last_router_probabilities = probabilities
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mixed = torch.zeros_like(features)
        for expert_id, expert in enumerate(self.experts):
            selected = (top_indices == expert_id).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            sample_indices = selected[:, 0]
            slot_indices = selected[:, 1]
            expert_output = expert(features.index_select(0, sample_indices))
            gates = top_values[sample_indices, slot_indices].view(-1, 1, 1, 1)
            mixed.index_add_(0, sample_indices, expert_output * gates)
        return features + self.gamma * mixed

    def load_balance_loss(self) -> torch.Tensor:
        if self._last_router_probabilities is None:
            return self.gamma.new_zeros(())
        mean_probability = self._last_router_probabilities.mean(dim=0)
        target = torch.full_like(mean_probability, 1.0 / self.num_experts)
        return self.num_experts * (mean_probability - target).pow(2).mean()


class MultiDatasetRTDETRClassifier(nn.Module):
    """Shared RT-DETR-L SSL backbone, optional sparse MoE, and profile heads."""

    def __init__(self, dropout: float = 0.1, moe_enabled: bool = False,
                 moe_num_experts: int = 4, moe_top_k: int = 2,
                 moe_bottleneck: int = 256, moe_gamma_init: float = 1e-3):
        super().__init__()
        detector = RTDETR_L(nc=20)
        self.backbone = detector.model[:10]
        self.moe_enabled = bool(moe_enabled)
        self.moe = ClassificationSparseMoE(
            channels=2048,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            bottleneck=moe_bottleneck,
            gamma_init=moe_gamma_init,
        ) if self.moe_enabled else None
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {dataset: nn.Linear(2048, len(classes)) for dataset, classes in DATASET_CLASSES.items()}
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if self.moe is not None:
            features = self.moe(features)
        return self.dropout(self.pool(features).flatten(1))

    def logits_for_dataset(self, features: torch.Tensor, dataset_id: int) -> torch.Tensor:
        return self.heads[DATASET_NAMES[int(dataset_id)]](features)
