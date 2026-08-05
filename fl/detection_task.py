"""FL: A Flower / PyTorch app."""

import os
from pathlib import Path
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ============================================================================
# [新增] 医疗数据集配置字典
# 使用方法: 修改 CURRENT_DATASET 变量来选择要运行的数据集
# ============================================================================
DATASET_CONFIGS = {
    # 原有数据集配置 (保留)
    "voc2007": {
        "name": "VOC2007",
        "nc": 20,
        "train_loader": "datasets/coco8/coco8_load_train.py",
        "val_loader":   "datasets/coco8/coco8_load_val.py",
        "test_loader":  "datasets/coco8/coco8_load_test.py",
        "data_dir": "datasets/local",
        "domain_id": 3,
    },

    # COVID-19 肺部检测数据集
    "covid19_lung_detect": {
        "name": "COVID-19 Lung Detection",
        "nc": 2,  # 类别: Right Lung, Left Lung
        "train_loader": "datasets/medical_data/covid19_lung_detect/load_train.py",
        "val_loader":   "datasets/medical_data/covid19_lung_detect/load_val.py",
        "test_loader":  "datasets/medical_data/covid19_lung_detect/load_test.py",
        "data_dir": "datasets/medical_data/covid19_lung_detect",
        "domain_id": 0,
    },

    # Hyper-Kvasir 息肉检测数据集
    "hyper_kvasir_polyp_detect": {
        "name": "Hyper-Kvasir Polyp Detection",
        "nc": 1,  # 类别: polyp
        "train_loader": "datasets/medical_data/hyper_kvasir_polyp_detect/load_train.py",
        "val_loader":   "datasets/medical_data/hyper_kvasir_polyp_detect/load_val.py",
        "test_loader":  "datasets/medical_data/hyper_kvasir_polyp_detect/load_test.py",
        "data_dir": "datasets/medical_data/hyper_kvasir_polyp_detect",
        "domain_id": 3,
    },

    # Hyper-Kvasir 息肉分割数据集
    "hyper_kvasir_polyp_segment": {
        "name": "Hyper-Kvasir Polyp Segmentation",
        "nc": 1,  # 类别: polyp
        "train_loader": "datasets/medical_data/hyper_kvasir_polyp_segment/load_train.py",
        "val_loader":   "datasets/medical_data/hyper_kvasir_polyp_segment/load_val.py",
        "test_loader":  "datasets/medical_data/hyper_kvasir_polyp_segment/load_test.py",
        "data_dir": "datasets/medical_data/hyper_kvasir_polyp_segment",
        "domain_id": 3,
    },

    # MedCoData 五个目标检测数据集，每个 client 对应一个医学检测域
    "medical5_detect": {
        "name": "MedCoData Medical5 Detection",
        "nc": 12,
        "data_dir": "datasets/medical_data/medical5_detect",
        "domain_id": -1,
        "class_names": [
            "urine_cast", "urine_epith", "urine_eryth", "urine_leuko", "urine_mycete",
            "kvasir_polyp",
            "txl_wbc", "txl_rbc", "txl_platelets",
            "mitosis",
            "tn5000_0", "tn5000_1",
        ],
        "clients": [
            {"name": "urine", "data_dir": "datasets/medical_data/medical5_detect/client0_urine", "domain_id": 0},
            {"name": "kvasir", "data_dir": "datasets/medical_data/medical5_detect/client1_kvasir", "domain_id": 1},
            {"name": "txl_pbc", "data_dir": "datasets/medical_data/medical5_detect/client2_txl_pbc", "domain_id": 2},
            {"name": "mitosis", "data_dir": "datasets/medical_data/medical5_detect/client3_mitosis", "domain_id": 3},
            {"name": "tn5000", "data_dir": "datasets/medical_data/medical5_detect/client4_tn5000", "domain_id": 4},
        ],
    },

    # Mixed-client 20% labeled split for formal supervised FL after SSL.
    "medical5_mixed_labeled20": {
        "name": "MedCoData Medical5 Mixed-Client Labeled20 Detection",
        "nc": 12,
        "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20",
        "domain_id": -1,
        "class_names": [
            "urine_cast", "urine_epith", "urine_eryth", "urine_leuko", "urine_mycete",
            "kvasir_polyp",
            "txl_wbc", "txl_rbc", "txl_platelets",
            "mitosis",
            "tn5000_0", "tn5000_1",
        ],
        "clients": [
            {"name": "mixed_client0", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20/client0/labeled", "domain_id": 0},
            {"name": "mixed_client1", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20/client1/labeled", "domain_id": 1},
            {"name": "mixed_client2", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20/client2/labeled", "domain_id": 2},
            {"name": "mixed_client3", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20/client3/labeled", "domain_id": 3},
            {"name": "mixed_client4", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20/client4/labeled", "domain_id": 4},
        ],
    },
}

# 30% train subset for faster hyperparameter tuning; val/test stay full-size.
DATASET_CONFIGS["medical5_mixed_labeled20_tune30"] = {
    **DATASET_CONFIGS["medical5_mixed_labeled20"],
    "name": "MedCoData Medical5 Mixed-Client Labeled20 Detection Tune30",
    "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30",
    "clients": [
        {"name": "mixed_client0_tune30", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30/client0/labeled", "domain_id": 0},
        {"name": "mixed_client1_tune30", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30/client1/labeled", "domain_id": 1},
        {"name": "mixed_client2_tune30", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30/client2/labeled", "domain_id": 2},
        {"name": "mixed_client3_tune30", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30/client3/labeled", "domain_id": 3},
        {"name": "mixed_client4_tune30", "data_dir": "datasets/medical_data/medical5_detect_mixed_ssl80_label20_tune30/client4/labeled", "domain_id": 4},
    ],
}

# Full 33,877-image Medical5 split: 80% SSL, 20% labeled; labeled 7:1:2.
DATASET_CONFIGS["medical5_full33877_mixed_labeled20_712"] = {
    **DATASET_CONFIGS["medical5_mixed_labeled20"],
    "name": "MedCoData Medical5 Full33877 Mixed-Client Labeled20 7-1-2 Detection",
    "data_dir": "datasets/medical_data/medical5_detect_full33877_mixed_ssl80_label20_712",
    "clients": [
        {
            "name": f"full33877_mixed_client{client_id}",
            "data_dir": (
                "datasets/medical_data/"
                f"medical5_detect_full33877_mixed_ssl80_label20_712/client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["medical5_full33877_branch_a_random"] = {
    **DATASET_CONFIGS["medical5_full33877_mixed_labeled20_712"],
    "name": "MedCoData Full33877 Branch A Random Signature-Balanced",
}

DATASET_CONFIGS["medical5_full33877_branch_b_boxbalanced"] = {
    **DATASET_CONFIGS["medical5_full33877_mixed_labeled20_712"],
    "name": "MedCoData Full33877 Branch B Box-Count and Density-Balanced",
    "data_dir": (
        "datasets/medical_data/"
        "medical5_detect_full33877_mixed_ssl80_label20_712_branch_b_boxbalanced"
    ),
    "clients": [
        {
            "name": f"branch_b_boxbalanced_client{client_id}",
            "data_dir": (
                "datasets/medical_data/"
                "medical5_detect_full33877_mixed_ssl80_label20_712_branch_b_boxbalanced/"
                f"client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["medical5_detection_clientfirst_seed42"] = {
    **DATASET_CONFIGS["medical5_mixed_labeled20"],
    "name": "MedCoData Medical5 Client-First Detection Seed42",
    "data_dir": "datasets/medical_data/medical5_detect_clientfirst_ssl80_label20_seed42",
    "clients": [
        {
            "name": f"clientfirst_client{client_id}",
            "data_dir": (
                "datasets/medical_data/"
                f"medical5_detect_clientfirst_ssl80_label20_seed42/client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["medical5_detection_clientfirst_rebalanced_seed20260719"] = {
    **DATASET_CONFIGS["medical5_detection_clientfirst_seed42"],
    "name": "MedCoData Medical5 Client-First Detection Labeled Rebalanced",
    "data_dir": "datasets/medical_data/medical5_detect_clientfirst_ssl80_label20_labeled_rebalanced_seed20260719",
    "clients": [
        {
            "name": f"clientfirst_rebalanced_client{client_id}",
            "data_dir": (
                "datasets/medical_data/"
                f"medical5_detect_clientfirst_ssl80_label20_labeled_rebalanced_seed20260719/client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["cbis_ddsm_clientfirst_seed42"] = {
    "name": "CBIS-DDSM Client-First Detection Seed42",
    "nc": 2,
    "class_names": ["mass", "calcification"],
    "data_dir": "datasets/detection/cbis_ddsm/prepared/cbis_ddsm_clientfirst_seed42",
    "clients": [
        {
            "name": f"cbis_ddsm_client{client_id}",
            "data_dir": (
                "datasets/detection/cbis_ddsm/prepared/"
                f"cbis_ddsm_clientfirst_seed42/client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["tbx11k_clientfirst_seed42"] = {
    "name": "TBX11K Public TrainVal Client-First Detection Seed42",
    "nc": 2,
    "class_names": ["active_tb", "latent_tb"],
    "data_dir": os.environ.get(
        "TBX11K_DATA_ROOT",
        "datasets/detection/tbx11k_clientfirst_seed42",
    ),
    "clients": [
        {
            "name": f"tbx11k_client{client_id}",
            "data_dir": str(
                Path(os.environ.get("TBX11K_DATA_ROOT", "datasets/detection/tbx11k_clientfirst_seed42"))
                / f"client{client_id}"
                / "labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

DATASET_CONFIGS["totalsegmentator_clientfirst_seed42"] = {
    "name": "TotalSegmentator V1 Client-First Detection Seed42",
    "nc": 8,
    "class_names": [
        "spleen",
        "kidney_right",
        "kidney_left",
        "liver",
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "heart_myocardium",
        "vertebrae_L1",
    ],
    "data_dir": "datasets/detection/totalsegmentator/prepared/totalsegmentator_clientfirst_seed42",
    "clients": [
        {
            "name": f"totalsegmentator_client{client_id}",
            "data_dir": (
                "datasets/detection/totalsegmentator/prepared/"
                f"totalsegmentator_clientfirst_seed42/client{client_id}/labeled"
            ),
            "domain_id": client_id,
        }
        for client_id in range(5)
    ],
}

# [新增] 当前使用的数据集 - 修改此变量切换数据集
# 可选值: "voc2007", "covid19_lung_detect", "hyper_kvasir_polyp_detect", "hyper_kvasir_polyp_segment"
CURRENT_DATASET = os.environ.get("FL_CURRENT_DATASET", "tbx11k_clientfirst_seed42")

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.optim as optim
from models.detection_loss import RTDETRDetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from flwr.common import Array, ArrayRecord
import torch
from torchvision.ops import box_iou, box_convert
from datetime import datetime

import logging
from flwr.common.logger import log, configure

configure(identifier="task")

PARTITION_SEED = 42

LOSS_KWARGS = {
    "use_vfl": True,
    "use_eqlv2": True,
    "loss_gain": {
        "class": 0.5,
        "eqlv2": 0.5,
        "bbox": 5.0,
        "giou": 2.0,
        "no_object": 0.1,
        "mask": 1.0,
        "dice": 1.0,
    },
    "gamma": 1.5,
    "alpha": 0.25,
    "eql_gamma": 12.0,
    "eql_mu": 0.8,
    "eql_alpha": 4.0,
}

# ============================================================================
# FedBN State Persistence Manager
# 解决 Ray 并行 Actor 下 FedBN 状态丢失的问题
# 使用文件持久化代替进程内存缓存
# ============================================================================
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Set, Tuple

# 全局状态目录（可配置）
FEDBN_STATE_DIR = Path(os.environ.get("FEDBN_STATE_DIR", "/tmp/fl_fedbn_states"))

# ============================================================================
# Mode Signature & State Isolation
# 解决多模式配置下的状态串用问题
# ============================================================================

# 当前运行的联邦模式签名（用于状态隔离和校验）
# 格式: {model_variant}:{strategy}:{dataset}:{nc}
# 例如: "RTDETR_L_WithAttention:FedAvg:hyper_kvasir_polyp_detect:1"
_current_mode_signature: Optional[str] = None


def get_mode_signature(
    model_variant: str = None,
    strategy: str = None,
    dataset: str = None,
    nc: int = None
) -> str:
    """
    生成模式签名，用于状态隔离和跨模式加载检测。

    Args:
        model_variant: 模型变体名称（如 'RTDETR_L', 'RTDETR_L_WithAttention'）
        strategy: 联邦策略名称（如 'FedAvg', 'FedYogi', 'FedBN'）
        dataset: 数据集名称
        nc: 类别数

    Returns:
        模式签名字符串
    """
    parts = []
    if model_variant:
        parts.append(f"model:{model_variant}")
    if strategy:
        parts.append(f"strategy:{strategy}")
    if dataset:
        parts.append(f"dataset:{dataset}")
    if nc:
        parts.append(f"nc:{nc}")

    return "|".join(parts) if parts else "default"


def set_current_mode(signature: str):
    """设置当前运行的模式签名"""
    global _current_mode_signature
    _current_mode_signature = signature
    log(logging.INFO, f"Mode signature set: {signature}")


def get_current_mode() -> Optional[str]:
    """获取当前运行的模式签名"""
    return _current_mode_signature


def extract_model_signature(model) -> str:
    """
    从模型实例提取关键特征签名。
    用于验证 server 发送的 arrays 与 client 模型是否匹配。
    """
    model_class = model.__class__.__name__

    # 检测关键特征
    features = []

    # 检测是否有 retnet 模块
    if hasattr(model, 'retnet_p5'):
        features.append("retnet_p5")
    if hasattr(model, 'retnet_f5'):
        features.append("retnet_f5")
    if hasattr(model, 'retnet_scale'):
        features.append("retnet_scale")

    # 检测是否有 ASEM / MoE 模块
    if hasattr(model, 'asem_p5'):
        features.append("asem_p5")
        if hasattr(model.asem_p5, 'routing_mode'):
            features.append(f"asem_routing={model.asem_p5.routing_mode}")
        if hasattr(model.asem_p5, 'top_k'):
            features.append(f"asem_topk={model.asem_p5.top_k}")

    # 获取 decoder nc
    if hasattr(model, 'nc'):
        features.append(f"nc={model.nc}")

    signature = f"{model_class}:{','.join(features)}"
    return signature


def validate_arrays_model_compatibility(
    model,
    array_keys: Set[str],
    mode_signature: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    验证 incoming arrays keys 与 model state_dict keys 的兼容性。

    Args:
        model: PyTorch 模型
        array_keys: Server 发送的 arrays 的 key 集合
        mode_signature: 可选的模式签名用于校验

    Returns:
        (is_compatible, diagnostic_info)
        - is_compatible: True 如果兼容
        - diagnostic_info: 包含 missing_keys, unexpected_keys, mode_mismatch 等诊断信息
    """
    model_keys = set(model.state_dict().keys())
    param_keys = set(name for name, _ in model.named_parameters())

    # 分类分析
    missing_in_model = array_keys - model_keys  # arrays 有，model 没有
    missing_in_arrays = model_keys - array_keys  # model 有，arrays 没有

    # 区分 parameter 和 buffer
    missing_params = [k for k in missing_in_model if k in param_keys or k not in model_keys]
    missing_buffers = [k for k in missing_in_model if k not in param_keys and '.' in k]

    # 检测模式不匹配的特征
    mode_mismatch_indicators = []

    # RTDETR_L vs RTDETR_L_WithAttention 检测
    has_retnet_in_arrays = any('retnet' in k for k in array_keys)
    has_retnet_in_model = hasattr(model, 'retnet_p5') or hasattr(model, 'retnet_f5')

    if has_retnet_in_arrays and not has_retnet_in_model:
        mode_mismatch_indicators.append(
            "Arrays contains 'retnet' keys but model has no RetNet module"
        )
    elif not has_retnet_in_arrays and has_retnet_in_model:
        mode_mismatch_indicators.append(
            "Arrays missing 'retnet' keys but model expects RTDETR_L_WithAttention parameters"
        )

    # RTDETR_L vs RTDETR_L_WithASEM 检测
    has_asem_in_arrays = any('asem' in k for k in array_keys)
    has_asem_in_model = hasattr(model, 'asem_p5')

    if has_asem_in_arrays and not has_asem_in_model:
        mode_mismatch_indicators.append(
            "Arrays contains 'asem' keys but model has no ASEM module"
        )
    elif not has_asem_in_arrays and has_asem_in_model:
        mode_mismatch_indicators.append(
            "Arrays missing 'asem' keys but model expects RTDETR_L_WithASEM parameters"
        )

    # nc 检测（通过 decoder 参数）
    decoder_keys_arrays = [k for k in array_keys if 'decoder' in k.lower() or k.startswith('model.28')]
    decoder_keys_model = [k for k in model_keys if 'decoder' in k.lower() or k.startswith('model.28')]

    # 简单的兼容性判断
    is_compatible = len(missing_params) == 0 and len(mode_mismatch_indicators) == 0

    diagnostic = {
        "is_compatible": is_compatible,
        "missing_in_model": sorted(missing_in_model),
        "missing_in_arrays": sorted(missing_in_arrays),
        "missing_params": sorted(missing_params),
        "missing_buffers": sorted(missing_buffers),
        "mode_mismatch_indicators": mode_mismatch_indicators,
        "model_signature": extract_model_signature(model),
        "array_count": len(array_keys),
        "model_key_count": len(model_keys),
    }

    return is_compatible, diagnostic


def format_compatibility_error(diagnostic: Dict[str, Any]) -> str:
    """
    格式化兼容性错误信息，用于清晰的错误报告。
    """
    lines = [
        "=".join("=" * 40),
        "⚠️ FEDERATED ARRAY / MODEL COMPATIBILITY ERROR",
        "=".join("=" * 40),
        f"Model Signature: {diagnostic['model_signature']}",
        f"Arrays count: {diagnostic['array_count']}, Model keys count: {diagnostic['model_key_count']}",
    ]

    if diagnostic['mode_mismatch_indicators']:
        lines.append("\n🔍 MODE MISMATCH DETECTED:")
        for indicator in diagnostic['mode_mismatch_indicators']:
            lines.append(f"  • {indicator}")

    if diagnostic['missing_in_model']:
        lines.append("\n❌ Keys in arrays but NOT in model state_dict:")
        # 只显示前 10 个，避免日志过长
        for k in diagnostic['missing_in_model'][:10]:
            lines.append(f"  • {k}")
        if len(diagnostic['missing_in_model']) > 10:
            lines.append(f"  ... and {len(diagnostic['missing_in_model']) - 10} more")

    if diagnostic['missing_in_arrays']:
        lines.append("\n⚠️ Keys in model but NOT in arrays (will keep model defaults):")
        for k in diagnostic['missing_in_arrays'][:10]:
            lines.append(f"  • {k}")
        if len(diagnostic['missing_in_arrays']) > 10:
            lines.append(f"  ... and {len(diagnostic['missing_in_arrays']) - 10} more")

    lines.append("\n" + "=" * 40)
    lines.append("POSSIBLE FIXES:")
    lines.append("  1. Ensure server and client use the SAME model class")
    lines.append("  2. Check if FedAvg/FedBN mode is consistent on both sides")
    lines.append("  3. Verify pretrained checkpoint is compatible with model")
    lines.append("=" * 40)

    return "\n".join(lines)

# 用于跟踪已参与训练的 client（用于检测异常缺失）
_client_training_history: Dict[str, int] = {}  # {client_key: last_round}


class FedBNStateManager:
    """
    FedBN 状态持久化管理器。
    使用文件系统存储每个 client 的 BN 状态，确保跨 actor/process 可访问。
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or FEDBN_STATE_DIR
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """确保基础目录存在"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_path(self, run_id, client_id: int) -> Path:
        """获取状态文件路径"""
        # run_id 可能是 int 或 str，统一转为 str
        return self.base_dir / str(run_id) / f"client_{client_id}.pt"

    def _get_meta_path(self, run_id, client_id: int) -> Path:
        """获取元数据文件路径"""
        return self.base_dir / str(run_id) / f"client_{client_id}.json"

    def _get_run_dir(self, run_id) -> Path:
        """获取 run 目录"""
        return self.base_dir / str(run_id)

    def save_state(
        self,
        run_id,  # int or str
        client_id: int,
        round_num: int,
        local_state: Dict[str, Any]
    ) -> Path:
        """
        保存 client 的 FedBN 本地状态到磁盘。

        Args:
            run_id: 联邦学习运行标识（int 或 str）
            client_id: 客户端 ID
            round_num: 当前轮次
            local_state: BN 状态字典

        Returns:
            保存的文件路径
        """
        run_dir = self._get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        state_path = self._get_state_path(run_id, client_id)
        meta_path = self._get_meta_path(run_id, client_id)

        # 原子写入：先写临时文件，再重命名
        # 使用 .pt.tmp 和 .json.tmp 作为临时文件名
        temp_state_path = Path(str(state_path) + '.tmp')
        temp_meta_path = Path(str(meta_path) + '.tmp')

        try:
            # 保存状态张量
            torch.save(local_state, temp_state_path)

            # 保存元数据
            meta = {
                "run_id": run_id,
                "client_id": client_id,
                "round_num": round_num,
                "num_tensors": len(local_state),
                "tensor_names": list(local_state.keys())[:10],  # 只保存前10个名字
            }
            with open(temp_meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            # 原子重命名
            temp_state_path.rename(state_path)
            temp_meta_path.rename(meta_path)

            # 记录训练历史
            client_key = f"{run_id}:{client_id}"
            _client_training_history[client_key] = round_num

            log(logging.INFO,
                f"[Client {client_id}] Saved FedBN local state to persistent store: "
                f"{state_path} ({len(local_state)} tensors, round {round_num})")

            return state_path

        except Exception as e:
            # 清理临时文件
            if temp_state_path.exists():
                temp_state_path.unlink()
            if temp_meta_path.exists():
                temp_meta_path.unlink()
            log(logging.ERROR,
                f"[Client {client_id}] Failed to save FedBN state: {e}")
            raise

    def load_state(
        self,
        run_id,  # int or str
        client_id: int,
        round_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        从磁盘加载 client 的 FedBN 本地状态。

        Args:
            run_id: 联邦学习运行标识（int 或 str）
            client_id: 客户端 ID
            round_num: 当前轮次（用于日志和异常检测）

        Returns:
            状态字典，如果不存在则返回 None
        """
        state_path = self._get_state_path(run_id, client_id)
        client_key = f"{run_id}:{client_id}"

        if not state_path.exists():
            # 判断是首次运行还是异常缺失
            if round_num > 1 and client_key in _client_training_history:
                log(logging.WARNING,
                    f"[Client {client_id}] ⚠️ ANOMALY: FedBN local state NOT found at round {round_num}, "
                    f"but client was trained in round {_client_training_history[client_key]}. "
                    f"Expected path: {state_path}")
            elif round_num > 1:
                log(logging.WARNING,
                    f"[Client {client_id}] ⚠️ FedBN local state NOT found at round {round_num}. "
                    f"This may indicate the client skipped training in earlier rounds. "
                    f"Expected path: {state_path}")
            else:
                log(logging.INFO,
                    f"[Client {client_id}] No FedBN local state found (round 1, first training). "
                    f"Using model defaults.")
            return None

        try:
            local_state = torch.load(state_path, map_location='cpu', weights_only=True)

            # 更新训练历史记录
            _client_training_history[client_key] = round_num

            log(logging.INFO,
                f"[Client {client_id}] Loaded FedBN local state from persistent store: "
                f"{state_path} ({len(local_state)} tensors, round {round_num})")

            return local_state

        except Exception as e:
            log(logging.ERROR,
                f"[Client {client_id}] Failed to load FedBN state from {state_path}: {e}")
            return None

    def state_exists(self, run_id, client_id: int) -> bool:
        """检查状态文件是否存在"""
        return self._get_state_path(run_id, client_id).exists()

    def clear_run(self, run_id):
        """清理某个 run 的所有状态文件"""
        run_dir = self._get_run_dir(run_id)
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
            log(logging.INFO, f"Cleared FedBN state directory: {run_dir}")


# 全局状态管理器实例
_fedbn_state_manager: Optional[FedBNStateManager] = None


def get_fedbn_state_manager() -> FedBNStateManager:
    """获取全局 FedBN 状态管理器实例"""
    global _fedbn_state_manager
    if _fedbn_state_manager is None:
        _fedbn_state_manager = FedBNStateManager()
    return _fedbn_state_manager


# ============================================================================
# 旧的内存缓存（保留以兼容，但标记为废弃）
# ============================================================================
LOCAL_STATE_CACHE = {}  # DEPRECATED: 保留以兼容旧代码，但不再使用


def _array_from_tensor(tensor):
    """Convert a tensor to Flower Array while normalizing scalar shapes."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return Array(arr)


def get_bn_layer_prefixes(model):
    """Return module-name prefixes for all BatchNorm layers."""
    prefixes = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefixes.add(module_name)
    return prefixes


def get_federated_param_names(model):
    """Return trainable parameter names excluding all BatchNorm params."""
    bn_prefixes = get_bn_layer_prefixes(model)
    federated_names = []
    for name, _ in model.named_parameters():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        if module_name in bn_prefixes:
            continue
        federated_names.append(name)
    return federated_names


def get_local_state_names(model):
    """Return state_dict names kept local under FedBN."""
    federated_names = set(get_federated_param_names(model))
    return [name for name in model.state_dict().keys() if name not in federated_names]


def describe_fedbn_layout(model, max_items=5):
    """Return a short human-readable FedBN layout summary."""
    federated_names = get_federated_param_names(model)
    local_names = get_local_state_names(model)
    return (
        f"federated_params={len(federated_names)} "
        f"sample={federated_names[:max_items]}, "
        f"local_state={len(local_names)} "
        f"sample={local_names[:max_items]}"
    )


def get_federated_model_record(model):
    """Serialize only non-BN trainable parameters for federated optimization."""
    input_data_for_record = {}
    params = dict(model.named_parameters())
    for name in get_federated_param_names(model):
        input_data_for_record[name] = _array_from_tensor(params[name])

    return ArrayRecord(input_data_for_record)


def load_federated_model_record(model, record, strict_mode_check: bool = True):
    """
    Load federated parameters/buffers from ArrayRecord into model.

    This function supports two scenarios:
    1. FedBN + FedYogi mode: record contains only non-BN federated params
    2. FedAvg mode: record contains full state_dict (params + buffers)

    Args:
        model: PyTorch 模型
        record: Flower ArrayRecord containing tensors
        strict_mode_check: 如果 True，在检测到跨模式加载时抛出错误
                          如果 False，允许宽松加载（仅警告）

    Raises:
        KeyError: 如果 strict_mode_check=True 且检测到跨模式不匹配
        ValueError: 如果 arrays 和 model 完全不兼容
    """
    # 获取 incoming arrays 的 keys
    array_keys = set(record.keys())

    # 获取 model 的完整 state_dict（包括 params + buffers）
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    param_keys = set(name for name, _ in model.named_parameters())

    # 兼容性检测
    is_compatible, diagnostic = validate_arrays_model_compatibility(model, array_keys)

    # 打印诊断信息（帮助调试）
    if not is_compatible or diagnostic['missing_in_model'] or diagnostic['missing_in_arrays']:
        log(logging.INFO,
            f"[Model Load] Compatibility check: model={diagnostic['model_signature']}, "
            f"arrays={len(array_keys)} keys, model={len(model_keys)} keys")

        if diagnostic['missing_in_model']:
            log(logging.WARNING,
                f"[Model Load] {len(diagnostic['missing_in_model'])} keys in arrays NOT found in model: "
                f"{diagnostic['missing_in_model'][:5]}...")

        if diagnostic['missing_in_arrays']:
            log(logging.INFO,
                f"[Model Load] {len(diagnostic['missing_in_arrays'])} model keys NOT in arrays (using defaults): "
                f"{diagnostic['missing_in_arrays'][:5]}...")

        if diagnostic['mode_mismatch_indicators']:
            for indicator in diagnostic['mode_mismatch_indicators']:
                log(logging.WARNING, f"[Model Load] ⚠️ {indicator}")

    # 如果检测到跨模式加载，抛出错误或警告
    if diagnostic['mode_mismatch_indicators'] and strict_mode_check:
        error_msg = format_compatibility_error(diagnostic)
        log(logging.ERROR, error_msg)
        raise ValueError(
            f"Cross-mode loading detected! {diagnostic['mode_mismatch_indicators'][0]}. "
            f"Model: {diagnostic['model_signature']}. "
            f"Set strict_mode_check=False to allow loose loading."
        )

    # 区分两种加载模式
    # 检测是否为 FedAvg 模式（arrays 包含 BN 参数或 buffer）
    bn_keys_in_arrays = [k for k in array_keys if 'bn' in k.lower() or 'batchnorm' in k.lower()]
    buffer_keys_in_arrays = [k for k in array_keys if k not in param_keys]

    is_fedavg_mode = len(bn_keys_in_arrays) > 0 or len(buffer_keys_in_arrays) > len(array_keys) * 0.1

    if is_fedavg_mode:
        # FedAvg 模式：加载完整 state_dict（包括 buffers）
        log(logging.INFO,
            f"[Model Load] FedAvg mode: loading {len(array_keys)} keys "
            f"(includes {len(buffer_keys_in_arrays)} buffers)")

        loaded_count = 0
        skipped_count = 0

        for name, array in record.items():
            if name not in model_state_dict:
                # Key 不存在于 model，跳过并记录
                skipped_count += 1
                if skipped_count <= 5:
                    log(logging.WARNING, f"[Model Load] Skipped key not in model: {name}")
                continue

            tensor = torch.from_numpy(array.numpy())
            target = model_state_dict[name]

            # 形状处理
            if tensor.ndim == 1 and tensor.numel() == 1 and target.ndim == 0:
                tensor = tensor.reshape(())

            tensor = tensor.to(device=target.device, dtype=target.dtype)

            # 直接写入 state_dict（对 parameter 会自动更新，对 buffer 需要特殊处理）
            if name in param_keys:
                # Parameter: 直接 copy
                target.copy_(tensor)
            else:
                # Buffer: 直接 copy（buffer 不在 named_parameters 中）
                target.copy_(tensor)

            loaded_count += 1

        log(logging.INFO,
            f"[Model Load] FedAvg loaded: {loaded_count} keys, skipped: {skipped_count} keys")

    else:
        # FedBN + FedYogi 模式：只加载 federated params（非 BN）
        log(logging.INFO,
            f"[Model Load] FedBN mode: loading {len(array_keys)} federated params")

        # 在 FedBN 模式下需要 param_dict
        param_dict = dict(model.named_parameters())

        loaded_count = 0
        for name, array in record.items():
            if name not in param_keys:
                # 不是 parameter（可能是 buffer），在 FedBN 模式下跳过
                log(logging.WARNING, f"[Model Load] FedBN mode skipping non-param: {name}")
                continue

            tensor = torch.from_numpy(array.numpy())
            target = param_dict[name]

            if tensor.ndim == 1 and tensor.numel() == 1 and target.ndim == 0:
                tensor = tensor.reshape(())

            tensor = tensor.to(device=target.device, dtype=target.dtype)
            target.data.copy_(tensor)
            loaded_count += 1

        log(logging.INFO, f"[Model Load] FedBN loaded: {loaded_count} federated params")


def load_model_arrays(model, record, mode: str = "auto", strict: bool = True):
    """
    统一的模型 arrays 加载函数，自动检测加载模式。

    Args:
        model: PyTorch 模型
        record: Flower ArrayRecord
        mode: 加载模式
            - "auto": 自动检测（推荐）
            - "fedavg": FedAvg 模式（完整 state_dict）
            - "fedbn": FedBN 模式（仅 federated params）
        strict: 是否严格检查跨模式加载

    这是 load_federated_model_record 的增强版本，推荐使用。
    """
    return load_federated_model_record(model, record, strict_mode_check=strict)


def get_local_state_dict(model):
    """Capture BN params/buffers and all other non-federated state."""
    state_dict = model.state_dict()
    return {
        name: state_dict[name].detach().cpu().clone()
        for name in get_local_state_names(model)
    }


def load_local_state_dict(model, local_state):
    """Restore BN-local state into model without touching federated params."""
    state_dict = model.state_dict()
    for name, tensor in local_state.items():
        if name not in state_dict:
            raise KeyError(f"Local state '{name}' not found in model")
        target = state_dict[name]
        src = tensor
        if src.ndim == 1 and src.numel() == 1 and target.ndim == 0:
            src = src.reshape(())
        src = src.to(device=target.device, dtype=target.dtype)
        target.copy_(src)


def restore_client_local_state(
    model,
    client_id: int,
    run_id = None,  # int or str, optional
    round_num: int = 1
) -> bool:
    """
    Restore client-local FedBN state from persistent storage.

    Args:
        model: PyTorch 模型
        client_id: 客户端 ID
        run_id: 联邦学习运行标识（int 或 str，必须提供以确保状态唯一性）
        round_num: 当前轮次（用于日志和异常检测）

    Returns:
        True 如果成功恢复状态，False 如果没有找到状态
    """
    if run_id is None:
        # 兼容旧代码：如果没有 run_id，尝试使用旧的内存缓存
        log(logging.WARNING,
            f"[Client {client_id}] ⚠️ No run_id provided, falling back to legacy memory cache. "
            "This may cause state loss in parallel actor mode!")
        local_state = LOCAL_STATE_CACHE.get(client_id)
        if local_state is None:
            log(logging.INFO,
                f"[Client {client_id}] No cached FedBN local state found; using model defaults")
            return False
        load_local_state_dict(model, local_state)
        log(logging.INFO,
            f"[Client {client_id}] Restored FedBN local state with {len(local_state)} tensors (legacy cache)")
        return True

    # 使用新的持久化存储
    manager = get_fedbn_state_manager()
    local_state = manager.load_state(run_id, client_id, round_num)

    if local_state is None:
        return False

    load_local_state_dict(model, local_state)
    return True


def cache_client_local_state(
    model,
    client_id: int,
    run_id = None,  # int or str, optional
    round_num: int = 1
) -> Dict[str, Any]:
    """
    Persist client-local FedBN state to durable storage.

    Args:
        model: PyTorch 模型
        client_id: 客户端 ID
        run_id: 联邦学习运行标识（int 或 str，必须提供以确保状态唯一性）
        round_num: 当前轮次

    Returns:
        本地状态字典
    """
    local_state = get_local_state_dict(model)

    if run_id is None:
        # 兼容旧代码：如果没有 run_id，使用旧的内存缓存
        log(logging.WARNING,
            f"[Client {client_id}] ⚠️ No run_id provided, falling back to legacy memory cache. "
            "This may cause state loss in parallel actor mode!")
        LOCAL_STATE_CACHE[client_id] = local_state
        log(logging.INFO,
            f"[Client {client_id}] Cached FedBN local state with {len(local_state)} tensors (legacy cache)")
        return local_state

    # 使用新的持久化存储
    manager = get_fedbn_state_manager()
    manager.save_state(run_id, client_id, round_num, local_state)

    # 同时更新旧的内存缓存（保持兼容性）
    LOCAL_STATE_CACHE[client_id] = local_state

    return local_state

import torch

def build_detection_loss(nc, device):
    """Build a shared RT-DETR loss so train/eval stay aligned."""
    return RTDETRDetectionLoss(nc=nc, **LOSS_KWARGS).to(device)

def load_rtdetr_weights(target_model, weight_path):
    """
    通用加载函数：支持加载官方权重或微调过的权重。
    会自动忽略形状不匹配的层（例如类别数不同的 Head 层）。
    """
    print(f"📂 正在加载权重文件: {weight_path}")

    # 1. 加载文件
    # map_location='cpu' 防止显存爆炸，之后再 .to(device)
    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

    # 2. 智能提取权重 (优先使用 EMA，因为推理效果更好)
    if 'ema' in ckpt and ckpt['ema'] is not None:
        print("✨ 发现 EMA 权重，正在提取 (最佳推理性能)...")
        source_model = ckpt['ema']
    elif 'model' in ckpt:
        print("📦 提取常规 Model 权重...")
        source_model = ckpt['model']
    else:
        source_model = ckpt # 假设文件本身就是 state_dict

    # 获取 source state_dict
    # 如果 source_model 是一个完整模型对象(nn.Module)，取其 .state_dict()
    if hasattr(source_model, 'state_dict'):
        source_sd = source_model.state_dict()
    else:
        source_sd = source_model

    # 3. 准备目标 state_dict
    target_sd = target_model.state_dict()

    # 4. 核心逻辑：过滤匹配的权重
    # 只有当 Key 存在且 Shape 完全一致时才加载
    filtered_sd = {}
    mismatched_keys = []

    for k, v in source_sd.items():
        if k in target_sd:
            if v.shape == target_sd[k].shape:
                filtered_sd[k] = v
            else:
                mismatched_keys.append(f"{k} (源: {v.shape} vs 目标: {target_sd[k].shape})")

    # 5. 加载权重
    # strict=False 允许跳过不匹配的层
    target_model.load_state_dict(filtered_sd, strict=False)

    # 6. 打印报告
    print("=" * 40)
    print(f"🎉 成功加载参数: {len(filtered_sd)} / {len(target_sd)}")
    if len(mismatched_keys) > 0:
        print(f"⚠️ 跳过 {len(mismatched_keys)} 层 (通常是因为 nc 类别数不同):")
        # 只打印前3个跳过的层，避免刷屏
        for k in mismatched_keys[:3]:
            print(f"   - {k}")
        if len(mismatched_keys) > 3:
            print(f"   - ... 以及其他 {len(mismatched_keys)-3} 层")
    else:
        print("✅ 完美匹配，所有层均已加载！")
    print("=" * 40)

    return target_model

def rtdetr_collate_fn(batch):
    """
    Final Robust Collate Function for RT-DETR
    """
    pixel_values = []
    batch_bboxes = []
    batch_cls = []
    gt_groups = []
    batch_idx_list = []
    domain_id_list = []

    for i, item in enumerate(batch):
        # 1. 图片堆叠
        pixel_values.append(item['pixel_values'])

        # 2. 标签处理
        # item['labels'] 应该是 apply_transforms 返回的 list
        # 格式: [[class, x, y, w, h], ...]
        labels = item['labels']
        domain_value = item.get('domain_id', item.get('domain_label', item.get('domain', None)))
        if domain_value is not None:
            if isinstance(domain_value, str):
                domain_map = {'lung': 0, 'dental': 1, 'bone': 2, 'general': 3}
                domain_id_list.append(domain_map.get(domain_value.strip().lower(), -1))
            else:
                domain_id_list.append(int(domain_value))
        num_gt = len(labels)
        gt_groups.append(num_gt)

        if num_gt > 0:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # 这里的转换非常关键
            cls = labels_tensor[:, 0].long()   # 必须是 Long
            bboxes = labels_tensor[:, 1:]      # 必须是 Float32

            # 检查坐标归一化 (防呆设计)
            if bboxes.max() > 1.0 + 1e-6:
                print(f"⚠️ Warning in Collate: Found bbox > 1.0 in image {i}, normalizing...")
                bboxes = torch.clamp(bboxes, 0.0, 1.0)

            batch_cls.append(cls)
            batch_bboxes.append(bboxes)

            # ⚠️ 关键修复：batch_idx 必须是 Long (int64) 类型
            # 形状要和 num_gt 一致
            b_idx = torch.full((num_gt,), i, dtype=torch.long)
            batch_idx_list.append(b_idx)

    # Stack images
    pixel_values = torch.stack(pixel_values, dim=0)

    # Flatten labels
    if len(batch_cls) > 0:
        targets_cls = torch.cat(batch_cls, dim=0)
        targets_bboxes = torch.cat(batch_bboxes, dim=0)
        targets_batch_idx = torch.cat(batch_idx_list, dim=0)
    else:
        # 空 Batch 处理
        targets_cls = torch.zeros(0, dtype=torch.long)
        targets_bboxes = torch.zeros(0, 4, dtype=torch.float32)
        targets_batch_idx = torch.zeros(0, dtype=torch.long)

    # 完整性校验 (Catch the error BEFORE model forward)
    assert len(targets_cls) == len(targets_bboxes) == len(targets_batch_idx) == sum(gt_groups), \
        "Data Mismatch! The collate function failed to align targets."

    result = {
        'images': pixel_values,
        'cls': targets_cls,
        'bboxes': targets_bboxes,
        'gt_groups': gt_groups,
        'batch_idx': targets_batch_idx
    }
    if len(domain_id_list) == len(batch):
        result['domain_id'] = torch.tensor(domain_id_list, dtype=torch.long)
    return result

fds = None  # Cache FederatedDataset
# Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
pytorch_transforms = A.Compose([
    A.LongestMaxSize(max_size=640),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, fill=(114, 114, 114)),
    A.Normalize(mean=(0,0,0), std=(1,1,1)), # 相当于 / 255.0
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     batch["pixel_values"] = [pytorch_transforms(img.convert("RGB")) for img in batch["image"]]
#     return batch

def apply_transforms(batch):
    """
    适配 Albumentations 的 transform 函数。
    输入 batch['labels'] 格式: List[List[class_id, x, y, w, h]]
    """
    pixel_values = []
    new_labels = []

    for img, labels in zip(batch["image"], batch["labels"]):
        # 1. 图片转 Numpy (Albumentations 不接受 PIL)
        image_np = np.array(img.convert("RGB"))

        # 2. 标签格式转换
        # 你的数据: [class, x, y, w, h]
        # Albumentations 需要将 class_id 分离，或者放在最后 [x, y, w, h, class_id]
        # 我们这里采用把 BBox 和 Class 分离传入的方式，最稳妥
        bboxes_only = []
        class_labels = []

        for item in labels:
            c, x, y, w, h = item
            # 将 YOLO bbox 按边界裁剪，避免贴边框因浮点误差出现 x_min/y_min < 0
            x = min(max(float(x), 0.0), 1.0)
            y = min(max(float(y), 0.0), 1.0)
            w = min(max(float(w), 0.0), 1.0)
            h = min(max(float(h), 0.0), 1.0)
            x1 = max(0.0, x - w / 2.0)
            y1 = max(0.0, y - h / 2.0)
            x2 = min(1.0, x + w / 2.0)
            y2 = min(1.0, y + h / 2.0)
            w = x2 - x1
            h = y2 - y1
            if w <= 1e-6 or h <= 1e-6:
                continue
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0

            bboxes_only.append([x, y, w, h, c]) # 临时把 c 放在最后传给 transform

        # 3. 执行变换
        # 注意: 必须使用关键字参数 image=, bboxes=
        try:
            transformed = pytorch_transforms(image=image_np, bboxes=bboxes_only)
        except ValueError as e:
            # 如果某个 bbox 数据严重错误导致变换失败，捕获异常并跳过该图（或打印警告）
            print(f"⚠️ Transform failed for an image: {e}")
            # 返回全黑图和空标签防止程序崩溃
            pixel_values.append(torch.zeros(3, 640, 640))
            new_labels.append([])
            continue

        # 4. 提取结果
        transformed_image = transformed["image"] # 已经是 Tensor
        transformed_bboxes = transformed["bboxes"] # 变换后的 bbox

        # 5. 将标签还原回 [class, x, y, w, h] 格式供 collate_fn 使用
        final_labels = []
        for bbox in transformed_bboxes:
            # Albumentations 返回的格式与传入一致: [x, y, w, h, c]
            x, y, w, h, c = bbox
            final_labels.append([c, x, y, w, h])

        pixel_values.append(transformed_image)
        new_labels.append(final_labels)

    # 更新 batch
    batch["pixel_values"] = pixel_values
    batch["labels"] = new_labels
    if "domain_id" not in batch and "domain_label" not in batch and "domain" not in batch:
        default_domain_id = DATASET_CONFIGS[CURRENT_DATASET].get("domain_id", -1)
        batch["domain_id"] = [default_domain_id] * len(pixel_values)
    return batch

# def load_data(partition_id: int, num_partitions: int):
#     """Load partition data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         # partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="partition_cls",
#         #                            alpha=0.5, min_partition_size=10,
#         #                            self_balancing=True)
#         fds_train = FederatedDataset(
#             dataset="datasets/coco8/coco8_load_train.py",
#             partitioners={"train": partitioner},
#             data_dir="datasets/VOC2007",
#             # data_dir="datasets/local",
#             trust_remote_code=True,
#             # download_mode="force_redownload"
#         )

#         fds_test = FederatedDataset(
#             dataset="datasets/coco8/coco8_load_test.py",
#             partitioners={"test": partitioner},
#             data_dir="datasets/VOC2007",
#             trust_remote_code=True,
#         )

#     partition_train = fds_train.load_partition(partition_id)
#     partition_test = fds_test.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train = partition_train.train_test_split(test_size=0.01, seed=42)
#     partition_test = partition_test.train_test_split(test_size=0.99, seed=42)
#     # Construct dataloaders
#     partition_train = partition_train.with_transform(apply_transforms)
#     partition_test = partition_test.with_transform(apply_transforms)
#     trainloader = DataLoader(
#         partition_train["train"],
#         batch_size=4,
#         shuffle=True,
#         collate_fn=rtdetr_collate_fn
#     )
#     testloader = DataLoader(
#         partition_test["test"],
#         batch_size=4,
#         shuffle=False,
#         collate_fn=rtdetr_collate_fn
#     )
#     return trainloader, testloader


class _Medical5YoloDataset(torch.utils.data.Dataset):
    """Direct YOLO-folder dataset for one MedCoData detection client."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, data_dir: str, split: str, domain_id: int):
        self.data_dir = Path(data_dir)
        self.split = split
        self.domain_id = int(domain_id)
        self.image_dir = self.data_dir / "images" / split
        self.label_dir = self.data_dir / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing medical5 image dir: {self.image_dir}")
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def _read_labels(self, image_path: Path):
        label_path = self.label_dir / f"{image_path.stem}.txt"
        labels = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    labels.append([float(parts[0]), *map(float, parts[1:5])])
                except ValueError:
                    continue
        return labels

    def __getitem__(self, idx: int):
        from PIL import Image
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        labels = self._read_labels(image_path)
        batch = apply_transforms({
            "image": [image],
            "labels": [labels],
            "domain_id": [self.domain_id],
        })
        return {
            "pixel_values": batch["pixel_values"][0],
            "labels": batch["labels"][0],
            "domain_id": self.domain_id,
            "image_path": str(image_path),
        }


def _load_medical5_data(partition_id: int, num_partitions: int, config: dict, nc: int):
    clients = config.get("clients", [])
    if num_partitions != len(clients):
        raise ValueError(
            f"{CURRENT_DATASET} requires exactly {len(clients)} supernodes/clients, "
            f"but got num_partitions={num_partitions}. Use --federation-config \"options.num-supernodes=5\"."
        )
    if partition_id < 0 or partition_id >= len(clients):
        raise ValueError(f"Invalid medical5 partition_id={partition_id}; expected 0..{len(clients)-1}")

    client_cfg = clients[partition_id]
    data_dir = client_cfg["data_dir"]
    domain_id = client_cfg.get("domain_id", partition_id)
    client_name = client_cfg.get("name", f"client{partition_id}")

    trainset = _Medical5YoloDataset(data_dir, "train", domain_id)
    valset = _Medical5YoloDataset(data_dir, "val", domain_id)
    testset = _Medical5YoloDataset(data_dir, "test", domain_id)

    def seed_worker(worker_id):
        worker_seed = PARTITION_SEED + partition_id * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_generator = torch.Generator().manual_seed(PARTITION_SEED + partition_id * 1000)
    eval_generator = torch.Generator().manual_seed(PARTITION_SEED + partition_id * 1000 + 1)
    trainloader = DataLoader(
        trainset, batch_size=16, shuffle=True, collate_fn=rtdetr_collate_fn,
        generator=train_generator, worker_init_fn=seed_worker,
    )
    valloader = DataLoader(
        valset, batch_size=8, shuffle=False, collate_fn=rtdetr_collate_fn,
        generator=eval_generator, worker_init_fn=seed_worker,
    )
    testloader = DataLoader(
        testset, batch_size=8, shuffle=False, collate_fn=rtdetr_collate_fn,
        generator=eval_generator, worker_init_fn=seed_worker,
    )

    log(logging.INFO,
        f"[Client {partition_id}] medical5 domain={client_name}, "
        f"train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}, nc={nc}")
    return trainloader, valloader, testloader, nc

# 缓存对象（train / val / test 各自独立）
fds_train, fds_val, fds_test = None, None, None

def load_data(partition_id: int, num_partitions: int):
    """
    加载分区数据。
    train / val / test 各自来自独立的 FederatedDataset，不做二次切分。
    返回 (trainloader, valloader, testloader, nc)
    """
    global fds_train, fds_val, fds_test

    config = DATASET_CONFIGS[CURRENT_DATASET]
    nc = config["nc"]

    if CURRENT_DATASET in {
        "medical5_detect",
        "medical5_mixed_labeled20",
        "medical5_mixed_labeled20_tune30",
        "medical5_full33877_mixed_labeled20_712",
        "medical5_full33877_branch_a_random",
        "medical5_full33877_branch_b_boxbalanced",
        "medical5_detection_clientfirst_seed42",
        "medical5_detection_clientfirst_rebalanced_seed20260719",
        "cbis_ddsm_clientfirst_seed42",
        "tbx11k_clientfirst_seed42",
        "totalsegmentator_clientfirst_seed42",
    }:
        return _load_medical5_data(partition_id, num_partitions, config, nc)

    if fds_train is None or fds_val is None or fds_test is None:
        random.seed(PARTITION_SEED)
        np.random.seed(PARTITION_SEED)
        torch.manual_seed(PARTITION_SEED)
        partitioner_train = IidPartitioner(num_partitions=num_partitions)

        random.seed(PARTITION_SEED)
        np.random.seed(PARTITION_SEED)
        torch.manual_seed(PARTITION_SEED)
        partitioner_val = IidPartitioner(num_partitions=num_partitions)

        random.seed(PARTITION_SEED)
        np.random.seed(PARTITION_SEED)
        torch.manual_seed(PARTITION_SEED)
        partitioner_test = IidPartitioner(num_partitions=num_partitions)

        fds_train = FederatedDataset(
            dataset=config["train_loader"],
            partitioners={"train": partitioner_train},
            data_dir=config["data_dir"],
            trust_remote_code=True,
        )
        fds_val = FederatedDataset(
            dataset=config["val_loader"],
            partitioners={"validation": partitioner_val},
            data_dir=config["data_dir"],
            trust_remote_code=True,
        )
        fds_test = FederatedDataset(
            dataset=config["test_loader"],
            partitioners={"test": partitioner_test},
            data_dir=config["data_dir"],
            trust_remote_code=True,
        )

    partition_train = fds_train.load_partition(partition_id, split="train")
    partition_val   = fds_val.load_partition(partition_id, split="validation")
    partition_test  = fds_test.load_partition(partition_id, split="test")

    partition_train = partition_train.with_transform(apply_transforms)
    partition_val   = partition_val.with_transform(apply_transforms)
    partition_test  = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=16, shuffle=True,  collate_fn=rtdetr_collate_fn)
    valloader   = DataLoader(partition_val,   batch_size=8, shuffle=False, collate_fn=rtdetr_collate_fn)
    testloader  = DataLoader(partition_test,  batch_size=8, shuffle=False, collate_fn=rtdetr_collate_fn)

    log(logging.INFO,
        f"[Client {partition_id}] Dataset sizes — "
        f"train: {len(partition_train)}, val: {len(partition_val)}, test: {len(partition_test)}")

    return trainloader, valloader, testloader, nc

from .myutils import visualize_batch

def train(net, trainloader, epochs, lr, device, nc=None, moe_domain_supervision=True, max_batches=0):
    """
    Train the model on the training set using RT-DETR loss.
    Returns: avg_trainloss (float)

    [新增] nc 参数: 类别数，如果不传则从配置字典获取
    """
    net.train()

    # [新增] 获取类别数
    if nc is None:
        nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    criterion = build_detection_loss(nc=nc, device=device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=0.0001,
    )

    total_loss = 0.0
    num_batches = 0

    vis_debug_max = 3  # 只在首个 epoch 保存前3个批次
    vis_debug_done = 0

    for epoch in range(epochs):
        for batch_index, batch in enumerate(trainloader):
            if max_batches and batch_index >= int(max_batches):
                break
            if not moe_domain_supervision:
                # The fixed client-first split mixes all five sources inside every
                # client.  A client id is therefore not a valid anatomy-domain id.
                # Keep task-driven sparse routing but disable the incorrect
                # supervised router target.
                batch.pop("domain_id", None)
                batch.pop("domain_label", None)
                batch.pop("domain", None)

            if epoch == 0 and vis_debug_done < vis_debug_max:
                visualize_batch(
                    batch,
                    save_dir="logs/vis",
                    prefix=f"epoch{epoch}_batch{vis_debug_done}",
                    class_names=None,  # 若有类别名列表可填上
                    max_images=None
                )
                vis_debug_done += 1

            # 1. 数据迁移
            images = batch['images'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device) # <--- 关键：CDN需要
            if 'domain_id' in batch and torch.is_tensor(batch['domain_id']):
                batch['domain_id'] = batch['domain_id'].to(device)

            optimizer.zero_grad()

            # 2. Forward (传入 batch 用于生成去噪锚框)
            outputs = net(images, batch=batch)

            # 3. Loss 计算
            # 解包: x = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs

            if enc_bboxes is not None:
                enc_bboxes = enc_bboxes.unsqueeze(0)
                enc_scores = enc_scores.unsqueeze(0)

            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta
            )

            # print(f"losses: {loss_dict.values()}")

            # 4. Backward
            loss = sum(loss_dict.values())
            aux_loss = net.get_aux_loss() if hasattr(net, 'get_aux_loss') else None
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()

            # 最佳实践：梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_trainloss = total_loss / max(num_batches, 1)
    return avg_trainloss


# def test(net, testloader, device):
#     """
#     Validate the model.
#     Returns: loss (float), map50-95 (float)
#     """
#     net.to(device)
#     net.eval()

#     criterion = RTDETRDetectionLoss(nc=20, use_vfl=True).to(device)
#     metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox").to(device)

#     total_loss = 0.0
#     num_batches = 0

#     with torch.no_grad():
#         for batch in testloader:
#             images = batch['images'].to(device)
#             batch['cls'] = batch['cls'].to(device)
#             batch['bboxes'] = batch['bboxes'].to(device)
#             # Eval 模式不需要 batch_idx 用于 CDN，但如果代码没改干净，移过去也无妨

#             # 1. Forward
#             # Eval模式下，你的forward返回 (y, x)
#             # y: (BS, 300, 4+nc) 推理结果
#             # x: (dec_bboxes, ...) 原始输出用于算 Loss
#             outputs = net(images, batch=batch)

#             # 确保我们拿到了正确的输出
#             if isinstance(outputs, tuple) and len(outputs) == 2:
#                 inference_out, raw_out = outputs
#                 dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw_out

#                 if enc_bboxes is not None:
#                     enc_bboxes = enc_bboxes.unsqueeze(0)
#                     enc_scores = enc_scores.unsqueeze(0)

#             else:
#                 # 兼容性 fallback，防止 return 结构变化
#                 continue

#             # 2. 计算 Validation Loss
#             loss_dict = criterion(
#                 preds=(dec_bboxes, dec_scores),
#                 batch=batch,
#                 dn_bboxes=enc_bboxes,
#                 dn_scores=enc_scores,
#                 dn_meta=dn_meta
#             )
#             total_loss += sum(loss_dict.values()).item()
#             num_batches += 1

#             # 3. 计算 mAP (Best Practice)
#             # 需要将 flattened 的 targets 还原回 per-image 格式
#             target_list = []
#             current_idx = 0
#             for num_gt in batch['gt_groups']:
#                 if num_gt > 0:
#                     t_boxes = batch['bboxes'][current_idx : current_idx + num_gt]
#                     t_labels = batch['cls'][current_idx : current_idx + num_gt]
#                     target_list.append(dict(boxes=t_boxes, labels=t_labels))
#                     current_idx += num_gt
#                 else:
#                     target_list.append(dict(boxes=torch.empty(0, 4, device=device), labels=torch.empty(0, device=device)))

#             # 解析预测结果 inference_out: (BS, 300, 4+nc)
#             # 格式通常是 [cx, cy, w, h, class_probs...] 或者 [cx, cy, w, h, max_score, class_id]
#             # 根据你 forward 的最后一行： torch.cat((bboxes, scores), -1)
#             # 前4位是 bbox, 后面是 scores
#             pred_list = []
#             bs = inference_out.shape[0]
#             for i in range(bs):
#                 pred_item = inference_out[i]
#                 p_boxes = pred_item[:, :4] # cx, cy, w, h
#                 p_scores = pred_item[:, 4:] # 80个类别的分数

#                 # 获取每个 box 的最大分数和对应类别
#                 scores, labels = p_scores.max(dim=-1)

#                 pred_list.append(dict(
#                     boxes=p_boxes,
#                     scores=scores,
#                     labels=labels
#                 ))

#             # 更新指标状态
#             metric.update(pred_list, target_list)

#     # 计算最终 mAP
#     metrics_dict = metric.compute()
#     map50 = metrics_dict['map_50'].item() # map 默认就是 mAP 50-95
#     avg_loss = total_loss / max(num_batches, 1)

#     # 这里的 map50_95 替代了原本的 accuracy
#     return avg_loss, map50

def test(net, testloader, device, nc=None, client_id=None, split_name="val", moe_domain_supervision=True, max_batches=0):
    """
        Validate the model.
        自动寻找最佳 F1 阈值。
        Returns:
            loss (float), map50 (float),
            best_precision (float), best_recall (float),
            best_f1 (float), best_threshold (float)

        [新增] nc 参数: 类别数，如果不传则从配置字典获取
        [新增] client_id, split_name: 用于诊断日志
    """
    net.to(device)
    net.eval()

    # [新增] 获取类别数
    if nc is None:
        nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # --- 诊断：统计 val/test 集基本信息 ---
    total_images = len(testloader.dataset) if hasattr(testloader, 'dataset') else -1
    log(logging.INFO,
        f"[Client {client_id}] [{split_name}] 开始评估 — 图片总数: {total_images}, nc={nc}")

    if total_images == 0:
        log(logging.WARNING,
            f"[Client {client_id}] [{split_name}] 评估集为空，跳过，返回 0.0")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    criterion = build_detection_loss(nc=nc, device=device)
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox").to(device)

    total_loss = 0.0
    num_batches = 0

    # 存储所有预测结果用于后续计算最佳阈值
    # 格式: list of tensors [score, is_tp]
    # is_tp: 1 if TP, 0 if FP
    pred_stats = []
    total_gt_count = 0 # 整个数据集的 GT 总数

    with torch.no_grad():
        for batch_index, batch in enumerate(testloader):
            if max_batches and batch_index >= int(max_batches):
                break
            if not moe_domain_supervision:
                batch.pop("domain_id", None)
                batch.pop("domain_label", None)
                batch.pop("domain", None)
            images = batch['images'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            if 'domain_id' in batch and torch.is_tensor(batch['domain_id']):
                batch['domain_id'] = batch['domain_id'].to(device)

            # 1. Forward
            outputs = net(images, batch=batch)

            if isinstance(outputs, tuple) and len(outputs) == 2:
                inference_out, raw_out = outputs
                dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw_out
                if enc_bboxes is not None:
                    enc_bboxes = enc_bboxes.unsqueeze(0)
                    enc_scores = enc_scores.unsqueeze(0)
            else:
                continue

            # 2. Loss
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta
            )
            total_loss += sum(loss_dict.values()).item()
            num_batches += 1

            # 3. 准备数据
            target_list = []
            current_idx = 0
            for num_gt in batch['gt_groups']:
                if num_gt > 0:
                    t_boxes = batch['bboxes'][current_idx : current_idx + num_gt]
                    t_labels = batch['cls'][current_idx : current_idx + num_gt]
                    target_list.append(dict(boxes=t_boxes, labels=t_labels))
                    current_idx += num_gt
                    total_gt_count += num_gt # 累计 GT 总数
                else:
                    target_list.append(dict(boxes=torch.empty(0, 4, device=device), labels=torch.empty(0, device=device)))

            pred_list = []
            bs = inference_out.shape[0]

            # --- 处理每一张图片，收集 TP/FP 状态 ---
            for i in range(bs):
                # A. 解析预测
                pred_item = inference_out[i]
                p_boxes = pred_item[:, :4]
                p_scores_all = pred_item[:, 4:]
                scores, labels = p_scores_all.max(dim=-1)

                # B. 加入 metric 更新队列 (计算 mAP 用)
                pred_list.append(dict(boxes=p_boxes, scores=scores, labels=labels))

                # C. 手动匹配逻辑 (计算 Best F1 用)
                # 即使分数很低也要保留，因为我们要画完整曲线，但为了显存可以设个极低门槛
                keep_mask = scores > 0.001
                filter_boxes = p_boxes[keep_mask]
                filter_scores = scores[keep_mask]
                filter_labels = labels[keep_mask]

                gt_boxes = target_list[i]['boxes']
                gt_labels = target_list[i]['labels'].long().flatten()

                # 如果没有预测框
                if len(filter_boxes) == 0:
                    continue

                # 记录当前图片的预测状态: [score, 0/1]
                # 默认为 FP (0)
                matches = torch.zeros(len(filter_boxes), device=device)

                if len(gt_boxes) > 0:
                    # 转换坐标 cxcywh -> xyxy
                    p_xyxy = box_convert(filter_boxes, in_fmt='cxcywh', out_fmt='xyxy')
                    g_xyxy = box_convert(gt_boxes, in_fmt='cxcywh', out_fmt='xyxy')

                    # 计算 IoU: [N_pred, M_gt]
                    iou_matrix = box_iou(p_xyxy, g_xyxy)
                    # A true positive must have both IoU > 0.5 and the same
                    # class.  The previous implementation matched across
                    # classes and could inflate P/R/F1.
                    class_match = filter_labels[:, None] == gt_labels[None, :]
                    iou_matrix = iou_matrix.masked_fill(~class_match, -1.0)

                    # 按分数从高到低排序预测框，进行贪婪匹配
                    sort_idx = torch.argsort(filter_scores, descending=True)
                    iou_matrix = iou_matrix[sort_idx]

                    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

                    # 这是一个临时的 vector 对应排序后的 indices
                    sorted_matches = torch.zeros(len(filter_boxes), device=device)

                    for p_idx in range(len(filter_boxes)):
                        # 找到该预测框最大 IoU 的 GT
                        iou_val, g_idx = iou_matrix[p_idx].max(0)

                        if iou_val > 0.5 and not gt_matched[g_idx]:
                            gt_matched[g_idx] = True
                            sorted_matches[p_idx] = 1.0 # 标记为 TP

                    # 恢复原来的顺序 (或者直接存排序后的 score 也行，这里为了简单直接存排序后的)
                    # 我们只需要 score 和是否 TP 的对应关系
                    pred_stats.append(torch.stack((filter_scores[sort_idx], sorted_matches), dim=1))

                else:
                    # 图片没有 GT，所有预测框都是 FP (matches 全 0)
                    pred_stats.append(torch.stack((filter_scores, matches), dim=1))

            metric.update(pred_list, target_list)

    # --- Loop 结束，开始计算指标 ---

    # 诊断：GT 总数
    log(logging.INFO,
        f"[Client {client_id}] [{split_name}] num_batches={num_batches}, "
        f"total_gt_boxes={total_gt_count}, total_images={total_images}")

    # 1. 计算标准 mAP
    metrics_dict = metric.compute()
    map50_raw = metrics_dict['map_50'].item()
    # NaN 保护：torchmetrics 在没有 GT 或没有预测时返回 NaN/-1
    map50 = map50_raw if (map50_raw == map50_raw and map50_raw >= 0) else 0.0
    avg_loss = total_loss / max(num_batches, 1)

    # NaN 保护：loss 异常检测
    if avg_loss != avg_loss or avg_loss < 0:
        log(logging.WARNING,
            f"[Client {client_id}] [{split_name}] avg_loss={avg_loss} 异常，强制置 0.0")
        avg_loss = 0.0

    # 2. 计算最佳 F1 及其对应的 P, R, Threshold
    if len(pred_stats) > 0:
        # 拼接所有 batch 的数据: shape [N_total_preds, 2]
        all_stats = torch.cat(pred_stats, dim=0)

        # 按分数从高到低排序
        sorted_indices = torch.argsort(all_stats[:, 0], descending=True)
        sorted_stats = all_stats[sorted_indices]

        all_scores = sorted_stats[:, 0]
        all_tps = sorted_stats[:, 1] # 1.0 or 0.0

        # 向量化计算累积 TP 和 FP
        # cumsum 告诉我们在当前 index (即当前阈值) 下，有多少个 TP 和 FP
        tp_cumsum = torch.cumsum(all_tps, dim=0)
        fp_cumsum = torch.cumsum(1 - all_tps, dim=0)

        # 防止除零
        eps = 1e-7

        # 计算 Precision 和 Recall 曲线
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        recall_curve = tp_cumsum / (total_gt_count + eps)

        # 计算 F1 曲线
        f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + eps)

        # 找到 F1 最大的位置
        best_idx = torch.argmax(f1_curve)

        best_f1 = f1_curve[best_idx].item()
        best_precision = precision_curve[best_idx].item()
        best_recall = recall_curve[best_idx].item()
        best_threshold = all_scores[best_idx].item()

    else:
        # 防止验证集为空或没有预测框的极端情况
        best_f1, best_precision, best_recall, best_threshold = 0.0, 0.0, 0.0, 0.0

    log(logging.INFO,
        f"[Client {client_id}] [{split_name}] RESULT — "
        f"loss={avg_loss:.4f}, map50={map50:.4f}, "
        f"P={best_precision:.4f}, R={best_recall:.4f}, F1={best_f1:.4f}, "
        f"thr={best_threshold:.4f}, gt_boxes={total_gt_count}")
    print(f"[Client {client_id}] [{split_name}] Best F1: {best_f1:.4f} @ Threshold: {best_threshold:.4f} "
          f"(P={best_precision:.4f}, R={best_recall:.4f}, map50={map50:.4f}, loss={avg_loss:.4f})")

    return avg_loss, map50, best_precision, best_recall, best_f1, best_threshold


# def train(net, trainloader, epochs, lr, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["images"].to(device)
#             labels = batch["labels"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     net.eval()
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["images"].to(device)
#             labels = batch["labels"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy
