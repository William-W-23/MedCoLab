"""FL: A Flower / PyTorch app."""

import logging
import os
import random
import numpy as np
import torch
from pathlib import Path
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

# ============================================================================
# 模型选择配置 - 必须与 server_app.py 保持一致！
# ============================================================================
# 模式 A: RTDETR_L（标准版，无 RetNet）
# 模式 B: RTDETR_L_WithAttention（带 RetNet Attention）
# 模式 C: RTDETR_L_WithASEM（解剖感知稀疏 MoE）

# 通过环境变量切换模型，必须和 server_app.py 的 FL_MODEL_VARIANT 一致。
from models import RTDETR_L, RTDETR_L_WithAttention, RTDETR_L_WithASEM

_MODEL_VARIANTS = {
    "RTDETR_L": RTDETR_L,
    "RTDETR_L_WithAttention": RTDETR_L_WithAttention,
    "RTDETR_L_WithASEM": RTDETR_L_WithASEM,
}
FL_MODEL_VARIANT = os.environ.get("FL_MODEL_VARIANT", "RTDETR_L_WithASEM")
if FL_MODEL_VARIANT not in _MODEL_VARIANTS:
    raise ValueError(f"Unsupported FL_MODEL_VARIANT={FL_MODEL_VARIANT!r}; choose one of {sorted(_MODEL_VARIANTS)}")
Net = _MODEL_VARIANTS[FL_MODEL_VARIANT]

from fl.detection_task import (
    get_federated_model_record,
    restore_client_local_state,
    cache_client_local_state,
    load_federated_model_record,
    load_model_arrays,
    describe_fedbn_layout,
    get_fedbn_state_manager,
    validate_arrays_model_compatibility,
    format_compatibility_error,
    extract_model_signature,
    get_mode_signature,
    set_current_mode,
    get_current_mode,
    DATASET_CONFIGS,
    CURRENT_DATASET,
    FEDBN_STATE_DIR,
)

from fl.detection_task import load_data

from fl.detection_task import test as test_fn
from fl.detection_task import train as train_fn

# Flower ClientApp
app = ClientApp()


def seed_client(partition_id: int, round_num: int) -> int:
    seed = int(os.environ.get("MASTER_SEED", "42")) + partition_id * 1000 + round_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def validate_mode_compatibility(
    msg: Message,
    context: Context,
    model,
    partition_id: int
) -> dict:
    """
    验证 server 发送的模式签名与 client 配置是否一致。

    Returns:
        config_dict: 解析后的配置字典
    """
    config = msg.content.get("config", {})

    # 解析 server 发送的模式信息
    server_mode_sig = config.get("mode_signature", "")
    server_model_variant = config.get("model_variant", "")
    server_strategy = config.get("strategy", "")
    server_dataset = config.get("dataset", "")
    server_nc = config.get("nc", 0)
    server_use_fedbn = config.get("use_fedbn", False)

    # Client 当前配置
    client_model_variant = Net.__name__
    client_dataset = CURRENT_DATASET
    client_nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # 构造 client 模式签名
    client_mode_sig = get_mode_signature(
        model_variant=client_model_variant,
        strategy="unknown",  # client 不知道具体策略
        dataset=client_dataset,
        nc=client_nc
    )

    # 模型签名
    model_sig = extract_model_signature(model)

    log(logging.INFO,
        f"[Client {partition_id}] Mode validation: "
        f"server_sig={server_mode_sig}, client_model={client_model_variant}, "
        f"model_sig={model_sig}")

    # 检测模型不匹配
    if server_model_variant and server_model_variant != client_model_variant:
        log(logging.ERROR,
            f"[Client {partition_id}] ⚠️ MODEL VARIANT MISMATCH! "
            f"Server: {server_model_variant}, Client: {client_model_variant}")
        log(logging.ERROR,
            f"[Client {partition_id}] This will cause KeyError during model loading!")
        log(logging.ERROR,
            f"[Client {partition_id}] FIX: Update client_app.py import to match server_app.py")

    # 检测 nc 不匹配
    if server_nc and server_nc != client_nc:
        log(logging.WARNING,
            f"[Client {partition_id}] ⚠️ NC mismatch: server={server_nc}, client={client_nc}")

    return {
        "lr": config.get("lr", 1e-4),
        "mode_signature": server_mode_sig,
        "model_variant": server_model_variant,
        "strategy": server_strategy,
        "use_fedbn": server_use_fedbn,
        "nc": server_nc,
    }


def get_fedbn_state_dir_for_mode(run_id, mode_signature: str) -> Path:
    """
    根据 mode_signature 创建隔离的 FedBN 状态目录。
    防止不同模式的 FedBN 状态串用。
    """
    # 将 mode_signature 转换为安全的目录名
    safe_signature = mode_signature.replace("|", "_").replace(":", "_")
    base_dir = FEDBN_STATE_DIR / safe_signature
    return base_dir / str(run_id)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Get run_id from context for persistent state management
    run_id = context.run_id if hasattr(context, 'run_id') else str(context.node_id)
    round_num = msg.metadata.round if hasattr(msg.metadata, 'round') else 1
    partition_id = context.node_config["partition-id"]
    seed_client(partition_id, round_num)

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # Load the model and initialize it with the received weights
    model = Net(nc=nc)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]

    # ============================================================================
    # 模式验证
    # ============================================================================
    config = validate_mode_compatibility(msg, context, model, partition_id)

    # 设置当前模式（用于日志和状态管理）
    if config["mode_signature"]:
        set_current_mode(config["mode_signature"])

    log(logging.INFO,
        f"[Client {partition_id}] Starting train in round {round_num}, "
        f"run_id={run_id}, node_id={context.node_id}, "
        f"use_fedbn={config['use_fedbn']}")

    # ============================================================================
    # 加载 federated arrays（使用增强的兼容性检测）
    # ============================================================================
    arrays = msg.content["arrays"]

    # 预检测兼容性（在加载前）
    array_keys = set(arrays.keys())
    is_compatible, diagnostic = validate_arrays_model_compatibility(model, array_keys)

    if not is_compatible:
        log(logging.WARNING,
            f"[Client {partition_id}] Pre-load compatibility warning: "
            f"{len(diagnostic['missing_in_model'])} keys missing in model, "
            f"{len(diagnostic['mode_mismatch_indicators'])} mode mismatches")

    # 加载 arrays（增强版本，支持 buffer 和宽松模式）
    # FedAvg 模式下使用 strict=False 允许宽松加载
    # FedBN 模式下使用 strict=True 严格校验
    strict_mode = config["use_fedbn"]  # FedBN 模式下严格校验
    load_model_arrays(model, arrays, strict=strict_mode)

    # ============================================================================
    # FedBN local state 管理（仅 FedBN 模式）
    # ============================================================================
    if config["use_fedbn"]:
        # 使用按模式隔离的 FedBN 状态目录
        fedbn_run_dir = get_fedbn_state_dir_for_mode(run_id, config["mode_signature"])

        log(logging.INFO,
            f"[Client {partition_id}] FedBN state dir: {fedbn_run_dir}")

        restore_client_local_state(model, partition_id, run_id=run_id, round_num=round_num)

    log(logging.INFO,
        f"[Client {partition_id}] FedBN train load: "
        f"{describe_fedbn_layout(model)}")

    # load_data 返回 trainloader, valloader, testloader, nc
    trainloader, _, _, nc_data = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        config["lr"],
        device,
        nc=nc,
    )

    # ============================================================================
    # Save FedBN local state（仅 FedBN 模式）
    # ============================================================================
    if config["use_fedbn"]:
        cache_client_local_state(model, partition_id, run_id=run_id, round_num=round_num)

    # ============================================================================
    # 构造返回的 model_record（根据模式）
    # ============================================================================
    if config["use_fedbn"]:
        # FedBN 模式：只返回 federated params
        model_record = get_federated_model_record(model)
    else:
        # FedAvg 模式：返回完整 state_dict
        model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Get run_id from context for persistent state management
    run_id = context.run_id if hasattr(context, 'run_id') else str(context.node_id)
    round_num = msg.metadata.round if hasattr(msg.metadata, 'round') else 1
    partition_id = context.node_config["partition-id"]
    seed_client(partition_id, round_num)

    # Get the correct number of classes for the current dataset
    nc = DATASET_CONFIGS[CURRENT_DATASET]["nc"]

    # Load the model and initialize it with the received weights
    model = Net(nc=nc)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]

    # ============================================================================
    # 模式验证
    # ============================================================================
    config = validate_mode_compatibility(msg, context, model, partition_id)

    if config["mode_signature"]:
        set_current_mode(config["mode_signature"])

    log(logging.INFO,
        f"[Client {partition_id}] Starting evaluation in round {round_num}, "
        f"run_id={run_id}, node_id={context.node_id}, "
        f"use_fedbn={config['use_fedbn']}")

    # ============================================================================
    # 加载 federated arrays
    # ============================================================================
    arrays = msg.content["arrays"]
    strict_mode = config["use_fedbn"]
    load_model_arrays(model, arrays, strict=strict_mode)

    # ============================================================================
    # FedBN local state 恢复（仅 FedBN 模式）
    # ============================================================================
    if config["use_fedbn"]:
        restored = restore_client_local_state(model, partition_id, run_id=run_id, round_num=round_num)

        # 健壮性检查：如果 evaluate 时状态缺失但该 client 应该已经训练过，发出警告
        if not restored and round_num > 1:
            manager = get_fedbn_state_manager()
            if manager.state_exists(run_id, partition_id):
                log(logging.WARNING,
                    f"[Client {partition_id}] ⚠️ FedBN state file exists but load failed "
                    f"in evaluate round {round_num}")
            else:
                log(logging.WARNING,
                    f"[Client {partition_id}] ⚠️ FedBN state missing for evaluation at round {round_num}. "
                    f"Client may not have participated in training yet.")

    log(logging.INFO,
        f"[Client {partition_id}] FedBN eval load: "
        f"{describe_fedbn_layout(model)}")

    # load_data 返回 trainloader, valloader, testloader, nc
    _, valloader, _, nc_data = load_data(partition_id, num_partitions)

    # Call the evaluation function — 用 val 集，传入 client_id 便于诊断
    eval_loss, eval_map50, eval_precision, eval_recall, eval_f1, _ = test_fn(
        model,
        valloader,
        device,
        nc=nc,
        client_id=partition_id,
        split_name="val",
    )

    # NaN 保护：确保上报的 metrics 都是有效 float
    def safe(v):
        return float(v) if (v == v and v >= 0) else 0.0

    # Construct and return reply Message
    metrics = {
        "eval_loss": safe(eval_loss),
        "eval_map50": safe(eval_map50),
        "eval_precision": safe(eval_precision),
        "eval_recall": safe(eval_recall),
        "eval_f1": safe(eval_f1),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
