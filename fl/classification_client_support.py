"""Flower ClientApp for federated multi-dataset medical classification."""

import csv
import json
import logging
import os
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from torch.utils.data import DataLoader, WeightedRandomSampler

from fl.classification_task import (
    MultiDatasetRTDETRClassifier,
    Medical5ClassificationDataset,
    evaluate,
    seed_everything,
    sha256_file,
    train_one_round,
)

app = ClientApp()


def _worker_seed(worker_id):
    seed=torch.initial_seed()%(2**32); random.seed(seed); np.random.seed(seed)


@lru_cache(maxsize=4)
def _load_manifest(path_text, expected_sha, group_audit_unavailable=False):
    path=Path(path_text)
    if sha256_file(path)!=expected_sha: raise RuntimeError("stratified manifest SHA256 mismatch")
    data=json.loads(path.read_text())
    group_split_leaks=data.get("group_split_leaks")
    if group_split_leaks is None:
        if not group_audit_unavailable: raise RuntimeError("group audit unavailable in stratified manifest")
    elif group_audit_unavailable or int(group_split_leaks)!=0:
        raise RuntimeError("group leakage policy mismatch in stratified manifest")
    return data


def _select_smoke(records, maximum):
    if maximum<=0 or len(records)<=maximum: return records
    by={}
    for r in records: by.setdefault(r["class_name"],[]).append(r)
    selected=[]; i=0
    while len(selected)<maximum:
        changed=False
        for name in sorted(by):
            if i<len(by[name]): selected.append(by[name][i]); changed=True
            if len(selected)>=maximum: break
        if not changed: break
        i+=1
    return selected



@lru_cache(maxsize=2)
def _pcam_group_index(source_manifest_text):
    """Map a PCam source filename stem to its immutable slide/group id."""
    groups = {}
    with Path(source_manifest_text).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != "pcam" or row["split"] == "ssl":
                continue
            groups[Path(row["source_path"]).stem] = row["group_id"]
    return groups


def _pcam_group_sampler(records, source_manifest_text, generator):
    """Keep PCam total sampling mass unchanged while equalising train groups."""
    group_index = _pcam_group_index(source_manifest_text)
    pcam_groups = {}
    for record in records:
        if record["dataset"] != "pcam":
            continue
        match = re.match(r"pcam__(pcam_(?:train|test)_\d+)__", Path(record["path"]).name)
        if match and match.group(1) in group_index:
            record["pcam_group"] = group_index[match.group(1)]
            group = record["pcam_group"]
            pcam_groups[group] = pcam_groups.get(group, 0) + 1
    if len(pcam_groups) < 2:
        return None, {}
    target = sum(pcam_groups.values()) / len(pcam_groups)
    weights = [
        target / pcam_groups[record["pcam_group"]]
        if record.get("pcam_group") else 1.0
        for record in records
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records), replacement=True, generator=generator,
    )
    return sampler, dict(sorted(pcam_groups.items()))


def build_stratified_loader(config, client, subset, round_num):
    data_root=Path(str(config["classification_data_root"])); manifest=_load_manifest(
        str(config["classification_stratified_manifest"]),str(config["classification_stratified_manifest_sha256"]),
        bool(config.get("classification_group_audit_unavailable",False)))
    allowed=set(manifest["clients"][str(client)][f"{subset}_paths"])
    datasets=[Medical5ClassificationDataset(data_root/f"client{client}",s,int(config["classification_image_size"]),train=subset=="train") for s in ("train","val","test")]
    dataset=datasets[0]; dataset.records=[r for d in datasets for r in d.records if str(Path(r["path"]).relative_to(data_root)) in allowed]
    maximum=int(config.get("classification_train_max_samples" if subset=="train" else "classification_eval_max_samples",0))
    dataset.records=_select_smoke(dataset.records,maximum); dataset.class_hist=Counter(r["class_name"] for r in dataset.records)
    if maximum<=0 and len(dataset)!=len(allowed): raise RuntimeError(f"manifest path mismatch client{client} {subset}: {len(dataset)} != {len(allowed)}")
    seed=int(config["classification_master_seed"])+client*10000+round_num+(0 if subset=="train" else 100000)
    workers=int(config.get("classification_num_workers",2)); generator=torch.Generator().manual_seed(seed)
    sampler = None
    pcam_group_hist = {}
    if subset == "train" and bool(config.get("classification_pcam_group_balanced", False)):
        sampler, pcam_group_hist = _pcam_group_sampler(dataset.records, manifest["source_manifest"], generator)
    loader=DataLoader(dataset,batch_size=int(config["classification_batch_size"] if subset=="train" else config["classification_eval_batch_size"]),
        shuffle=subset=="train" and sampler is None,sampler=sampler,num_workers=workers,pin_memory=torch.cuda.is_available(),worker_init_fn=_worker_seed,
        generator=generator,persistent_workers=workers>0)
    stats={"client":client,"split":subset,"samples":len(dataset),"class_hist":dict(sorted(dataset.class_hist.items())),
           "pcam_group_balanced": sampler is not None, "pcam_group_hist": pcam_group_hist}
    return loader,stats


def _config(context: Context, msg: Message) -> dict:
    config = dict(context.run_config)
    config.update(dict(msg.content.get("config", {})))
    return config


def _model_from_message(msg: Message, device: torch.device):
    model = MultiDatasetRTDETRClassifier(dropout=0.1)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    return model.to(device)


@app.train()
def train(msg: Message, context: Context):
    config = _config(context, msg)
    partition_id = int(context.node_config["partition-id"])
    round_num = int(getattr(msg.metadata, "round", 1))
    master_seed = int(config.get("classification_master_seed", os.environ.get("MASTER_SEED", 42)))
    seed_everything(master_seed + partition_id * 10_000 + round_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _model_from_message(msg, device)
    loader, stats = build_stratified_loader(config, partition_id, "train", round_num)
    log(logging.INFO, f"[Classification client{partition_id}] round={round_num} train={stats}")
    metrics = train_one_round(
        model,
        loader,
        device=device,
        epochs=int(config["classification_local_epochs"]),
        lr=float(config["classification_lr"]),
        max_batches=int(config.get("classification_train_max_batches", 0)),
        backbone_lr=float(config.get("classification_backbone_lr", config["classification_lr"])),
        head_lr=float(config.get("classification_head_lr", config["classification_lr"])),
        weight_decay=float(config.get("classification_weight_decay", 1e-4)),
        label_smoothing=float(config.get("classification_label_smoothing", 0.0)),
        class_weight_power=float(config.get("classification_class_weight_power", 1.0)),
    )
    metrics["num-examples"] = len(loader.dataset)
    return Message(
        content=RecordDict({
            "arrays": msg.content["arrays"].from_torch_state_dict(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    config = _config(context, msg)
    partition_id = int(context.node_config["partition-id"])
    round_num = int(getattr(msg.metadata, "round", 1))
    master_seed = int(config.get("classification_master_seed", os.environ.get("MASTER_SEED", 42)))
    seed_everything(master_seed + partition_id * 10_000 + 100_000 + round_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _model_from_message(msg, device)
    loader, stats = build_stratified_loader(config, partition_id, "val", round_num)
    log(logging.INFO, f"[Classification client{partition_id}] round={round_num} val={stats}")
    metrics = evaluate(model, loader, device=device, max_batches=int(config.get("classification_eval_max_batches", 0)))
    metrics["num-examples"] = len(loader.dataset)
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
