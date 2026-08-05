#!/usr/bin/env python
"""Local client fine-tuning from a global medical5 detection model.

Each client starts from the same global checkpoint, fine-tunes only on its own
labeled train split, evaluates on its own test split, and saves a local model.
"""
import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault('FL_CURRENT_DATASET', 'medical5_mixed_labeled20')

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.ops import box_convert, box_iou

from models import RTDETR_L, RTDETR_L_WithAttention, RTDETR_L_WithASEM
from fl.detection_task import (
    DATASET_CONFIGS,
    CURRENT_DATASET,
    _Medical5YoloDataset,
    rtdetr_collate_fn,
    build_detection_loss,
    test as evaluate_detection,
)

MODEL_REGISTRY = {
    'RTDETR_L': RTDETR_L,
    'RTDETR_L_WithAttention': RTDETR_L_WithAttention,
    'RTDETR_L_WithASEM': RTDETR_L_WithASEM,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Global best_model_by_map50.pt')
    p.add_argument('--model-template', default='',
                   help='Optional per-client checkpoint template containing {client_id}; overrides --model.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--model-variant', default=os.environ.get('FL_MODEL_VARIANT', 'RTDETR_L_WithASEM'))
    p.add_argument('--dataset', default=os.environ.get('FL_CURRENT_DATASET', 'medical5_mixed_labeled20'))
    p.add_argument('--clients', default='0,1,2,3,4')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--conservative-personalization', action='store_true')
    p.add_argument('--fedbn-aware-progressive', action='store_true')
    p.add_argument('--validation-only', action='store_true',
                   help='Hyperparameter sweep mode: never instantiate or evaluate test data; weights are omitted unless explicitly requested.')
    p.add_argument('--save-validation-checkpoint', action='store_true',
                   help='In validation-only mode, save only the val-selected best checkpoint for reproducibility.')
    p.add_argument('--inner-validation', action='store_true',
                   help='In validation-only mode, select hyperparameters on a deterministic holdout from labeled/train instead of official val.')
    p.add_argument('--inner-val-fraction', type=float, default=0.2)
    p.add_argument('--backbone-lr', type=float, default=1e-6)
    p.add_argument('--late-backbone-lr', type=float, default=2e-7)
    p.add_argument('--neck-lr', type=float, default=5e-6)
    p.add_argument('--decoder-lr', type=float, default=3e-6)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--head-only-decoder-lr', type=float, default=None)
    p.add_argument('--head-only-head-lr', type=float, default=None)
    p.add_argument('--moe-lr', type=float, default=2e-6)
    p.add_argument('--head-only-epochs', type=int, default=10)
    p.add_argument('--l2sp-lambda', type=float, default=0.0)
    p.add_argument('--freeze-fedbn-bn', action='store_true')
    p.add_argument('--bn-recalibration', action='store_true',
                   help='Recompute client BN running statistics on augmentation-free labeled/train, then gate on val.')
    p.add_argument('--train-augmentation', action='store_true')
    p.add_argument('--image-size', type=int, default=640)
    p.add_argument('--class-aware-sampling', action='store_true')
    p.add_argument('--class-aware-max-weight', type=float, default=5.0)
    p.add_argument('--copy-paste-classes', default='',
                   help='Comma-separated class IDs for train-only Copy-Paste; empty disables it.')
    p.add_argument('--copy-paste-probability', type=float, default=0.0)
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--rare-class-protection', action='store_true')
    p.add_argument('--rare-class-freeze-threshold', type=int, default=10)
    p.add_argument('--rare-class-full-threshold', type=int, default=30)
    p.add_argument('--freeze-backbone-epochs', type=int, default=5)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--early-stop-patience', type=int, default=15)
    p.add_argument('--early-stop-min-delta', type=float, default=0.002)
    p.add_argument('--bn-momentum', type=float, default=0.01)
    p.add_argument('--top-k-checkpoints', type=int, default=5)
    p.add_argument('--soup-alphas', default='0,0.25,0.5,0.75,1')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--disable-moe-domain-supervision', action='store_true')
    p.add_argument('--strict-load', action='store_true')
    p.add_argument('--save-every-client', action='store_true', default=True)
    return p.parse_args()


def count_gt_boxes(dataset):
    counter = Counter()
    for image in dataset.images:
        for row in dataset._read_labels(image):
            if len(row):
                counter[int(float(row[0]))] += 1
    return counter


def count_gt_boxes_indices(dataset, indices):
    counter = Counter()
    for idx in indices:
        image = dataset.images[idx]
        for row in dataset._read_labels(image):
            if len(row):
                counter[int(float(row[0]))] += 1
    return counter


def class_aware_sample_weights(dataset, class_counts, max_weight=5.0):
    """Return capped image weights derived only from labeled/train GT support."""
    positive_counts = [count for count in class_counts.values() if count > 0]
    reference = max(positive_counts, default=1)
    weights = []
    for image in dataset.images:
        classes = {
            int(float(row[0])) for row in dataset._read_labels(image) if len(row)
        }
        if not classes:
            weights.append(1.0)
            continue
        rarity = max(
            math.sqrt(reference / max(class_counts.get(class_id, 1), 1))
            for class_id in classes
        )
        weights.append(min(float(max_weight), max(1.0, rarity)))
    return weights


def _dataset_source_key(image_path):
    value = str(image_path).lower()
    for source in ('kvasir', 'mitosis', 'tn5000', 'txl_pbc', 'urine'):
        if source in value:
            return source
    return Path(image_path).stem.split('_')[0]


def stratified_inner_split(dataset, fraction, seed):
    """Deterministic source-and-label-signature holdout without moving files."""
    if not 0.0 < fraction < 0.5:
        raise ValueError(f'inner-val-fraction must be in (0, 0.5), got {fraction}')
    strata = {}
    for idx, image in enumerate(dataset.images):
        classes = tuple(sorted({
            int(float(row[0])) for row in dataset._read_labels(image) if len(row)
        }))
        key = (_dataset_source_key(image), classes)
        strata.setdefault(key, []).append(idx)
    train_indices, val_indices = [], []
    for offset, key in enumerate(sorted(strata, key=str)):
        indices = list(strata[key])
        random.Random(seed + 104729 * (offset + 1)).shuffle(indices)
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        holdout = max(1, int(round(len(indices) * fraction)))
        holdout = min(holdout, len(indices) - 1)
        val_indices.extend(indices[:holdout])
        train_indices.extend(indices[holdout:])
    if not train_indices or not val_indices:
        raise RuntimeError('inner validation split produced an empty train or validation subset')
    return sorted(train_indices), sorted(val_indices)


class PersonalizationDataset(_Medical5YoloDataset):
    """Medical5 dataset with optional train-only, bbox-safe mild augmentation."""

    def __init__(self, data_dir, split, domain_id, train_augmentation=False,
                 image_size=640, copy_paste_classes=(), copy_paste_probability=0.0):
        super().__init__(data_dir, split, domain_id)
        self.train_augmentation = bool(train_augmentation and split == 'train')
        self.image_size = int(image_size)
        self.copy_paste_classes = {int(value) for value in copy_paste_classes}
        self.copy_paste_probability = float(copy_paste_probability) if split == 'train' else 0.0
        self.copy_paste_sources = []
        if self.copy_paste_classes and self.copy_paste_probability > 0:
            for image_path in self.images:
                for row in self._read_labels(image_path):
                    if len(row) and int(float(row[0])) in self.copy_paste_classes:
                        self.copy_paste_sources.append((image_path, row))
        transforms = []
        if self.train_augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
                    rotate=(-5, 5), shear=(-2, 2), border_mode=0,
                    fill=(114, 114, 114), p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3,
                ),
            ])
        transforms.extend([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size, min_width=self.image_size, border_mode=0,
                fill=(114, 114, 114),
            ),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', min_visibility=0.1),
        )

    def __getitem__(self, idx):
        from PIL import Image

        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        bboxes = []
        for class_id, x, y, width, height in self._read_labels(image_path):
            x = min(max(float(x), 0.0), 1.0)
            y = min(max(float(y), 0.0), 1.0)
            width = min(max(float(width), 0.0), 1.0)
            height = min(max(float(height), 0.0), 1.0)
            x1, y1 = max(0.0, x - width / 2), max(0.0, y - height / 2)
            x2, y2 = min(1.0, x + width / 2), min(1.0, y + height / 2)
            width, height = x2 - x1, y2 - y1
            if width <= 1e-6 or height <= 1e-6:
                continue
            bboxes.append([
                (x1 + x2) / 2, (y1 + y2) / 2, width, height, class_id,
            ])
        if (self.copy_paste_sources and random.random() < self.copy_paste_probability):
            source_path, source_row = random.choice(self.copy_paste_sources)
            source_image = np.array(Image.open(source_path).convert('RGB'))
            class_id, x, y, width, height = source_row
            source_h, source_w = source_image.shape[:2]
            x1 = max(0, int((float(x) - float(width) / 2) * source_w))
            y1 = max(0, int((float(y) - float(height) / 2) * source_h))
            x2 = min(source_w, int((float(x) + float(width) / 2) * source_w))
            y2 = min(source_h, int((float(y) + float(height) / 2) * source_h))
            crop = source_image[y1:y2, x1:x2]
            if crop.size:
                target_h, target_w = image.shape[:2]
                crop_h, crop_w = crop.shape[:2]
                scale = min(1.0, target_w / max(crop_w, 1), target_h / max(crop_h, 1))
                if scale < 1.0:
                    resized_w = max(1, int(crop_w * scale))
                    resized_h = max(1, int(crop_h * scale))
                    crop = np.array(Image.fromarray(crop).resize((resized_w, resized_h)))
                    crop_h, crop_w = crop.shape[:2]
                paste_x = random.randint(0, max(target_w - crop_w, 0))
                paste_y = random.randint(0, max(target_h - crop_h, 0))
                image[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = crop
                bboxes.append([
                    (paste_x + crop_w / 2) / target_w,
                    (paste_y + crop_h / 2) / target_h,
                    crop_w / target_w,
                    crop_h / target_h,
                    int(float(class_id)),
                ])
        transformed = self.transform(image=image, bboxes=bboxes)
        labels = [
            [row[4], row[0], row[1], row[2], row[3]]
            for row in transformed['bboxes']
        ]
        return {
            'pixel_values': transformed['image'],
            'labels': labels,
            'domain_id': self.domain_id,
            'image_path': str(image_path),
        }


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def batchnorm_state_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f'{module_name}.' if module_name else ''
            names.update(
                name for name in model.state_dict() if name.startswith(prefix)
            )
    return names


def batchnorm_affine_state_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f'{module_name}.' if module_name else ''
        if module.weight is not None:
            names.add(f'{prefix}weight')
        if module.bias is not None:
            names.add(f'{prefix}bias')
    return names


def max_state_difference(reference, current, names):
    differences = []
    for name in names:
        left, right = reference[name], current[name].detach().cpu()
        if left.is_floating_point():
            differences.append(float((left - right).abs().max().item()))
        else:
            differences.append(0.0 if torch.equal(left, right) else float('inf'))
    return max(differences, default=0.0)


def average_states(states, protected_state=None, protected_names=None):
    protected_names = protected_names or set()
    averaged = {}
    for name in states[0]:
        if name in protected_names:
            averaged[name] = protected_state[name].clone()
            continue
        values = [state[name] for state in states]
        if values[0].is_floating_point():
            averaged[name] = torch.stack(values).mean(dim=0)
        else:
            averaged[name] = values[0].clone()
    return averaged


def blend_states(initial, tuned, alpha, protected_names=None):
    protected_names = protected_names or set()
    blended = {}
    for name, initial_value in initial.items():
        if name in protected_names:
            blended[name] = initial_value.clone()
            continue
        tuned_value = tuned[name]
        if initial_value.is_floating_point():
            blended[name] = initial_value.mul(1.0 - alpha).add(tuned_value, alpha=alpha)
        else:
            blended[name] = tuned_value.clone() if alpha >= 0.5 else initial_value.clone()
    return blended


def parameter_group_name(name):
    if name.startswith('asem_p5.experts.'):
        return 'moe_expert'
    if name.startswith('asem_p5.'):
        return 'moe_router'
    if name.startswith('model.28.'):
        if any(token in name for token in (
            'score_head', 'bbox_head', 'denoising_class_embed'
        )):
            return 'head'
        return 'decoder'
    match = re.match(r'model\.(\d+)\.', name)
    if match:
        layer_index = int(match.group(1))
        if layer_index <= 6:
            return 'backbone_early'
        if layer_index <= 9:
            return 'backbone_late'
    return 'neck'


def configure_optimizer(model, args):
    if args.fedbn_aware_progressive:
        target_lrs = {
            'backbone_early': 0.0,
            'backbone_late': args.late_backbone_lr,
            'neck': args.neck_lr,
            'decoder': args.decoder_lr,
            'head': args.head_lr,
            'moe_router': args.moe_lr,
            'moe_expert': 0.0,
        }
    else:
        target_lrs = {
            'backbone_early': args.backbone_lr,
            'backbone_late': args.backbone_lr,
            'neck': args.neck_lr,
            'decoder': args.neck_lr,
            'head': args.head_lr,
            'moe_router': args.moe_lr,
            'moe_expert': args.moe_lr,
        }
    grouped = {name: [] for name in target_lrs}
    for name, parameter in model.named_parameters():
        grouped[parameter_group_name(name)].append(parameter)
    head_only_lrs = {
        'decoder': args.head_only_decoder_lr,
        'head': args.head_only_head_lr,
    }
    groups = [
        {
            'params': grouped[name],
            'lr': lr,
            'target_lr': lr,
            'head_only_target_lr': head_only_lrs.get(name),
            'group_name': name,
        }
        for name, lr in target_lrs.items() if grouped[name]
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return optimizer, {name: len(params) for name, params in grouped.items()}


def set_stage_trainability(model, stage, freeze_fedbn_bn=False):
    active = {'decoder', 'head'} if stage == 1 else {
        'backbone_late', 'neck', 'decoder', 'head', 'moe_router'
    }
    bn_parameter_ids = set()
    if freeze_fedbn_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                bn_parameter_ids.update(
                    id(parameter) for parameter in module.parameters(recurse=False)
                )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            parameter_group_name(name) in active and id(parameter) not in bn_parameter_ids
        )


def set_legacy_trainability(model, backbone_frozen):
    for name, parameter in model.named_parameters():
        group_name = parameter_group_name(name)
        parameter.requires_grad_(
            not (backbone_frozen and group_name in {'backbone_early', 'backbone_late'})
        )


def freeze_all_bn(model, freeze_affine=True):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            if freeze_affine:
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(False)


def freeze_backbone_bn_stats(model):
    for name, module in model.named_modules():
        match = re.match(r'model\.(\d+)(?:\.|$)', name)
        if (match and int(match.group(1)) <= 9
                and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
            module.eval()


def set_bn_momentum(model, momentum):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


@torch.no_grad()
def recalibrate_batchnorm(model, loader, device,
                          disable_moe_domain_supervision=False):
    """Recompute only BN running statistics on deterministic client train data."""
    modules = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not modules:
        return {'bn_modules': 0, 'batches': 0, 'mode': 'not_applicable'}

    model.to(device)
    model.eval()
    original_momenta = [module.momentum for module in modules]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    batches = 0
    for batch in loader:
        if disable_moe_domain_supervision:
            batch.pop('domain_id', None)
            batch.pop('domain_label', None)
            batch.pop('domain', None)
        images = batch['images'].to(device)
        if 'domain_id' in batch and torch.is_tensor(batch['domain_id']):
            batch['domain_id'] = batch['domain_id'].to(device)
        model(images, batch=batch)
        batches += 1

    for module, momentum in zip(modules, original_momenta):
        module.momentum = momentum
    model.eval()
    return {
        'bn_modules': len(modules),
        'batches': batches,
        'mode': 'reset_then_cumulative_average',
    }


def update_learning_rates(optimizer, epoch, epochs, warmup_epochs, head_only_epochs):
    stage = 1 if epoch <= head_only_epochs else 2
    stage_epoch = epoch if stage == 1 else epoch - head_only_epochs
    stage_epochs = head_only_epochs if stage == 1 else max(epochs - head_only_epochs, 1)
    explicit_head_only_lrs = stage == 1 and any(
        group.get('head_only_target_lr') is not None
        for group in optimizer.param_groups
    )
    if explicit_head_only_lrs:
        factor = 1.0
    elif warmup_epochs > 0 and stage_epoch <= warmup_epochs:
        factor = stage_epoch / warmup_epochs
    else:
        progress = (stage_epoch - warmup_epochs) / max(stage_epochs - warmup_epochs, 1)
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    active = {'decoder', 'head'} if stage == 1 else {
        'backbone_late', 'neck', 'decoder', 'head', 'moe_router'
    }
    result = {}
    for group in optimizer.param_groups:
        target_lr = group['target_lr']
        if stage == 1 and group.get('head_only_target_lr') is not None:
            target_lr = group['head_only_target_lr']
        lr = float(target_lr) * factor if group['group_name'] in active else 0.0
        group['lr'] = lr
        result[group['group_name']] = lr
    return result, stage


def update_legacy_learning_rates(optimizer, epoch, epochs, warmup_epochs, backbone_frozen):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        factor = epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    result = {}
    for group in optimizer.param_groups:
        lr = float(group['target_lr']) * factor
        if backbone_frozen and group['group_name'] in {'backbone_early', 'backbone_late'}:
            lr = 0.0
        group['lr'] = lr
        result[group['group_name']] = lr
    return result


def build_l2sp_anchors(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter_group_name(name) not in {'head'}
    }


def l2sp_penalty(model, anchors):
    penalty = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in anchors:
            continue
        value = torch.sum((parameter - anchors[name]) ** 2)
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return next(model.parameters()).new_zeros(())
    return penalty


def protect_rare_class_gradients(model, class_counts, nc, freeze_threshold, full_threshold):
    """Prevent scarce local classes from erasing federated classification rows."""
    scales = []
    for class_id in range(nc):
        count = int(class_counts.get(class_id, 0))
        if count < freeze_threshold:
            scales.append(0.0)
        elif count < full_threshold:
            scales.append(math.sqrt(count / max(full_threshold, 1)))
        else:
            scales.append(1.0)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if ('score_head' not in name and
                'denoising_class_embed.weight' not in name):
            continue
        if parameter.grad.ndim == 0 or parameter.grad.shape[0] < nc:
            continue
        scale = parameter.grad.new_tensor(scales)
        view_shape = [nc] + [1] * (parameter.grad.ndim - 1)
        parameter.grad[:nc].mul_(scale.view(view_shape))
    return scales


@torch.no_grad()
def evaluate_fixed_operating_point(model, loader, device, threshold,
                                   disable_moe_domain_supervision=False):
    """Compute fixed-threshold P/R/F1, matched-TP mIoU and normalized FROC-AUC."""
    model.to(device)
    model.eval()
    rows = []
    total_gt = 0
    total_images = 0
    for batch in loader:
        if disable_moe_domain_supervision:
            batch.pop('domain_id', None)
            batch.pop('domain_label', None)
            batch.pop('domain', None)
        images = batch['images'].to(device)
        batch['cls'] = batch['cls'].to(device)
        batch['bboxes'] = batch['bboxes'].to(device)
        if 'domain_id' in batch and torch.is_tensor(batch['domain_id']):
            batch['domain_id'] = batch['domain_id'].to(device)
        outputs = model(images, batch=batch)
        if not (isinstance(outputs, tuple) and len(outputs) == 2):
            continue
        inference_out, _ = outputs
        offset = 0
        for image_index, num_gt in enumerate(batch['gt_groups']):
            num_gt = int(num_gt)
            gt_boxes = batch['bboxes'][offset:offset + num_gt]
            gt_labels = batch['cls'][offset:offset + num_gt].long().flatten()
            offset += num_gt
            total_gt += num_gt
            total_images += 1

            prediction = inference_out[image_index]
            pred_boxes = prediction[:, :4]
            pred_scores, pred_labels = prediction[:, 4:].max(dim=-1)
            keep = pred_scores > 0.001
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]
            matched_gt = torch.zeros(num_gt, dtype=torch.bool, device=device)
            iou_matrix = None
            if num_gt and len(pred_boxes):
                iou_matrix = box_iou(
                    box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
                    box_convert(gt_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
                )
                iou_matrix = iou_matrix.masked_fill(
                    pred_labels[:, None] != gt_labels[None, :], -1.0
                )
            for pred_index, score in enumerate(pred_scores):
                is_tp = 0.0
                matched_iou = 0.0
                if iou_matrix is not None:
                    iou_value, gt_index = iou_matrix[pred_index].max(dim=0)
                    if float(iou_value) > 0.5 and not bool(matched_gt[gt_index]):
                        matched_gt[gt_index] = True
                        is_tp = 1.0
                        matched_iou = float(iou_value)
                rows.append((float(score), is_tp, matched_iou))

    if not rows or total_gt == 0 or total_images == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'miou': 0.0,
            'froc_auc': 0.0, 'froc_sensitivity': {}, 'threshold': float(threshold),
        }
    values = np.asarray(rows, dtype=np.float64)
    selected = values[:, 0] >= float(threshold)
    tp = float(values[selected, 1].sum())
    fp = float(selected.sum() - tp)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / total_gt
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    matched = values[selected & (values[:, 1] > 0.5), 2]
    miou = float(matched.mean()) if len(matched) else 0.0

    order = np.argsort(-values[:, 0], kind='stable')
    sorted_tp = values[order, 1]
    tp_curve = np.cumsum(sorted_tp)
    fp_curve = np.cumsum(1.0 - sorted_tp) / total_images
    sensitivity_curve = tp_curve / total_gt
    x = np.concatenate(([0.0], fp_curve))
    y = np.concatenate(([0.0], sensitivity_curve))
    unique_x = np.unique(x)
    unique_y = np.asarray([y[x == point].max() for point in unique_x])
    dense_grid = np.linspace(0.0, 8.0, 801)
    dense_sensitivity = np.interp(dense_grid, unique_x, unique_y)
    froc_auc = float(np.trapz(dense_sensitivity, dense_grid) / 8.0)
    froc_points = np.asarray([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    froc_sensitivity = np.interp(froc_points, unique_x, unique_y)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou,
        'froc_auc': froc_auc,
        'froc_sensitivity': {
            f'{point:g}': float(value) for point, value in zip(froc_points, froc_sensitivity)
        },
        'threshold': float(threshold),
        'total_gt': int(total_gt),
        'total_images': int(total_images),
    }


def train_one_client(model, trainloader, valloader, epochs, lr, device, nc, client_id,
                     client_out, args, train_class_counts, calibrationloader=None,
                     disable_moe_domain_supervision=False):
    criterion = build_detection_loss(nc=nc, device=device)
    model.to(device)
    protected_bn_names = batchnorm_state_names(model) if args.freeze_fedbn_bn else set()
    bn_affine_names = batchnorm_affine_state_names(model) if args.freeze_fedbn_bn else set()
    if args.bn_recalibration and calibrationloader is None:
        raise ValueError('--bn-recalibration requires an augmentation-free train loader')
    if args.freeze_fedbn_bn:
        freeze_all_bn(model, freeze_affine=True)
    if args.conservative_personalization:
        optimizer, group_counts = configure_optimizer(model, args)
        if not args.freeze_fedbn_bn:
            set_bn_momentum(model, args.bn_momentum)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        group_counts = {'legacy_all_parameters': sum(1 for _ in model.parameters())}
    l2sp_anchors = (
        build_l2sp_anchors(model) if args.l2sp_lambda > 0 else {}
    )
    rare_class_scales = {
        str(class_id): 1.0 for class_id in range(nc)
    }
    history = []
    initial_state = clone_state(model)
    eval_loss, map50, precision, recall, f1, threshold = evaluate_detection(
        model, valloader, device, nc=nc, client_id=client_id, split_name='val',
        moe_domain_supervision=not disable_moe_domain_supervision,
    )
    initial_metrics = {
        'epoch': 0,
        'train_loss': None,
        'val_eval_loss': float(eval_loss),
        'val_map50': float(map50),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
        'val_threshold': float(threshold),
    }
    history.append(initial_metrics)
    best_map50 = float(map50)
    best_epoch = 0
    best_metrics = dict(initial_metrics)
    best_single_state = initial_state
    top_states = [(float(map50), 0, initial_state)]
    patience_anchor = float(map50)
    stale_epochs = 0
    best_path = client_out / 'best_model_by_map50.pt'
    best_single_path = client_out / 'best_single_by_map50.pt'
    final_model_path = client_out / 'final_model.pt'
    if not args.validation_only:
        torch.save(initial_state, best_single_path)
    print(
        f'[client {client_id}] epoch 0/{epochs} val_map50={map50:.6f} '
        f'val_f1={f1:.6f} (FedBN personalized initialization)',
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        if args.conservative_personalization:
            if args.fedbn_aware_progressive:
                if epoch == args.head_only_epochs + 1:
                    patience_anchor = best_map50
                    stale_epochs = 0
                current_lrs, stage = update_learning_rates(
                    optimizer, epoch, epochs, args.warmup_epochs,
                    args.head_only_epochs,
                )
                set_stage_trainability(
                    model, stage, freeze_fedbn_bn=args.freeze_fedbn_bn,
                )
            else:
                stage = 0
                backbone_frozen = epoch <= args.freeze_backbone_epochs
                set_legacy_trainability(model, backbone_frozen)
                current_lrs = update_legacy_learning_rates(
                    optimizer, epoch, epochs, args.warmup_epochs,
                    backbone_frozen,
                )
        else:
            stage = 0
            current_lrs = {'all': float(optimizer.param_groups[0]['lr'])}
        model.train()
        if args.freeze_fedbn_bn:
            freeze_all_bn(model, freeze_affine=True)
        elif args.conservative_personalization and not args.fedbn_aware_progressive and backbone_frozen:
            freeze_backbone_bn_stats(model)
        total_loss = 0.0
        total_task_loss = 0.0
        total_l2sp = 0.0
        total_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(trainloader, start=1):
            if disable_moe_domain_supervision:
                batch.pop('domain_id', None)
                batch.pop('domain_label', None)
                batch.pop('domain', None)
            images = batch['images'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device)
            if 'domain_id' in batch and torch.is_tensor(batch['domain_id']):
                batch['domain_id'] = batch['domain_id'].to(device)

            outputs = model(images, batch=batch)
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs
            if enc_bboxes is not None:
                enc_bboxes = enc_bboxes.unsqueeze(0)
                enc_scores = enc_scores.unsqueeze(0)
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta,
            )
            task_loss = sum(loss_dict.values())
            aux_loss = model.get_aux_loss() if hasattr(model, 'get_aux_loss') else None
            if aux_loss is not None:
                task_loss = task_loss + aux_loss
            regularizer = l2sp_penalty(model, l2sp_anchors)
            loss = task_loss + args.l2sp_lambda * regularizer
            (loss / args.grad_accum_steps).backward()
            should_step = (
                batch_index % args.grad_accum_steps == 0
                or batch_index == len(trainloader)
            )
            if should_step:
                if args.rare_class_protection:
                    rare_class_scales = protect_rare_class_gradients(
                        model, train_class_counts, nc,
                        args.rare_class_freeze_threshold,
                        args.rare_class_full_threshold,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            total_task_loss += float(task_loss.item())
            total_l2sp += float(regularizer.item())
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        avg_task_loss = total_task_loss / max(total_batches, 1)
        avg_l2sp = total_l2sp / max(total_batches, 1)
        eval_loss, map50, precision, recall, f1, threshold = evaluate_detection(
            model, valloader, device, nc=nc, client_id=client_id, split_name='val',
            moe_domain_supervision=not disable_moe_domain_supervision,
        )
        row = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_task_loss': avg_task_loss,
            'train_l2sp_penalty': avg_l2sp,
            'val_eval_loss': float(eval_loss),
            'val_map50': float(map50),
            'val_precision': float(precision),
            'val_recall': float(recall),
            'val_f1': float(f1),
            'val_threshold': float(threshold),
            'learning_rates': current_lrs,
            'personalization_stage': stage,
        }
        history.append(row)
        if float(map50) > best_map50:
            best_map50 = float(map50)
            best_epoch = epoch
            best_metrics = dict(row)
            best_single_state = clone_state(model)
            if not args.validation_only:
                torch.save(best_single_state, best_single_path)
        if args.conservative_personalization:
            candidate = (float(map50), epoch, clone_state(model))
            top_states.append(candidate)
            top_states.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            top_states = top_states[:max(args.top_k_checkpoints, 1)]
            if float(map50) > patience_anchor + args.early_stop_min_delta:
                patience_anchor = float(map50)
                stale_epochs = 0
            else:
                stale_epochs += 1
        (client_out / 'metrics_history.json').write_text(json.dumps(history, indent=2))
        print(
            f'[client {client_id}] epoch {epoch}/{epochs} train_loss={avg_loss:.6f} '
            f'task_loss={avg_task_loss:.6f} l2sp={avg_l2sp:.6f} stage={stage} '
            f'val_map50={map50:.6f} val_f1={f1:.6f} best_epoch={best_epoch} '
            f'lrs={current_lrs} stale={stale_epochs}',
            flush=True,
        )
        early_stop_enabled = (
            not args.fedbn_aware_progressive or epoch > args.head_only_epochs
        )
        if (args.conservative_personalization and early_stop_enabled
                and args.early_stop_patience > 0
                and stale_epochs >= args.early_stop_patience):
            print(
                f'[client {client_id}] early stop at epoch {epoch}: '
                f'patience={args.early_stop_patience}',
                flush=True,
            )
            break

    if not args.validation_only:
        torch.save(model.state_dict(), final_model_path)
    raw_best_epoch = int(best_epoch)
    selection_history = []
    calibration_info = None
    recalibrated_state = None
    if args.conservative_personalization:
        soup_state = average_states(
            [item[2] for item in top_states],
            protected_state=initial_state,
            protected_names=protected_bn_names,
        )
        soup_path = client_out / 'topk_checkpoint_soup.pt'
        if not args.validation_only and not args.bn_recalibration:
            torch.save(soup_state, soup_path)
        candidates = [('A_fedbn_epoch0', initial_state), ('local_best_single', best_single_state)]
        if not args.bn_recalibration:
            alphas = sorted({float(value) for value in args.soup_alphas.split(',') if value.strip()})
            for alpha in alphas:
                if not 0.0 <= alpha <= 1.0:
                    raise ValueError(f'Soup alpha must be in [0,1], got {alpha}')
                candidates.append((
                    f'local_epoch0_topk_soup_alpha_{alpha:g}',
                    blend_states(
                        initial_state, soup_state, alpha,
                        protected_names=protected_bn_names,
                    ),
                ))
        evaluated_candidates = []
        for name, state in candidates:
            model.load_state_dict(state, strict=True)
            values = evaluate_detection(
                model, valloader, device, nc=nc, client_id=client_id,
                split_name=f'val_selection_{name}',
                moe_domain_supervision=not disable_moe_domain_supervision,
            )
            candidate_metrics = {
                'selection': name,
                'val_eval_loss': float(values[0]),
                'val_map50': float(values[1]),
                'val_precision': float(values[2]),
                'val_recall': float(values[3]),
                'val_f1': float(values[4]),
                'val_threshold': float(values[5]),
            }
            selection_history.append(candidate_metrics)
            evaluated_candidates.append((candidate_metrics, state))

        initial_candidate = next(
            item for item in evaluated_candidates
            if item[0]['selection'] == 'A_fedbn_epoch0'
        )
        local_candidates = [
            item for item in evaluated_candidates
            if item[0]['selection'] != 'A_fedbn_epoch0'
            and not item[0]['selection'].endswith('_alpha_0')
        ]
        best_local_metrics, best_local_state = max(
            local_candidates,
            key=lambda item: (item[0]['val_map50'], item[0]['val_f1']),
        )
        best_local_metrics = {
            **best_local_metrics,
            'candidate_role': 'B_local_finetuned',
        }
        training_bn_max_abs_difference = max_state_difference(
            initial_state, best_local_state, protected_bn_names,
        )
        final_candidates = [
            ({**initial_candidate[0], 'candidate_role': 'A_fedbn_epoch0'}, initial_candidate[1]),
            (best_local_metrics, best_local_state),
        ]
        if args.bn_recalibration:
            model.load_state_dict(best_local_state, strict=True)
            calibration_info = recalibrate_batchnorm(
                model, calibrationloader, device,
                disable_moe_domain_supervision=disable_moe_domain_supervision,
            )
            recalibrated_state = clone_state(model)
            values = evaluate_detection(
                model, valloader, device, nc=nc, client_id=client_id,
                split_name='val_selection_C_bn_recalibrated',
                moe_domain_supervision=not disable_moe_domain_supervision,
            )
            recalibrated_metrics = {
                'selection': f"C_{best_local_metrics['selection']}_bn_recalibrated",
                'candidate_role': 'C_local_finetuned_bn_recalibrated',
                'source_selection': best_local_metrics['selection'],
                'val_eval_loss': float(values[0]),
                'val_map50': float(values[1]),
                'val_precision': float(values[2]),
                'val_recall': float(values[3]),
                'val_f1': float(values[4]),
                'val_threshold': float(values[5]),
                'bn_recalibration': calibration_info,
            }
            selection_history.append(recalibrated_metrics)
            final_candidates.append((recalibrated_metrics, recalibrated_state))

        best_metrics, selected_state = max(
            final_candidates,
            key=lambda item: (item[0]['val_map50'], item[0]['val_f1']),
        )
        if best_metrics['candidate_role'] == 'A_fedbn_epoch0':
            best_epoch = 0
        elif 'best_single' in best_metrics.get('source_selection', best_metrics['selection']):
            best_epoch = raw_best_epoch
        else:
            best_epoch = -1
        model.load_state_dict(selected_state, strict=True)
        if not args.validation_only or args.save_validation_checkpoint:
            torch.save(selected_state, best_path)
    else:
        training_bn_max_abs_difference = max_state_difference(
            initial_state, best_single_state, protected_bn_names,
        )
        model.load_state_dict(best_single_state, strict=True)
        if not args.validation_only or args.save_validation_checkpoint:
            torch.save(best_single_state, best_path)

    fedbn_bn_max_abs_difference = max_state_difference(
        initial_state, model.state_dict(), protected_bn_names,
    )
    fedbn_bn_affine_max_abs_difference = max_state_difference(
        initial_state, model.state_dict(), bn_affine_names,
    )
    if args.freeze_fedbn_bn and training_bn_max_abs_difference != 0.0:
        raise RuntimeError(
            'FedBN invariant violated: BN changed during gradient training; '
            f'max_abs_difference={training_bn_max_abs_difference}'
        )
    if args.freeze_fedbn_bn and fedbn_bn_affine_max_abs_difference != 0.0:
        raise RuntimeError(
            'FedBN invariant violated: frozen BN affine state changed; '
            f'max_abs_difference={fedbn_bn_affine_max_abs_difference}'
        )

    metadata = {
        **best_metrics,
        'raw_best_single_epoch': raw_best_epoch,
        'raw_best_single_val_map50': float(best_map50),
        'top_k_epochs': [int(item[1]) for item in top_states],
        'top_k_val_map50': [float(item[0]) for item in top_states],
        'parameter_group_counts': group_counts,
        'conservative_personalization': bool(args.conservative_personalization),
        'fedbn_aware_progressive': bool(args.fedbn_aware_progressive),
        'freeze_fedbn_bn': bool(args.freeze_fedbn_bn),
        'bn_aware_soup': bool(args.freeze_fedbn_bn),
        'fedbn_bn_state_preserved_during_gradient_training': (
            not args.freeze_fedbn_bn or training_bn_max_abs_difference == 0.0
        ),
        'fedbn_bn_affine_preserved': (
            not args.freeze_fedbn_bn or fedbn_bn_affine_max_abs_difference == 0.0
        ),
        'fedbn_bn_state_preserved': (
            not args.freeze_fedbn_bn or fedbn_bn_max_abs_difference == 0.0
        ),
        'bn_recalibration_enabled': bool(args.bn_recalibration),
        'bn_recalibration_selected': best_metrics.get('candidate_role') == 'C_local_finetuned_bn_recalibrated',
        'bn_recalibration': calibration_info,
        'fedbn_bn_training_max_abs_difference': training_bn_max_abs_difference,
        'fedbn_bn_affine_max_abs_difference': fedbn_bn_affine_max_abs_difference,
        'fedbn_bn_max_abs_difference': fedbn_bn_max_abs_difference,
        'head_only_epochs': int(args.head_only_epochs),
        'l2sp_lambda': float(args.l2sp_lambda),
        'rare_class_protection': bool(args.rare_class_protection),
        'train_class_counts': {
            str(class_id): int(train_class_counts.get(class_id, 0))
            for class_id in range(nc)
        },
        'rare_class_gradient_scales': (
            rare_class_scales if args.rare_class_protection else None
        ),
    }
    (client_out / 'best_model_by_map50_meta.json').write_text(json.dumps(metadata, indent=2))
    (client_out / 'selection_history.json').write_text(json.dumps(selection_history, indent=2))
    return history, best_epoch, metadata, best_path, final_model_path


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError('--grad-accum-steps must be >= 1')
    if args.image_size < 320 or args.image_size % 32:
        raise ValueError('--image-size must be >= 320 and divisible by 32')
    if not 0.0 <= args.copy_paste_probability <= 1.0:
        raise ValueError('--copy-paste-probability must be in [0,1]')
    copy_paste_classes = tuple(
        int(value) for value in args.copy_paste_classes.split(',') if value.strip()
    )
    if args.inner_validation and not args.validation_only:
        raise ValueError('--inner-validation requires --validation-only')
    if args.bn_recalibration and not args.freeze_fedbn_bn:
        raise ValueError('--bn-recalibration requires --freeze-fedbn-bn')
    if (args.head_only_decoder_lr is None) != (args.head_only_head_lr is None):
        raise ValueError('Set both --head-only-decoder-lr and --head-only-head-lr')
    if args.dataset != CURRENT_DATASET:
        raise RuntimeError(f'FL_CURRENT_DATASET must be {args.dataset}, got imported CURRENT_DATASET={CURRENT_DATASET}. Set env before running.')
    cfg = DATASET_CONFIGS[args.dataset]
    nc = cfg['nc']
    client_ids = [int(x) for x in args.clients.split(',') if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cls = MODEL_REGISTRY[args.model_variant]
    base_state = None if args.model_template else torch.load(
        args.model, map_location='cpu', weights_only=False
    )
    device = torch.device(args.device)
    rows = []
    all_client_histories = {}
    print('Local finetune started:', datetime.now().isoformat(), flush=True)
    print('Global model:', args.model, flush=True)
    print('Per-client model template:', args.model_template or '(disabled)', flush=True)
    print('Output dir:', out_dir, flush=True)

    for client_id in client_ids:
        client_seed = args.seed + client_id
        random.seed(client_seed)
        np.random.seed(client_seed)
        torch.manual_seed(client_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(client_seed)
        client_cfg = cfg['clients'][client_id]
        data_dir = client_cfg['data_dir']
        domain_id = client_cfg.get('domain_id', client_id)
        client_name = client_cfg.get('name', f'client{client_id}')
        print(f'\n=== Client {client_id} {client_name} ===', flush=True)

        train_full = PersonalizationDataset(
            data_dir, 'train', domain_id,
            train_augmentation=(
                args.fedbn_aware_progressive and args.train_augmentation
            ),
            image_size=args.image_size,
            copy_paste_classes=copy_paste_classes,
            copy_paste_probability=args.copy_paste_probability,
        )
        if args.inner_validation:
            if not args.validation_only:
                raise ValueError('--inner-validation is only allowed with --validation-only')
            train_eval_full = PersonalizationDataset(
                data_dir, 'train', domain_id, train_augmentation=False,
                image_size=args.image_size,
            )
            inner_train_indices, inner_val_indices = stratified_inner_split(
                train_full, args.inner_val_fraction, client_seed + 2000,
            )
            trainset = Subset(train_full, inner_train_indices)
            valset = Subset(train_eval_full, inner_val_indices)
            calibration_set = Subset(train_eval_full, inner_train_indices)
            train_class_counts = count_gt_boxes_indices(
                train_full, inner_train_indices,
            )
            print(
                f'Client {client_id} inner validation: '
                f'train={len(trainset)} holdout={len(valset)} '
                f'fraction={args.inner_val_fraction}',
                flush=True,
            )
        else:
            trainset = train_full
            valset = PersonalizationDataset(
                data_dir, 'val', domain_id, train_augmentation=False,
                image_size=args.image_size,
            )
            calibration_set = (
                PersonalizationDataset(
                    data_dir, 'train', domain_id, train_augmentation=False,
                    image_size=args.image_size,
                )
                if args.bn_recalibration else None
            )
            train_class_counts = count_gt_boxes(train_full)
        testset = None if args.validation_only else PersonalizationDataset(
            data_dir, 'test', domain_id, train_augmentation=False,
            image_size=args.image_size,
        )
        train_generator = None
        if args.fedbn_aware_progressive:
            train_generator = torch.Generator().manual_seed(client_seed + 1000)
        sampler = None
        if args.class_aware_sampling:
            full_weights = class_aware_sample_weights(
                train_full, train_class_counts, args.class_aware_max_weight,
            )
            sample_weights = (
                [full_weights[index] for index in trainset.indices]
                if isinstance(trainset, Subset) else full_weights
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(trainset), replacement=True,
                generator=train_generator,
            )
            print(
                f'Client {client_id} class-aware sampling enabled: '
                f'min_weight={min(sample_weights):.4f} '
                f'max_weight={max(sample_weights):.4f}',
                flush=True,
            )
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=sampler is None,
            sampler=sampler,
            collate_fn=rtdetr_collate_fn, num_workers=0,
            generator=train_generator if sampler is None else None,
        )
        valloader = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=rtdetr_collate_fn, num_workers=0)
        calibrationloader = None if calibration_set is None else DataLoader(
            calibration_set, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=rtdetr_collate_fn, num_workers=0,
        )
        testloader = None if args.validation_only else DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=rtdetr_collate_fn, num_workers=0,
        )
        gt_boxes = 0 if args.validation_only else int(sum(count_gt_boxes(testset).values()))

        client_model_path = (
            args.model_template.format(client_id=client_id)
            if args.model_template else args.model
        )
        client_base_state = base_state if base_state is not None else torch.load(
            client_model_path, map_location='cpu', weights_only=False
        )
        model = model_cls(nc=nc)
        model.load_state_dict(client_base_state, strict=args.strict_load)
        print(f'Client {client_id} initialization: {client_model_path}', flush=True)
        client_out = out_dir / f'client{client_id}_{client_name}'
        client_out.mkdir(parents=True, exist_ok=True)
        history, best_epoch, best_metrics, best_path, final_model_path = train_one_client(
            model, trainloader, valloader, args.epochs, args.lr, device, nc, client_id,
            client_out, args, train_class_counts, calibrationloader=calibrationloader,
            disable_moe_domain_supervision=args.disable_moe_domain_supervision,
        )
        all_client_histories[str(client_id)] = history

        if args.validation_only:
            row = {
                'client_id': client_id,
                'client_name': client_name,
                'selection': best_metrics.get('selection', 'best_single'),
                'selected_epoch': best_epoch,
                'raw_best_single_epoch': best_metrics.get('raw_best_single_epoch'),
                'train_images': len(trainset),
                'val_images': len(valset),
                'val_map50': float(best_metrics['val_map50']),
                'val_precision': float(best_metrics['val_precision']),
                'val_recall': float(best_metrics['val_recall']),
                'val_f1': float(best_metrics['val_f1']),
                'val_threshold': float(best_metrics['val_threshold']),
                'model_path': str(best_path) if args.save_validation_checkpoint else None,
            }
            rows.append(row)
            (client_out / 'validation_metrics.json').write_text(json.dumps(row, indent=2))
            del model
            torch.cuda.empty_cache()
            continue

        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False), strict=False)
        eval_loss, map50, _, _, _, _ = evaluate_detection(
            model, testloader, device, nc=nc, client_id=client_id, split_name='test',
            moe_domain_supervision=not args.disable_moe_domain_supervision,
        )
        val_threshold = float(best_metrics['val_threshold'])
        operating = evaluate_fixed_operating_point(
            model, testloader, device, threshold=val_threshold,
            disable_moe_domain_supervision=args.disable_moe_domain_supervision,
        )
        precision = operating['precision']
        recall = operating['recall']
        f1 = operating['f1']
        row = {
            'client_id': client_id,
            'client_name': client_name,
            'model_path': str(best_path),
            'final_model_path': str(final_model_path),
            'best_epoch': best_epoch,
            'best_val_metrics': best_metrics,
            'train_images': len(trainset),
            'val_images': len(valset),
            'test_images': len(testset),
            'test_gt_boxes': gt_boxes,
            'eval_loss': float(eval_loss),
            'map50': float(map50),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'miou': float(operating['miou']),
            'froc_auc': float(operating['froc_auc']),
            'froc_sensitivity': operating['froc_sensitivity'],
            'threshold': val_threshold,
            'threshold_source': 'selected_model_val',
        }
        rows.append(row)
        (client_out / 'metrics.json').write_text(json.dumps(row, indent=2))
        (client_out / 'train_history.json').write_text(json.dumps(history, indent=2))
        print(
            f'[client {client_id}] TEST map50={map50:.6f} P={precision:.6f} '
            f'R={recall:.6f} F1={f1:.6f} mIoU={operating["miou"]:.6f} '
            f'FROC-AUC={operating["froc_auc"]:.6f} loss={eval_loss:.6f}',
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

    if args.validation_only:
        summary = {
            'mode': 'validation_only_hyperparameter_sweep',
            'test_data_accessed': False,
            'validation_checkpoint_saved': bool(args.save_validation_checkpoint),
            'started_from_model': args.model,
            'per_client_model_template': args.model_template,
            'model_variant': args.model_variant,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'seed': args.seed,
            'inner_validation': args.inner_validation,
            'inner_val_fraction': args.inner_val_fraction,
            'backbone_lr': args.backbone_lr,
            'late_backbone_lr': args.late_backbone_lr,
            'neck_lr': args.neck_lr,
            'decoder_lr': args.decoder_lr,
            'head_lr': args.head_lr,
            'head_only_decoder_lr': args.head_only_decoder_lr,
            'head_only_head_lr': args.head_only_head_lr,
            'moe_lr': args.moe_lr,
            'fedbn_aware_progressive': args.fedbn_aware_progressive,
            'freeze_fedbn_bn': args.freeze_fedbn_bn,
            'bn_recalibration': args.bn_recalibration,
            'head_only_epochs': args.head_only_epochs,
            'l2sp_lambda': args.l2sp_lambda,
            'train_augmentation': args.train_augmentation,
            'image_size': args.image_size,
            'class_aware_sampling': args.class_aware_sampling,
            'class_aware_max_weight': args.class_aware_max_weight,
            'copy_paste_classes': list(copy_paste_classes),
            'copy_paste_probability': args.copy_paste_probability,
            'grad_accum_steps': args.grad_accum_steps,
            'rare_class_protection': args.rare_class_protection,
            'rare_class_freeze_threshold': args.rare_class_freeze_threshold,
            'rare_class_full_threshold': args.rare_class_full_threshold,
            'freeze_backbone_epochs': args.freeze_backbone_epochs,
            'warmup_epochs': args.warmup_epochs,
            'early_stop_patience': args.early_stop_patience,
            'early_stop_min_delta': args.early_stop_min_delta,
            'bn_momentum': args.bn_momentum,
            'top_k_checkpoints': args.top_k_checkpoints,
            'soup_alphas': args.soup_alphas,
            'clients': rows,
            'mean_val_map50': sum(row['val_map50'] for row in rows) / len(rows),
            'mean_val_f1': sum(row['val_f1'] for row in rows) / len(rows),
            'min_client_val_map50': min(row['val_map50'] for row in rows),
        }
        (out_dir / 'local_finetune_summary.json').write_text(json.dumps(summary, indent=2))
        with (out_dir / 'client_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            'Mode: validation-only hyperparameter sweep',
            'Test data accessed: false',
            f"Mean val mAP50: {summary['mean_val_map50']:.4f}",
            f"Mean val F1: {summary['mean_val_f1']:.4f}",
            f"Worst-client val mAP50: {summary['min_client_val_map50']:.4f}",
        ]
        (out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
        return

    def weighted(metric, weight_key):
        denom = sum(float(r[weight_key]) for r in rows)
        return sum(float(r[metric]) * float(r[weight_key]) for r in rows) / denom if denom else 0.0

    summary = {
        'started_from_model': args.model,
        'per_client_model_template': args.model_template,
        'model_variant': args.model_variant,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'conservative_personalization': args.conservative_personalization,
        'backbone_lr': args.backbone_lr,
        'late_backbone_lr': args.late_backbone_lr,
        'neck_lr': args.neck_lr,
        'decoder_lr': args.decoder_lr,
        'head_lr': args.head_lr,
        'head_only_decoder_lr': args.head_only_decoder_lr,
        'head_only_head_lr': args.head_only_head_lr,
        'moe_lr': args.moe_lr,
        'fedbn_aware_progressive': args.fedbn_aware_progressive,
        'freeze_fedbn_bn': args.freeze_fedbn_bn,
        'bn_recalibration': args.bn_recalibration,
        'head_only_epochs': args.head_only_epochs,
        'l2sp_lambda': args.l2sp_lambda,
        'train_augmentation': args.train_augmentation,
        'image_size': args.image_size,
        'class_aware_sampling': args.class_aware_sampling,
        'class_aware_max_weight': args.class_aware_max_weight,
        'copy_paste_classes': list(copy_paste_classes),
        'copy_paste_probability': args.copy_paste_probability,
        'grad_accum_steps': args.grad_accum_steps,
        'rare_class_protection': args.rare_class_protection,
        'rare_class_freeze_threshold': args.rare_class_freeze_threshold,
        'rare_class_full_threshold': args.rare_class_full_threshold,
        'freeze_backbone_epochs': args.freeze_backbone_epochs,
        'warmup_epochs': args.warmup_epochs,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_min_delta': args.early_stop_min_delta,
        'bn_momentum': args.bn_momentum,
        'top_k_checkpoints': args.top_k_checkpoints,
        'soup_alphas': args.soup_alphas,
        'seed': args.seed,
        'moe_domain_supervision': not args.disable_moe_domain_supervision,
        'strict_checkpoint_load': args.strict_load,
        'num_clients': len(rows),
        'clients': rows,
        'mean_map50': sum(r['map50'] for r in rows) / len(rows),
        'mean_precision': sum(r['precision'] for r in rows) / len(rows),
        'mean_recall': sum(r['recall'] for r in rows) / len(rows),
        'mean_f1': sum(r['f1'] for r in rows) / len(rows),
        'mean_miou': sum(r['miou'] for r in rows) / len(rows),
        'mean_froc_auc': sum(r['froc_auc'] for r in rows) / len(rows),
        'weighted_by_test_images': {
            'map50': weighted('map50', 'test_images'),
            'precision': weighted('precision', 'test_images'),
            'recall': weighted('recall', 'test_images'),
            'f1': weighted('f1', 'test_images'),
            'miou': weighted('miou', 'test_images'),
            'froc_auc': weighted('froc_auc', 'test_images'),
        },
        'weighted_by_test_gt_boxes': {
            'map50': weighted('map50', 'test_gt_boxes'),
            'precision': weighted('precision', 'test_gt_boxes'),
            'recall': weighted('recall', 'test_gt_boxes'),
            'f1': weighted('f1', 'test_gt_boxes'),
            'miou': weighted('miou', 'test_gt_boxes'),
            'froc_auc': weighted('froc_auc', 'test_gt_boxes'),
        },
    }
    (out_dir / 'local_finetune_summary.json').write_text(json.dumps(summary, indent=2))
    with (out_dir / 'client_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"Model: {args.model}",
        f"Epochs: {args.epochs}",
        f"LR: {args.lr}",
        '',
        'Client metrics:',
    ]
    for r in rows:
        lines.append(
            f"client{r['client_id']} {r['client_name']}: map50={r['map50']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}, "
            f"mIoU={r['miou']:.4f}, FROC-AUC={r['froc_auc']:.4f}, "
            f"test_images={r['test_images']}, gt_boxes={r['test_gt_boxes']}"
        )
    lines.extend([
        '',
        f"Mean map50: {summary['mean_map50']:.4f}",
        f"Mean F1: {summary['mean_f1']:.4f}",
        f"Mean mIoU: {summary['mean_miou']:.4f}",
        f"Mean FROC-AUC: {summary['mean_froc_auc']:.4f}",
        f"Weighted by test images map50: {summary['weighted_by_test_images']['map50']:.4f}",
        f"Weighted by test images F1: {summary['weighted_by_test_images']['f1']:.4f}",
        f"Weighted by GT boxes map50: {summary['weighted_by_test_gt_boxes']['map50']:.4f}",
        f"Weighted by GT boxes F1: {summary['weighted_by_test_gt_boxes']['f1']:.4f}",
    ])
    (out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
