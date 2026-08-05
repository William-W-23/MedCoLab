import torch
from pathlib import Path
from PIL import Image, ImageDraw

def visualize_batch(batch, save_dir="logs/vis", prefix="", class_names=None, max_images=None):
    """
    将当前 batch 的 GT 以框绘制到图像并保存，用于目检数据是否正确。
    - batch: 来自 rtdetr_collate_fn 的 dict
    - save_dir: 输出目录
    - prefix: 文件名前缀，如 "epoch0_batch3"
    - class_names: 可选，类别名称列表；未提供则写入类别ID
    - max_images: 可选，最多保存的图数
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    images = batch['images']          # [B, 3, H, W]
    bboxes = batch['bboxes']          # [N, 4] (cx, cy, w, h) 归一化到 [0,1]
    cls     = batch['cls']            # [N] Long
    batch_idx = batch['batch_idx']    # [N] Long
    gt_groups = batch['gt_groups']    # List[int], 每张图的 GT 数量

    B, C, H, W = images.shape
    # 验证 batch_idx 与 gt_groups 的一致性（便于快速定位偏移问题）
    counts = torch.bincount(batch_idx, minlength=B).tolist()
    if counts != gt_groups:
        print(f"[VIS] gt_groups 与 batch_idx 计数不一致: counts={counts}, gt_groups={gt_groups}")

    num_to_save = B if max_images is None else min(B, max_images)

    for i in range(num_to_save):
        # 1) 将图像转为 PIL
        img_t = images[i].detach().cpu()
        # 容错归一化：若数值大于 1.5，认为是 0-255；否则按 0-1 处理
        if img_t.max().item() > 1.5:
            img_t = img_t.clamp(0, 255)
        else:
            img_t = (img_t * 255.0).clamp(0, 255)
        img_np = img_t.permute(1, 2, 0).byte().numpy()  # [H, W, 3]
        pil = Image.fromarray(img_np, mode="RGB")
        draw = ImageDraw.Draw(pil)

        out_path = Path(save_dir) / f"{prefix}_img{i}_ori.png"
        pil.save(out_path)

        # 2) 取属于第 i 张图的框与类别
        mask = (batch_idx == i)
        bbs_i = bboxes[mask].detach().cpu()
        cls_i = cls[mask].detach().cpu()

        # 3) 绘制框（cx,cy,w,h → xyxy）
        for k in range(bbs_i.shape[0]):
            cx, cy, w_norm, h_norm = bbs_i[k].tolist()
            x1 = (cx - w_norm / 2.0) * W
            y1 = (cy - h_norm / 2.0) * H
            x2 = (cx + w_norm / 2.0) * W
            y2 = (cy + h_norm / 2.0) * H

            # 防越界
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))

            # 颜色简单按类别ID分配（可扩展为更丰富的调色板）
            color = (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_str = str(int(cls_i[k]))
            if class_names and 0 <= int(cls_i[k]) < len(class_names):
                label_str = class_names[int(cls_i[k])]
            draw.text((x1 + 2, y1 + 2), label_str, fill=color)

        # 4) 保存
        out_path = Path(save_dir) / f"{prefix}_img{i}.png"
        pil.save(out_path)