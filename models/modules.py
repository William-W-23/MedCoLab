import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -----------------------------------------------------------------------
# 辅助函数与基础组件 (保持原有逻辑优化)
# -----------------------------------------------------------------------

def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    out = x.flatten(-2)
    return out

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        # 这里的 angle 和 decay 是固定参数，与输入尺寸无关
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w], indexing='ij') # 显式指定 indexing='ij' 消除警告
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, slen: Tuple[int], chunkwise_recurrent=True):
        # 动态根据输入的 H, W 生成位置编码
        index = torch.arange(slen[0] * slen[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(slen[0], slen[1], -1)

        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(slen[0], slen[1], -1)

        mask_h = self.generate_1d_decay(slen[0])
        mask_w = self.generate_1d_decay(slen[1])

        retention_rel_pos = ((sin, cos), (mask_h, mask_w))
        return retention_rel_pos

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        # 输入期望: [B, H, W, C] -> 转为 [B, C, H, W] 计算 -> 转回
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class VisionRetentionChunk(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        # 确保 sin/cos 与 x 在同一设备
        sin, cos = sin.to(q.device), cos.to(q.device)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # 宽度方向 Attention
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        # 确保 mask 在同一设备
        mask_w = mask_w.to(qk_mat_w.device)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        # 高度方向 Attention
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        # 确保 mask 在同一设备
        mask_h = mask_h.to(qk_mat_h.device)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

# -----------------------------------------------------------------------
# 【核心封装】即插即用的 Attention 类
# -----------------------------------------------------------------------

class RetNetVisualAttention(nn.Module):
    """
    即插即用的 Vision Retention Attention 模块。
    输入输出格式均为: [Batch, Channel, Height, Width]
    """
    def __init__(self, channels, num_heads=4, initial_value=1, heads_range=3):
        super().__init__()
        # 检查通道数是否能被头数整除
        assert channels % num_heads == 0, f"Channels {channels} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads

        # 内部实例化位置编码生成器
        self.pos_enc_generator = RetNetRelPos2d(
            embed_dim=channels,
            num_heads=num_heads,
            initial_value=initial_value,
            heads_range=heads_range
        )

        # 内部实例化 Retention 计算核心
        self.retention_core = VisionRetentionChunk(
            embed_dim=channels,
            num_heads=num_heads
        )

    def forward(self, x):
        """
        x: [B, C, H, W] (Standard CNN format)
        return: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. 维度变换: [B, C, H, W] -> [B, H, W, C] 以适配 RetNet 内部计算
        x_permuted = x.permute(0, 2, 3, 1)

        # 2. 动态生成位置编码 (基于当前特征图的 H, W)
        #    注意：chunkwise_recurrent=True 是该2D变体的默认模式
        rel_pos = self.pos_enc_generator((H, W), chunkwise_recurrent=True)

        # 3. 执行 Retention 计算
        out = self.retention_core(x_permuted, rel_pos)

        # 4. 维度还原: [B, H, W, C] -> [B, C, H, W]
        out = out.permute(0, 3, 1, 2)

        return out



# -----------------------------------------------------------------------
# Anatomy-aware Sparse Expert Module (ASEM)
# -----------------------------------------------------------------------

class ASEMConvExpert(nn.Module):
    """Lightweight domain expert used by AnatomyAwareSparseExpertModule."""

    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_channels = max(16, int(channels * hidden_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        last_bn = self.net[-1]
        if isinstance(last_bn, nn.BatchNorm2d):
            nn.init.zeros_(last_bn.weight)
            nn.init.zeros_(last_bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASEMRouter(nn.Module):
    """GAP + MLP router that predicts anatomy-domain expert probabilities."""

    def __init__(self, channels: int, num_experts: int = 4, hidden_ratio: float = 0.25):
        super().__init__()
        hidden_channels = max(16, int(channels * hidden_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return F.softmax(self.mlp(pooled), dim=-1)


class AnatomyAwareSparseExpertModule(nn.Module):
    """
    Anatomy-aware sparse MoE residual adapter for medical X-ray detection.

    Domain ids follow the expert order:
      0: Lung, 1: Dental, 2: Bone, 3: General.
    If domain labels are absent, the module still routes features but does not
    contribute a router supervision loss.
    """

    def __init__(
        self,
        channels: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        gamma_init: float = 1e-3,
        router_loss_weight: float = 0.05,
        routing_mode: str = "topk",
        expert_hidden_ratio: float = 0.5,
    ):
        super().__init__()
        if num_experts != 4:
            raise ValueError("ASEM currently expects 4 anatomy experts: lung, dental, bone, general.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}.")
        if routing_mode not in {"soft", "topk"}:
            raise ValueError("routing_mode must be 'soft' or 'topk'.")

        self.channels = channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_loss_weight = router_loss_weight
        self.routing_mode = routing_mode
        self.domain_names = ("lung", "dental", "bone", "general")

        self.router = ASEMRouter(channels, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [ASEMConvExpert(channels, hidden_ratio=expert_hidden_ratio) for _ in range(num_experts)]
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        domain_soft_targets = torch.tensor(
            [
                [0.7, 0.0, 0.2, 0.1],
                [0.0, 0.7, 0.2, 0.1],
                [0.0, 0.0, 0.8, 0.2],
                [0.1, 0.1, 0.2, 0.6],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("domain_soft_targets", domain_soft_targets)

        self.last_router_probs = None
        self.last_routing_weights = None
        self.last_router_loss = None

    def set_routing_mode(self, routing_mode: str) -> None:
        if routing_mode not in {"soft", "topk"}:
            raise ValueError("routing_mode must be 'soft' or 'topk'.")
        self.routing_mode = routing_mode

    def _topk_weights(self, probs: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(probs, k=self.top_k, dim=-1)
        weights = torch.zeros_like(probs)
        weights.scatter_(1, indices, values)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _labels_to_soft_targets(self, domain_labels, device) -> torch.Tensor | None:
        if domain_labels is None:
            return None
        if isinstance(domain_labels, torch.Tensor):
            labels = domain_labels.detach().to(device=device, dtype=torch.long).flatten()
        else:
            label_ids = []
            for item in domain_labels:
                if isinstance(item, str):
                    value = item.strip().lower()
                    label_ids.append(self.domain_names.index(value) if value in self.domain_names else -1)
                else:
                    label_ids.append(int(item))
            labels = torch.tensor(label_ids, device=device, dtype=torch.long)

        valid = (labels >= 0) & (labels < self.domain_soft_targets.shape[0])
        if not torch.any(valid):
            return None
        targets = torch.zeros(labels.numel(), self.num_experts, device=device, dtype=torch.float32)
        targets[valid] = self.domain_soft_targets[labels[valid]].to(device=device, dtype=torch.float32)
        return targets, valid

    def _router_loss(self, probs: torch.Tensor, domain_labels) -> torch.Tensor:
        target_info = self._labels_to_soft_targets(domain_labels, probs.device)
        if target_info is None:
            return probs.new_zeros(())
        targets, valid = target_info
        log_probs = torch.log(probs.clamp_min(1e-6))
        loss = -(targets[valid] * log_probs[valid]).sum(dim=-1).mean()
        return loss * float(self.router_loss_weight)

    def forward(self, x: torch.Tensor, domain_labels=None, routing_mode: str | None = None) -> torch.Tensor:
        probs = self.router(x)
        mode = routing_mode or self.routing_mode
        if mode == "soft":
            routing_weights = probs
        elif mode == "topk":
            sparse_weights = self._topk_weights(probs)
            # Preserve sparse top-k values in the forward pass, while allowing
            # task loss gradients to reach the router without fake domain labels.
            routing_weights = sparse_weights + probs - probs.detach()
        else:
            raise ValueError("routing_mode must be 'soft' or 'topk'.")

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_mix = (expert_outputs * routing_weights[:, :, None, None, None]).sum(dim=1)
        out = x + self.gamma * expert_mix

        self.last_router_probs = probs.detach()
        self.last_routing_weights = routing_weights.detach()
        self.last_router_loss = self._router_loss(probs, domain_labels)
        return out

if __name__ == '__main__':
    # --- 测试即插即用特性 ---

    # 模拟一个常见的 CNN 特征图输入: Batch=2, Channels=64, Height=32, Width=32
    input_tensor = torch.randn(2, 64, 32, 32)

    # 初始化模块 (只需要指定通道数，类似初始化 Conv2d)
    attention_block = RetNetVisualAttention(channels=64, num_heads=4)

    # 前向传播
    output_tensor = attention_block(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    # 验证输入输出是否一致
    assert input_tensor.shape == output_tensor.shape
    print("✅ Plug-and-play validation passed!")
