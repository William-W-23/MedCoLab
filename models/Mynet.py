import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保环境中有 ultralytics 库
from ultralytics.nn.modules.block import *
from ultralytics.nn.modules.conv import *
from ultralytics.nn.modules.head import *
from ultralytics.nn.modules.transformer import AIFI

from .modules import RetNetVisualAttention, AnatomyAwareSparseExpertModule

class RTDETR_L(nn.Module):
    def __init__(self, nc=20):
        super().__init__()
        self.nc = nc

        # 使用 nn.Sequential 对齐官方权重的 model.0 ~ model.28
        self.model = nn.Sequential(
            # ================= Backbone (0-9) =================
            HGStem(c1=3, cm=32, c2=48),                                                   # [0] Stem
            HGBlock(c1=48, cm=48, c2=128, k=3, n=6),                                      # [1] Stage 1
            DWConv(c1=128, c2=128, k=3, s=2, d=1, act=False),                             # [2] Downsample
            HGBlock(c1=128, cm=96, c2=512, k=3, n=6),                                     # [3] Stage 2 (Output P3)
            DWConv(c1=512, c2=512, k=3, s=2, d=1, act=False),                             # [4] Downsample
            HGBlock(c1=512, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=False),   # [5] Stage 3.1
            HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [6] Stage 3.2
            HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [7] Stage 3.3 (Output P4)
            DWConv(c1=1024, c2=1024, k=3, s=2, d=1, act=False),                           # [8] Downsample
            HGBlock(c1=1024, cm=384, c2=2048, k=5, n=6, lightconv=True, shortcut=False),  # [9] Stage 4 (Output P5)

            # ================= Encoder / FPN (10-21) =================
            Conv(c1=2048, c2=256, k=1, s=1, act=False),      # [10] Proj P5
            AIFI(c1=256, cm=1024, num_heads=8),              # [11] AIFI
            Conv(c1=256, c2=256, k=1, s=1),                  # [12] Lat P5 (Y5)
            nn.Upsample(scale_factor=2, mode="nearest"),     # [13] Upsample
            Conv(c1=1024, c2=256, k=1, s=1, act=False),      # [14] Proj P4
            Concat(dimension=1),                             # [15] Concat (P5_up + P4_proj)
            RepC3(c1=512, c2=256, n=3),                      # [16] Fusion P4
            Conv(c1=256, c2=256, k=1, s=1),                  # [17] Lat P4 (Y4)
            nn.Upsample(scale_factor=2, mode="nearest"),     # [18] Upsample
            Conv(c1=512, c2=256, k=1, s=1, act=False),       # [19] Proj P3
            Concat(dimension=1),                             # [20] Concat (P4_up + P3_proj)
            RepC3(c1=512, c2=256, n=3),                      # [21] Fusion P3 (Output X3)

            # ================= PAN / Bottom-up (22-27) =================
            Conv(c1=256, c2=256, k=3, s=2),                  # [22] Downsample X3
            Concat(dimension=1),                             # [23] Concat (X3_down + Y4)
            RepC3(c1=512, c2=256, n=3),                      # [24] Fusion F4 (Output F4)
            Conv(c1=256, c2=256, k=3, s=2),                  # [25] Downsample F4
            Concat(dimension=1),                             # [26] Concat (F4_down + Y5)
            RepC3(c1=512, c2=256, n=3),                      # [27] Fusion F5 (Output F5)

            # ================= Decoder (28) =================
            RTDETRDecoder(nc=nc, ch=(256, 256, 256))         # [28] Transformer Decoder
        )

    def forward(self, x, batch=None):
        m = self.model

        # --- Backbone ---
        x = m[0](x)
        x = m[1](x)
        x = m[2](x)
        p3 = m[3](x)         # Save P3
        x = m[4](p3)
        x = m[5](x)
        x = m[6](x)
        p4 = m[7](x)         # Save P4
        x = m[8](p4)
        p5 = m[9](x)         # Save P5

        # --- Encoder (Top-down) ---
        # Process P5
        x = m[10](p5)
        x = m[11](x)
        y5 = m[12](x)        # Save Y5 for PAN

        # Process P4
        up_p5 = m[13](y5)
        proj_p4 = m[14](p4)
        x = m[15]([up_p5, proj_p4])
        x = m[16](x)
        y4 = m[17](x)        # Save Y4 for PAN

        # Process P3
        up_p4 = m[18](y4)
        proj_p3 = m[19](p3)
        x = m[20]([up_p4, proj_p3])
        x3 = m[21](x)        # Output X3

        # --- PAN (Bottom-up) ---
        down_x3 = m[22](x3)
        x = m[23]([down_x3, y4])
        f4 = m[24](x)        # Output F4

        down_f4 = m[25](f4)
        x = m[26]([down_f4, y5])
        f5 = m[27](x)        # Output F5

        # --- Decoder ---
        # Input features: [X3, F4, F5]
        return m[28]([x3, f4, f5], batch=batch)



class RTDETR_L_WithASEM(RTDETR_L):
    """RT-DETR-L with an Anatomy-aware Sparse Expert Module after P5 projection."""

    def __init__(
        self,
        nc=20,
        asem_top_k=2,
        asem_router_loss_weight=0.05,
        asem_routing="topk",
    ):
        super().__init__(nc=nc)
        self.asem_p5 = AnatomyAwareSparseExpertModule(
            channels=256,
            num_experts=4,
            top_k=asem_top_k,
            gamma_init=1e-3,
            router_loss_weight=asem_router_loss_weight,
            routing_mode=asem_routing,
        )
        self._last_aux_loss = None

    def set_asem_routing(self, routing_mode: str) -> None:
        self.asem_p5.set_routing_mode(routing_mode)

    def get_aux_loss(self):
        if self._last_aux_loss is None:
            return None
        return self._last_aux_loss

    @staticmethod
    def _extract_domain_labels(batch):
        if not isinstance(batch, dict):
            return None
        for key in ("domain_id", "domain_label", "domain"):
            if key in batch:
                return batch[key]
        return None

    def forward(self, x, batch=None):
        m = self.model

        # Backbone
        x = m[0](x)
        x = m[1](x)
        x = m[2](x)
        p3 = m[3](x)
        x = m[4](p3)
        x = m[5](x)
        x = m[6](x)
        p4 = m[7](x)
        x = m[8](p4)
        p5 = m[9](x)

        # Encoder / FPN with ASEM after P5 projection
        x = m[10](p5)
        x = self.asem_p5(x, domain_labels=self._extract_domain_labels(batch))
        self._last_aux_loss = self.asem_p5.last_router_loss
        x = m[11](x)
        y5 = m[12](x)

        up_p5 = m[13](y5)
        proj_p4 = m[14](p4)
        x = m[15]([up_p5, proj_p4])
        x = m[16](x)
        y4 = m[17](x)

        up_p4 = m[18](y4)
        proj_p3 = m[19](p3)
        x = m[20]([up_p4, proj_p3])
        x3 = m[21](x)

        # PAN
        down_x3 = m[22](x3)
        x = m[23]([down_x3, y4])
        f4 = m[24](x)

        down_f4 = m[25](f4)
        x = m[26]([down_f4, y5])
        f5 = m[27](x)

        return m[28]([x3, f4, f5], batch=batch)


# class RTDETR_L_WithAttention(nn.Module):
#     def __init__(self, nc=20):
#         super().__init__()
#         self.nc = nc

#         # === 1. 初始化 RetNet 模块 ===
#         # 放在此处不影响 self.model 的索引顺序
#         # 输入通道为 256 (RT-DETR Neck 的标准输出通道)
#         self.retnet_f5 = RetNetVisualAttention(channels=256, num_heads=4)
#         # ===========================

#         # 使用 nn.Sequential 对齐官方权重的 model.0 ~ model.28
#         self.model = nn.Sequential(
#             # ================= Backbone (0-9) =================
#             HGStem(c1=3, cm=32, c2=48),                                                   # [0] Stem
#             HGBlock(c1=48, cm=48, c2=128, k=3, n=6),                                      # [1] Stage 1
#             DWConv(c1=128, c2=128, k=3, s=2, d=1, act=False),                             # [2] Downsample
#             HGBlock(c1=128, cm=96, c2=512, k=3, n=6),                                     # [3] Stage 2 (Output P3)
#             DWConv(c1=512, c2=512, k=3, s=2, d=1, act=False),                             # [4] Downsample
#             HGBlock(c1=512, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=False),   # [5] Stage 3.1
#             HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [6] Stage 3.2
#             HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [7] Stage 3.3 (Output P4)
#             DWConv(c1=1024, c2=1024, k=3, s=2, d=1, act=False),                           # [8] Downsample
#             HGBlock(c1=1024, cm=384, c2=2048, k=5, n=6, lightconv=True, shortcut=False),  # [9] Stage 4 (Output P5)

#             # ================= Encoder / FPN (10-21) =================
#             Conv(c1=2048, c2=256, k=1, s=1, act=False),      # [10] Proj P5
#             AIFI(c1=256, cm=1024, num_heads=8),              # [11] AIFI
#             Conv(c1=256, c2=256, k=1, s=1),                  # [12] Lat P5 (Y5)
#             nn.Upsample(scale_factor=2, mode="nearest"),     # [13] Upsample
#             Conv(c1=1024, c2=256, k=1, s=1, act=False),      # [14] Proj P4
#             Concat(dimension=1),                             # [15] Concat (P5_up + P4_proj)
#             RepC3(c1=512, c2=256, n=3),                      # [16] Fusion P4
#             Conv(c1=256, c2=256, k=1, s=1),                  # [17] Lat P4 (Y4)
#             nn.Upsample(scale_factor=2, mode="nearest"),     # [18] Upsample
#             Conv(c1=512, c2=256, k=1, s=1, act=False),       # [19] Proj P3
#             Concat(dimension=1),                             # [20] Concat (P4_up + P3_proj)
#             RepC3(c1=512, c2=256, n=3),                      # [21] Fusion P3 (Output X3)

#             # ================= PAN / Bottom-up (22-27) =================
#             Conv(c1=256, c2=256, k=3, s=2),                  # [22] Downsample X3
#             Concat(dimension=1),                             # [23] Concat (X3_down + Y4)
#             RepC3(c1=512, c2=256, n=3),                      # [24] Fusion F4 (Output F4)
#             Conv(c1=256, c2=256, k=3, s=2),                  # [25] Downsample F4
#             Concat(dimension=1),                             # [26] Concat (F4_down + Y5)
#             RepC3(c1=512, c2=256, n=3),                      # [27] Fusion F5 (Output F5)

#             # ================= Decoder (28) =================
#             RTDETRDecoder(nc=nc, ch=(256, 256, 256))         # [28] Transformer Decoder
#         )

#     def forward(self, x, batch=None):
#         m = self.model

#         # --- Backbone ---
#         x = m[0](x)
#         x = m[1](x)
#         x = m[2](x)
#         p3 = m[3](x)
#         x = m[4](p3)
#         x = m[5](x)
#         x = m[6](x)
#         p4 = m[7](x)
#         x = m[8](p4)
#         p5 = m[9](x)

#         # --- Encoder (Top-down) ---
#         x = m[10](p5)
#         x = m[11](x)
#         y5 = m[12](x)

#         up_p5 = m[13](y5)
#         proj_p4 = m[14](p4)
#         x = m[15]([up_p5, proj_p4])
#         x = m[16](x)
#         y4 = m[17](x)

#         up_p4 = m[18](y4)
#         proj_p3 = m[19](p3)
#         x = m[20]([up_p4, proj_p3])
#         x3 = m[21](x)        # Output X3 (Smallest stride, largest scale)

#         # --- PAN (Bottom-up) ---
#         down_x3 = m[22](x3)
#         x = m[23]([down_x3, y4])
#         f4 = m[24](x)        # Output F4 (Medium scale)

#         down_f4 = m[25](f4)
#         x = m[26]([down_f4, y5])
#         f5 = m[27](x)        # Output F5 (Largest stride, smallest scale)

#         # === 2. 插入点：增强 F5 特征 ===
#         # F5 是最顶层的语义特征，这里加入 RetNet 可以增强全局上下文
#         # 帮助 Decoder 更好地 Query 到大物体和复杂场景信息
#         f5 = self.retnet_f5(f5)
#         # ============================

#         # --- Decoder ---
#         return m[28]([x3, f4, f5], batch=batch)

class RTDETR_L_WithAttention(nn.Module):
    def __init__(self, nc=20):
        super().__init__()
        self.nc = nc

        # ============================================================
        # 1. 初始化 RetNet 模块
        # 位置：P5 投影层之后 (256通道)
        # ============================================================
        self.retnet_p5 = RetNetVisualAttention(channels=256, num_heads=4)
        self.retnet_scale = nn.Parameter(torch.tensor(0.001), requires_grad=True)

        # 建议：在这里对 retnet_p5 进行 "零初始化" (Zero Initialization)
        # 这样在训练刚开始时，retnet_p5 输出接近 0，整个结构等同于原版模型
        # for m in self.retnet_p5.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.constant_(m.weight, 0)
        #         if m.bias is not None: nn.init.constant_(m.bias, 0)

        # ============================================================
        # 2. 定义模型结构
        # ============================================================
        self.model = nn.Sequential(
            # ----------------- Backbone (0-9) -----------------
            HGStem(c1=3, cm=32, c2=48),                                           # [0]
            HGBlock(c1=48, cm=48, c2=128, k=3, n=6),                              # [1]
            DWConv(c1=128, c2=128, k=3, s=2, d=1, act=False),                     # [2]
            HGBlock(c1=128, cm=96, c2=512, k=3, n=6),                             # [3] P3
            DWConv(c1=512, c2=512, k=3, s=2, d=1, act=False),                     # [4]
            HGBlock(c1=512, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=False),   # [5]
            HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [6]
            HGBlock(c1=1024, cm=192, c2=1024, k=5, n=6, lightconv=True, shortcut=True),   # [7] P4
            DWConv(c1=1024, c2=1024, k=3, s=2, d=1, act=False),                   # [8]
            HGBlock(c1=1024, cm=384, c2=2048, k=5, n=6, lightconv=True, shortcut=False),  # [9] P5

            # ----------------- Encoder / FPN (10-21) -----------------
            Conv(c1=2048, c2=256, k=1, s=1, act=False),      # [10] Proj P5
            AIFI(c1=256, cm=1024, num_heads=8),              # [11] AIFI
            Conv(c1=256, c2=256, k=1, s=1),                  # [12] Lat P5
            nn.Upsample(scale_factor=2, mode="nearest"),     # [13]
            Conv(c1=1024, c2=256, k=1, s=1, act=False),      # [14] Proj P4
            Concat(dimension=1),                             # [15]
            RepC3(c1=512, c2=256, n=3),                      # [16] Fusion P4
            Conv(c1=256, c2=256, k=1, s=1),                  # [17] Lat P4
            nn.Upsample(scale_factor=2, mode="nearest"),     # [18]
            Conv(c1=512, c2=256, k=1, s=1, act=False),       # [19] Proj P3
            Concat(dimension=1),                             # [20]
            RepC3(c1=512, c2=256, n=3),                      # [21] X3

            # ----------------- PAN / Bottom-up (22-27) -----------------
            Conv(c1=256, c2=256, k=3, s=2),                  # [22]
            Concat(dimension=1),                             # [23]
            RepC3(c1=512, c2=256, n=3),                      # [24] F4
            Conv(c1=256, c2=256, k=3, s=2),                  # [25]
            Concat(dimension=1),                             # [26]
            RepC3(c1=512, c2=256, n=3),                      # [27] F5

            # ----------------- Decoder (28) -----------------
            RTDETRDecoder(nc=nc, ch=(256, 256, 256))         # [28]
        )

    def forward(self, x, batch=None):
        m = self.model

        # ----------------- Backbone Forward -----------------
        x = m[0](x)
        x = m[1](x)
        x = m[2](x)
        p3 = m[3](x)
        x = m[4](p3)
        x = m[5](x)
        x = m[6](x)
        p4 = m[7](x)
        x = m[8](p4)
        p5 = m[9](x)

        # ----------------- Encoder Forward -----------------
        # 1. 先将 P5 (2048) 投影到 (256)
        x = m[10](p5)

        # ============================================================
        # 【关键修改】 3. 插入 RetNet 并使用残差连接 (Add Shortcut)
        # ============================================================
        # 原始特征 x 与 Attention 特征相加。
        # 即使 RetNet 还没训练好，x 的信息也不会丢失。
        x = x + self.retnet_p5(x) * self.retnet_scale
        # ============================================================

        # 4. 继续进入 AIFI
        # (注：如果加上残差后效果依然不好，可尝试注释掉下面这行 m[11] 跳过 AIFI)
        x = m[11](x)

        y5 = m[12](x) # Lat P5

        # ... 后续 FPN/PAN 逻辑保持不变 ...
        up_p5 = m[13](y5)
        proj_p4 = m[14](p4)
        x = m[15]([up_p5, proj_p4])
        x = m[16](x)
        y4 = m[17](x)

        up_p4 = m[18](y4)
        proj_p3 = m[19](p3)
        x = m[20]([up_p4, proj_p3])
        x3 = m[21](x)

        # ----------------- PAN Forward -----------------
        down_x3 = m[22](x3)
        x = m[23]([down_x3, y4])
        f4 = m[24](x)

        down_f4 = m[25](f4)
        x = m[26]([down_f4, y5])
        f5 = m[27](x)

        # ----------------- Decoder Forward -----------------
        return m[28]([x3, f4, f5], batch=batch)

if __name__ == "__main__":
    # 快速测试
    net = RTDETR_L(nc=80)
    print(f"✅ 模型实例化成功，Sequential 层数: {len(net.model)}")

    # 模拟输入
    x = torch.randn(1, 3, 640, 640)
    y = net(x)
    print(f"✅ 推理测试成功")