import torch
import torch.nn as nn

from .Mynet import RTDETR_L


class RTDETR_L_Classifier(RTDETR_L):
    """RT-DETR-L backbone/neck with the detection decoder replaced by a classifier.

    The module keeps RT-DETR layers 0..27 so pretrained detector weights can be
    reused. Layer 28 is replaced because classification needs image-level logits
    instead of detection queries and boxes.
    """

    def __init__(self, nc: int = 2, dropout: float = 0.1):
        super().__init__(nc=nc)
        self.nc = nc
        self.model[28] = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 3, nc),
        )

    def forward_features(self, x: torch.Tensor):
        m = self.model

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

        x = m[10](p5)
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

        down_x3 = m[22](x3)
        x = m[23]([down_x3, y4])
        f4 = m[24](x)

        down_f4 = m[25](f4)
        x = m[26]([down_f4, y5])
        f5 = m[27](x)

        return x3, f4, f5

    def forward(self, x: torch.Tensor, batch=None) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = [self.pool(feature).flatten(1) for feature in features]
        return self.classifier(torch.cat(pooled, dim=1))
