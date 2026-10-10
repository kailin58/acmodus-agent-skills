"""vyton.core.blocks — 模型积木库(唯一实现,处处复用)
用法:
    from vyton.core.blocks import ConvBlk, UNetLite, MultiHeadCls
    net = UNetLite(in_ch=6, out_ch=3, ch=48)      # 换装/化妆回归
    cls = MultiHeadCls(backbone="enb3", heads={"cat":9,"col":12})  # 多头分类
"""
import torch, torch.nn as nn
import torchvision.models as tvm


def ConvBlk(i, o, norm="group"):
    """卷积块(全项目唯一)"""
    n = nn.GroupNorm(8, o) if norm == "group" else nn.InstanceNorm2d(o)
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), n, nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNetLite(nn.Module):
    """轻量UNet(Try学生/Glow共用架构)"""
    def __init__(self, in_ch, out_ch, ch=24, aux_heads=None):
        super().__init__()
        self.d1 = ConvBlk(in_ch, ch); self.d2 = ConvBlk(ch, ch*2); self.d3 = ConvBlk(ch*2, ch*3)
        self.mid = ConvBlk(ch*3, ch*3)
        self.u3 = ConvBlk(ch*6, ch*2); self.u2 = ConvBlk(ch*4, ch); self.u1 = ConvBlk(ch*2, ch)
        self.out = nn.Conv2d(ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.aux = nn.ModuleDict({k: nn.Linear(ch*3, v) for k, v in (aux_heads or {}).items()})

    def forward(self, x):
        H, W = x.shape[-2:]
        h1 = self.d1(x); h2 = self.d2(self.pool(h1)); h3 = self.d3(self.pool(h2))
        m = self.mid(self.pool(h3))
        aux = {k: head(m.mean((2, 3))) for k, head in self.aux.items()}
        def up_to(t, ref):
            return nn.functional.interpolate(t, size=ref.shape[-2:], mode="nearest")
        u3 = self.u3(torch.cat([up_to(m, h3), h3], 1))
        u2 = self.u2(torch.cat([up_to(u3, h2), h2], 1))
        u1 = self.u1(torch.cat([up_to(u2, h1), h1], 1))
        out = self.out(u1)
        if out.shape[-2:] != (H, W):
            out = nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return torch.tanh(out), aux


class MultiHeadCls(nn.Module):
    """多任务分类头(Style四头统一)"""
    def __init__(self, backbone="enb3", heads=None, pretrained=True):
        super().__init__()
        w = {"enb3": (tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)}[backbone]
        self.back = w[0](weights=w[1] if pretrained else None)
        f = self.back.classifier[1].in_features
        self.back.classifier = nn.Identity()
        self.heads = nn.ModuleDict({k: nn.Linear(f, v) for k, v in heads.items()})

    def forward(self, x):
        h = self.back.features(x)
        h = nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return {k: head(h) for k, head in self.heads.items()}
