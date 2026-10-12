"""vyton.packs.bench_backbone — 基座评估基准(换基座决策工具)
用法:python bench_backbone.py <backbone名>
输出:标准测集上各任务头的可达精度+速度/体积 → 换基座决策表
"""
import torch, torch.nn as nn, glob, json, os, time
import numpy as np
from PIL import Image
from torchvision import models as tvm, transforms

# 标准测集(1000张,各源分层抽样,冻结)
def build_testset(n=1000):
    srcs = {"pin": r"B:\mode\pinterest\outfit_imgs\*.jpg",
            "yt": r"B:\mode\yt_outfit\*.jpg",
            "mt": r"B:\mode\glow_data\mt\extracted\makeup\MAKE\*.jpg"}
    out = []
    per = n // len(srcs)
    for k, pat in srcs.items():
        fs = sorted(glob.glob(pat))
        step = max(1, len(fs) // per)
        out += [(f, k) for f in fs[::step][:per]]
    return out

BB = {
    "enb3": lambda: (lambda m: (m.classifier.__setitem__(1, nn.Identity()) or m))(tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)),
    "mbv3s": lambda: (lambda m: (setattr(m, "fc", nn.Identity()) or m))(tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)),
    "resnet18": lambda: (lambda m: (setattr(m, "fc", nn.Identity()) or m))(tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)),
    "mbv3l": lambda: (lambda m: (setattr(m, "fc", nn.Identity()) or m))(tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1)),
}

def bench(name, n=300):
    net = BB[name]().cuda().eval()
    tf = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    test = build_testset(n)
    # 1.特征质量代理:源分类线性探针(pin/yt/mt三域可分性)
    X, Y = [], []
    with torch.no_grad():
        for f, k in test:
            try:
                x = tf(Image.open(f).convert("RGB")).unsqueeze(0).cuda()
                X.append(net.features(x).flatten(1).cpu().numpy()[0] if hasattr(net, "features") and name=="enb3" else net(x).flatten(1).cpu().numpy()[0])
                Y.append({"pin":0,"yt":1,"mt":2}[k])
            except Exception:
                continue
    X, Y = np.array(X), np.array(Y)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    acc = cross_val_score(LogisticRegression(max_iter=300), X, Y, cv=3).mean()
    # 2.速度/体积
    x = torch.randn(1,3,160,160).cuda()
    with torch.no_grad():
        for _ in range(5): net(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(30): net(x)
        torch.cuda.synchronize()
    ms = (time.time()-t0)/30*1000
    par = sum(p.numel() for p in net.parameters())/1e6
    return {"backbone": name, "feat_dim": X.shape[1], "probe_acc": round(acc, 3), "gpu_ms": round(ms, 1), "params_m": round(par, 1)}

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["mbv3s", "resnet18", "mbv3l", "enb3"]
    results = [bench(n) for n in names]
    print(json.dumps(results, ensure_ascii=False, indent=1))
    json.dump(results, open(r"B:\mode\vyton\packs\backbone_bench.json", "w"))
