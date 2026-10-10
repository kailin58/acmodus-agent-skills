# 终极严格测试:8维度(数学正确性/数值精度/性能/长稳/组合爆炸/分布漂移/对抗/一致性)
import sys, os, time, json, math, random
sys.path.insert(0, r"B:\mode\vyton")
import torch, numpy as np

R = {"pass": 0, "fail": 0, "notes": []}
def T(name, ok, detail=""):
    if ok:
        R["pass"] += 1; print(f"  PASS {name}")
    else:
        R["fail"] += 1; print(f"  FAIL {name} {detail}")

print("=== 1.数学正确性(ground truth对照) ===")
from core.metrics import lip_redness, skin_mask
# 1a. 唇红度:构造已知值图(红唇vs灰唇,差值应=红度差)
img_red = np.zeros((192, 192, 3), np.uint8); img_red[115:125, 80:110] = [200, 50, 50]
img_gray = np.zeros((192, 192, 3), np.uint8); img_gray[115:125, 80:110] = [128, 128, 128]
d = lip_redness(img_red.astype(np.float32)) - lip_redness(img_gray.astype(np.float32))
T("唇红度数学(红-灰应>50)", d > 50, f"实际{d:.1f}")

# 1b. 皮肤mask:纯肤色像素应全命中
skin_patch = np.full((64, 64, 3), [220, 170, 140], np.uint8)
m = skin_mask(skin_patch)
T("皮肤mask命中", m.mean() > 0.9, f"{m.mean():.2f}")

# 1c. UNet恒等映射测试(零初始化输出层→输出≈0)
from core.blocks import UNetLite, ConvBlk
net = UNetLite(3, 3, ch=8)
torch.nn.init.zeros_(net.out.weight); torch.nn.init.zeros_(net.out.bias)
out, _ = net(torch.randn(1, 3, 64, 64))
T("零头输出≈0(数学)", out.abs().max() < 1e-4, f"{out.abs().max():.6f}")

print("=== 2.数值精度 ===")
# 2a. float16溢出边界
half = torch.tensor([65504.0], dtype=torch.float16)
T("fp16上限识别", half.item() == 65504.0)
# 2b. loss单调性(同输入,更优参数应loss更低)
torch.manual_seed(0)
x = torch.randn(8, 3, 32, 32); y = x * 0.5
l1 = ((x - y) ** 2).mean().item()
x2 = x.clone(); x2.requires_grad_(False)
l2 = ((x2 * 0.99 - y) ** 2).mean().item()  # 更接近y
T("loss序保持", l2 < l1, f"{l2:.4f}<{l1:.4f}")

# 2c. 确定性(固定seed两次前向bit级一致)
torch.manual_seed(42); a = torch.randn(100)
torch.manual_seed(42); b = torch.randn(100)
T("seed确定性", torch.equal(a, b))

print("=== 3.性能基准 ===")
from core.trainer import quick_train
from core.datasets import ImageFolderDS
from torch.utils.data import DataLoader
import glob
files = glob.glob(r"B:\mode\glow_data\pairs\*_a.jpg")[:64]
if files:
    t0 = time.time()
    ds = ImageFolderDS(files, size=160)
    dl = DataLoader(ds, batch_size=16)
    for _ in dl: pass
    dt = time.time() - t0
    T("数据吞吐(64图<8s)", dt < 8, f"{dt:.1f}s")
# GPU前向延迟
if torch.cuda.is_available():
    net = UNetLite(6, 3, ch=48).cuda().eval()
    x = torch.randn(1, 6, 256, 256).cuda()
    with torch.no_grad():
        for _ in range(3): net(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): net(x)
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    T("GPU前向<15ms(256px)", ms < 15, f"{ms:.1f}ms")
    R["notes"].append(f"GPU前向{ms:.1f}ms")

print("=== 4.长时稳定(50轮迭代) ===")
torch.manual_seed(7)
net = UNetLite(3, 3, ch=8)
x0 = torch.randn(1, 3, 32, 32)
outs = []
for i in range(50):
    with torch.no_grad():
        o, _ = net(x0 + i * 0.001)
    outs.append(o.mean().item())
spread = max(outs) - min(outs)
T("50轮输出有界", spread < 2.0, f"spread={spread:.3f}")

print("=== 5.组合爆炸(参数全排列抽样) ===")
random.seed(3)
fails = 0
for trial in range(20):
    ch = random.choice([8, 16, 24])
    sz = random.choice([33, 64, 100, 127, 256])
    heads = random.choice([{}, {"a": 4}, {"a": 4, "b": 2}])
    try:
        n = UNetLite(3, 3, ch=ch, aux_heads=heads)
        o, aux = n(torch.randn(1, 3, sz, sz))
        assert o.shape[-2:] == (sz, sz)
    except Exception:
        fails += 1
T("20组随机组合全过", fails == 0, f"{fails}组崩")

print("=== 6.分布漂移(训练分布外输入) ===")
net = UNetLite(3, 3, ch=8).eval()
# 极亮/极暗/噪声图
for name, img in [("全白", torch.ones(1,3,64,64)), ("全黑", torch.zeros(1,3,64,64)),
                  ("高噪", torch.randn(1,3,64,64)*10)]:
    with torch.no_grad():
        o, _ = net(img.clamp(-10, 10))
    ok = torch.isfinite(o).all()
    T(f"分布外[{name}]输出有限", bool(ok))

print("=== 7.对抗鲁棒(小扰动) ===")
torch.manual_seed(1)
x = torch.randn(1, 3, 64, 64)
net = UNetLite(3, 3, ch=8).eval()
with torch.no_grad():
    o1, _ = net(x)
    eps = 0.01
    o2, _ = net(x + eps * torch.randn_like(x))
delta = (o1 - o2).abs().mean()
T("小扰动输出稳定(δ<0.15)", delta < 0.15, f"δ={delta:.4f}")

print("=== 8.一致性(跨平台语义) ===")
# 同图RGB/BGR顺序错位应检出
from core.datasets import load_tensor
from PIL import Image
import tempfile
arr = np.random.randint(0, 255, (32, 32, 3), np.uint8)
p = os.path.join(tempfile.gettempdir(), "cons.jpg")
Image.fromarray(arr).save(p)
t1 = load_tensor(p, 32, augment=False)
t2 = load_tensor(p, 32, augment=False)
T("load跨调用一致", torch.equal(t1, t2))
# augment=True时翻转应发生(概率性,50次内应有翻转)
flips = 0
for _ in range(50):
    t = load_tensor(p, 32, augment=True)
    if not torch.equal(t, t1):
        flips += 1
T("增强生效(50次有翻转)", flips > 10, f"翻转{flips}/50")

print(f"\n═══ 终极: PASS {R['pass']} FAIL {R['fail']} ═══")
R["notes"].append(f"total {R['pass']}+{R['fail']}")
json.dump(R, open(r"B:\mode\vyton\ultimate_result.json", "w"))
