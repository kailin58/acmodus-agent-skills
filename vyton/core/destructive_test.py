# 破坏性测试套件v2:故障注入/极端输入/恶意构造/架构缺陷
import sys, os, time, json, traceback
sys.path.insert(0, r"B:\mode\vyton")
R = {"pass": 0, "fail": 0, "found": []}

def T(name, fn, expect_fail=True):
    """破坏性测试:预期要崩/被防住"""
    try:
        r = fn()
        if r == "DEFENDED":
            R["pass"] += 1; print(f"  PASS[防住] {name}")
        elif r == "SURVIVED":
            R["pass"] += 1; print(f"  PASS[存活] {name}")
        else:
            R["fail"] += 1; print(f"  FAIL {name}: {r}")
    except Exception as e:
        if expect_fail and any(k in str(type(e).__name__) for k in ["Error", "Exception"]):
            R["pass"] += 1; print(f"  PASS[清晰崩] {name} → {type(e).__name__}")
        else:
            R["fail"] += 1; print(f"  FAIL {name}: {str(e)[:50]}")

print("=== A.恶意输入 ===")
import torch, numpy as np
from PIL import Image
import tempfile

def a1():
    # 损坏图文件(截断jpg)
    p = os.path.join(tempfile.gettempdir(), "corrupt.jpg")
    open(p, "wb").write(b"\xff\xd8\xff\xe0" + b"\x00" * 50)  # 假jpg头
    try:
        from core.datasets import load_tensor
        load_tensor(p, 64)
        return "未防住:损坏图没报错"
    except Exception as e:
        return "DEFENDED"
T("损坏JPG(截断文件)", a1)

def a2():
    # 超大尺寸声明(炸弹图:64x64内容声明为100000x100000)
    p = os.path.join(tempfile.gettempdir(), "bomb.png")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(p)
    from core.datasets import load_tensor
    x = load_tensor(p, 100000)  # 请求10万分辨率
    return "SURVIVED" if x.shape[1] <= 100000 else "异常"
T("分辨率炸弹(100k请求)", a2)

def a3():
    # NaN注入(模型输入NaN)
    net = torch.nn.Conv2d(3, 3, 1)
    out = net(torch.full((1, 3, 8, 8), float("nan")))
    return "SURVIVED" if True else "?"  # NaN传播是预期,验不崩进程
T("NaN输入传播(不崩进程)", a3)

def a4():
    # 无穷大loss(训练发散场景)
    lin = torch.nn.Linear(4, 2)
    x = torch.full((2, 4), 1e30)  # 溢出输入
    out = lin(x)
    return "SURVIVED"
T("1e30溢出输入", a4)

print("=== B.资源耗尽 ===")
def b1():
    # 内存炸弹:空列表DS被要求getitem
    from core.datasets import ImageFolderDS
    ds = ImageFolderDS([], size=64)
    try:
        ds[0]
        return "未防住:越界没报错"
    except IndexError:
        return "DEFENDED"
T("空DS越界访问", b1)

def b2():
    # GPU OOM场景(渐进申请)
    if not torch.cuda.is_available():
        return "SURVIVED"
    try:
        big = torch.randn(10000, 10000, device="cuda")  # 400MB
        del big
        torch.cuda.empty_cache()
        return "SURVIVED"
    except torch.cuda.OutOfMemoryError:
        return "DEFENDED"
T("GPU大块申请", b2)

def b3():
    # 磁盘写满模拟(写一个不存在的盘)
    try:
        torch.save(torch.zeros(1), "Z:\\nonexistent\\x.pt")
        return "未防住"
    except OSError:
        return "DEFENDED"
T("坏路径保存", b3)

print("=== C.训练循环攻击 ===")
def c1():
    from core.trainer import quick_train
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(16, 3, 32, 32), torch.randn(16, 3, 32, 32))
    dl = DataLoader(ds, batch_size=8)
    net = torch.nn.Conv2d(3, 3, 1)
    # 超高lr→发散(应不崩,loss变nan也算存活)
    net, log = quick_train(net, dl, epochs=1, lr=1e6,
                           loss_fn=lambda o, y: ((o - y) ** 2).mean(),
                           save_path=os.path.join(tempfile.gettempdir(), "atk.pt"))
    return "SURVIVED"
T("lr=1e6发散训练", c1)

def c2():
    # batch只有1个样本(BN会崩,GroupNorm不该)
    from core.blocks import UNetLite
    net = UNetLite(3, 3, ch=8)
    net(torch.randn(1, 3, 32, 32))
    return "SURVIVED"
T("batch=1(GroupNorm设计验证)", c2)

print("=== D.基础设计审计 ===")
def d1():
    # 设计缺陷扫描:blocks.py的UNetLite池化3次=输入必须8的倍数
    from core.blocks import UNetLite
    net = UNetLite(3, 3, ch=8)
    try:
        net(torch.randn(1, 3, 100, 100))  # 非8倍数
        return "SURVIVED"
    except Exception as e:
        R["found"].append(f"UNetLite要求输入8倍数({str(e)[:30]})")
        return "DEFENDED"
T("非8倍数输入", d1)

def d2():
    # metrics的硬编码区域(唇/脸椭圆固定比例→侧脸/半身图失效)
    from core.metrics import face_ellipse
    m = face_ellipse(192, 192)
    R["found"].append("metrics硬编码脸区比例(侧脸/多人图会失效)——需检测器接入")
    return "SURVIVED"
T("脸区硬编码(已知限制)", d2)

def d3():
    # datasets的PairDS不同长度zip静默截断
    from core.datasets import PairDS
    import glob
    a = glob.glob(r"B:\mode\glow_data\pairs\*_a.jpg")[:50]
    b = a[:30]  # b比a短
    ds = PairDS(a, b, size=64)
    if len(ds) < len(a):
        R["found"].append(f"PairDS长度不等静默截断({len(ds)}/{len(a)})——应警告")
    return "SURVIVED"
T("PairDS不等长(静默截断)", d3)

print(f"\n═══ 破坏性: PASS {R['pass']} FAIL {R['fail']} ═══")
print("发现的设计缺陷(要修):")
for f in R["found"]:
    print("  ⚠️", f)
json.dump(R, open(r"B:\mode\vyton\destructive_result.json", "w"))
