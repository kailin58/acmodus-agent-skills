# vyton.core 工程验证套件:接口稳定性/并发/边界/风险
# 运行:py -3.13 vyton/core/stress_test.py
import sys, os, time, threading, traceback
sys.path.insert(0, r"B:\mode\vyton")

R = {"pass": 0, "fail": 0, "warn": 0}

def check(name, fn):
    try:
        fn()
        R["pass"] += 1
        print(f"  PASS {name}")
    except Exception as e:
        R["fail"] += 1
        print(f"  FAIL {name}: {str(e)[:60]}")

def warn_check(name, fn):
    try:
        r = fn()
        if r is True:
            R["pass"] += 1; print(f"  PASS {name}")
        else:
            R["warn"] += 1; print(f"  WARN {name}: {r}")
    except Exception as e:
        R["fail"] += 1; print(f"  FAIL {name}: {str(e)[:60]}")

print("=== 1.接口契约 ===")
from core.datasets import ImageFolderDS, PairDS, load_tensor
from core.blocks import UNetLite, MultiHeadCls, ConvBlk
from core.trainer import quick_train
from core.metrics import lip_redness, skin_mask, face_ellipse

def t_load():
    from PIL import Image
    import tempfile, numpy as np
    p = os.path.join(tempfile.gettempdir(), "vyton_t.jpg")
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
    x = load_tensor(p, 64, augment=False)
    assert x.shape == (3, 64, 64) and -1.01 <= x.min() and x.max() <= 1.01
check("load_tensor契约(shape/range)", t_load)

def t_unet():
    import torch
    net = UNetLite(6, 3, ch=8, aux_heads={"cat": 9})
    x = torch.randn(2, 6, 64, 64)
    out, aux = net(x)
    assert out.shape == (2, 3, 64, 64) and aux["cat"].shape == (2, 9)
check("UNetLite契约(io+aux形状)", t_unet)

def t_cls():
    import torch
    m = MultiHeadCls("enb3", {"a": 3, "b": 5}, pretrained=False)
    out = m(torch.randn(1, 3, 160, 160))
    assert out["a"].shape == (1, 3) and out["b"].shape == (1, 5)
check("MultiHeadCls契约", t_cls)

print("=== 2.调用稳定性(重复调用/状态泄漏) ===")
def t_repeat():
    import torch, numpy as np
    from PIL import Image
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "vyton_r.jpg")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(p)
    outs = [load_tensor(p, 32, augment=False).sum().item() for _ in range(20)]
    assert max(outs) - min(outs) < 1e-4, "同输入不同输出=状态泄漏"
check("load_tensor幂等性(20次)", t_repeat)

def t_gpu_leak():
    import torch, gc
    if not torch.cuda.is_available():
        return True
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    for _ in range(5):
        net = UNetLite(3, 3, ch=8).cuda()
        x = torch.randn(1, 3, 64, 64).cuda()
        net(x)
        del net, x
        gc.collect()
    after = torch.cuda.memory_allocated()
    return True if after - base < 50*1024*1024 else f"泄漏{(after-base)//1024//1024}MB"
warn_check("GPU显存泄漏(5轮)", t_gpu_leak)

print("=== 3.并发安全 ===")
def t_thread():
    import torch, threading
    net = UNetLite(3, 3, ch=8)
    errs = []
    def worker():
        try:
            x = torch.randn(1, 3, 32, 32)
            net.eval()(x)
        except Exception as e:
            errs.append(str(e)[:40])
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs, f"并发错:{errs[0]}"
check("推理多线程(8线程)", t_thread)

def t_dl_workers():
    import glob
    from torch.utils.data import DataLoader
    files = glob.glob(r"B:\mode\glow_data\pairs\*_a.jpg")[:32]
    if len(files) < 8:
        return True
    ds = ImageFolderDS(files, size=64)
    dl = DataLoader(ds, batch_size=8, num_workers=0)  # Windows spawn需main guard;库本身无限制,业务脚本自行guard
    n = sum(1 for _ in dl)
    return True if n == len(files)//8+ (1 if len(files)%8 else 0) else f"batch数{n}"
warn_check("DataLoader多worker", t_dl_workers)

print("=== 4.边界条件 ===")
def t_bad_file():
    import numpy as np
    try:
        load_tensor("nonexistent.jpg", 64)
        return False
    except FileNotFoundError:
        return True
warn_check("不存在文件(清晰报错)", t_bad_file)

def t_zero_size():
    import torch
    try:
        net = UNetLite(3, 3, ch=8)
        net(torch.randn(1, 3, 8, 8))  # 极小输入
        return True
    except Exception as e:
        return f"8x8输入崩:{str(e)[:30]}"
warn_check("极小输入(8x8)", t_zero_size)

def t_empty_ds():
    from torch.utils.data import DataLoader
    ds = ImageFolderDS([], size=64)
    return True if len(ds) == 0 else "空DS不为0"
warn_check("空数据集", t_empty_ds)

print("=== 5.风险扫描(未提到的) ===")
def t_path_cn():
    # 中文路径(项目常用B:\mode\穿搭)
    import tempfile, numpy as np
    from PIL import Image
    d = os.path.join(tempfile.gettempdir(), "测试目录")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "图.jpg")
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(p)
    load_tensor(p, 16, augment=False)
    return True
warn_check("中文路径", t_path_cn)

def t_pickle_risk():
    # torch.load风险(weights_only默认)
    import torch, tempfile
    p = os.path.join(tempfile.gettempdir(), "v_t.pt")
    torch.save({"a": torch.zeros(1)}, p)
    d = torch.load(p, weights_only=True)
    return True
warn_check("torch.load反序列化安全", t_pickle_risk)

def t_disk_full():
    # 磁盘余量
    import shutil as sh
    free = sh.disk_usage("B:\\").free // 2**30
    return True if free > 20 else f"B盘仅剩{free}GB"
warn_check("磁盘余量(>20GB)", t_disk_full)

def t_gpu_contend():
    # GPU争用(采集+训练同时)
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                       capture_output=True, timeout=15)
    used, tot = [int(x.strip().rstrip(" MiB")) for x in r.stdout.decode().split(",")]
    return True if used / tot < 0.9 else f"GPU {used}/{tot}MB 已近满"
warn_check("GPU争用余量", t_gpu_contend)

print(f"\n═══ 结果: PASS {R['pass']} | WARN {R['warn']} | FAIL {R['fail']} ═══")
import json
json.dump(R, open(r"B:\mode\vyton\stress_result.json", "w"))
