
# 抠图全面测试v2:多类图+边缘质量+失败检测
import glob, os
import numpy as np
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession(r"B:\mode\models\cut\vyton_cut_int8.onnx", providers=["CPUExecutionProvider"])
MEAN = np.array([0.485,0.456,0.406], np.float32)
STD = np.array([0.229,0.224,0.225], np.float32)

def cut(img):
    im = img.convert("RGB").resize((320, 320))
    x = (np.asarray(im, np.float32)/255.0 - MEAN) / STD
    x = x.transpose(2,0,1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0][0]
    m = (out > 0.5).astype(np.uint8) * 255
    return Image.fromarray(m).resize(img.size, Image.LANCZOS)

pools = {
    "人像": glob.glob(r"B:\mode\glow_data\mt\extracted\makeup\*.jpg")[:3],
    "宠物": glob.glob(r"B:\mode\pet\lora_data_v3\dog_*_src.png")[:2],
    "服装白底": glob.glob(r"B:\mode\vton_training\whitebg_50\*.jpg")[:2],
    "复杂背景": glob.glob(r"B:\mode\pinterest\outfit_imgs\*.jpg")[:3],
}
os.makedirs(r"B:\mode\models\cut\test_out", exist_ok=True)
report = []
for cat, files in pools.items():
    for f in files:
        img = Image.open(f)
        m = cut(img)
        a = np.asarray(m, np.float32) / 255
        H, W = a.shape
        # 指标1:主体覆盖率(合理域5%-90%)
        cov = a.mean()
        # 指标2:边缘锐度(mask梯度和,高=边缘清晰)
        gy, gx = np.gradient(a)
        edge = float(np.sqrt(gx**2 + gy**2).mean() * 100)
        # 指标3:背景残留(四角10%区域应接近0)
        corners = np.concatenate([a[:H//10, :W//10].ravel(), a[:H//10, -W//10:].ravel(),
                                  a[-H//10:, :W//10].ravel(), a[-H//10:, -W//10:].ravel()])
        bg_res = float(corners.mean())
        # 保存结果
        rgba = np.dstack([np.asarray(img.convert("RGB")), (a * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(rf"B:\mode\models\cut\test_out\{cat}_{os.path.basename(f)[:12]}.png")
        ok = (0.03 < cov < 0.92) and bg_res < 0.15
        report.append((cat, os.path.basename(f)[:15], round(cov,3), round(edge,1), round(bg_res,3), "PASS" if ok else "WARN"))
        print(f"{cat:6s} {os.path.basename(f)[:15]:15s} 覆盖{cov:.2f} 边缘{edge:.0f} 背景残留{bg_res:.2f} {'PASS' if ok else 'WARN'}")
n_pass = sum(1 for r in report if r[5] == "PASS")
print(f"\n=== {n_pass}/{len(report)} PASS ===")
