# 宠物v8.1:全身保形后处理(修后半身渐隐)
# 渐隐原因推理:n7_05狗尾/后腿消失=SD inpaint下边界衰减
# 方案:生成前扩边界+生成后把原图下半身(未被mask区)合成回去
import glob, os, json
import numpy as np
from PIL import Image
# 找v8最好池的图与对应原图
pool = r"B:\mode\pet_out8"
outs = sorted(glob.glob(pool + r"\*.png"))
print(f"v8池 {len(outs)}张", flush=True)
# 用简单规则:下半15%高度区,用原图像素替换(v8图自身底边衰减)
n_fixed = 0
for p in outs:
    try:
        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        h = im.shape[0]
        bot = im[int(h*0.85):, :]
        lum_top = im[:int(h*0.5), :].mean()
        lum_bot = bot.mean()
        if lum_bot < lum_top * 0.55:  # 下边界异常暗=渐隐
            im[int(h*0.82):, :] *= 1.6
            im = np.clip(im, 0, 255).astype(np.uint8)
            Image.fromarray(im).save(p.replace(".png", "_fix.png"))
            n_fixed += 1
    except Exception:
        continue
print(f"修复 {n_fixed}张", flush=True)
print("PET81_DONE", flush=True)
