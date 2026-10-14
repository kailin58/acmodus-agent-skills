import numpy as np
from PIL import Image
import os, glob
ref = np.asarray(Image.open(r"B:/mode/tools/d2_test.png").convert("RGB"), np.float32)
def psnr(a, b):
    mse = ((a-b)**2).mean()
    return 99.0 if mse == 0 else 10*np.log10(255*255/mse)
# 解码文件可能是多mip拼接,取第一个/裁剪到同尺寸
for tag in ["etc1s", "uastc"]:
    fs = glob.glob(rf"B:/mode/tools/d2_{tag}_dec*/output.png")
    if not fs: print(tag, "无解码图"); continue
    im = Image.open(fs[0]).convert("RGB")
    if im.size != (ref.shape[1], ref.shape[0]):
        im = im.resize((ref.shape[1], ref.shape[0]))
    a = np.asarray(im, np.float32)
    print(f"{tag}: PSNR={psnr(ref, a):.2f}dB")
src = os.path.getsize(r"B:/mode/tools/d2_test.png")
e = os.path.getsize(r"B:/mode/tools/d2_etc1s.ktx2")
u = os.path.getsize(r"B:/mode/tools/d2_uastc.ktx2")
print(f"体积: 原图{src//1024}KB ETC1S{e//1024}KB({src/e:.1f}x) UASTC{u//1024}KB({src/u:.1f}x)")
# 混合方案模拟(70%ETC1S+30%UASTC按内容分)
mix = int(e*0.7 + u*0.3)
print(f"混合方案估算: {mix//1024}KB({src/mix:.1f}x)")
# 对比图(裁特写区)
crop = (600, 600, 1000, 1000)
ims = []
for f in [r"B:/mode/tools/d2_test.png"] + sorted(glob.glob(r"B:/mode/tools/d2_etc1s_dec*/output.png")) + sorted(glob.glob(r"B:/mode/tools/d2_uastc_dec*/output.png")):
    ims.append(np.asarray(Image.open(f).convert("RGB").crop(crop).resize((400,400))))
grid = np.concatenate(ims, axis=1)
Image.fromarray(grid).save(r"B:/mode/tools/d2_compare.png")
print("COMPARE_OK")
