
# 教师质量对比:u2net vs BiRefNet伪标签
import glob, os
import numpy as np
import onnxruntime as ort
from PIL import Image

pseudo = sorted(glob.glob(r"B:\mode\models\cut\pseudo_labels\*.png"))[:4]
sess = ort.InferenceSession(r"B:\mode\models\cut\vyton_cut_int8.onnx", providers=["CPUExecutionProvider"])
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
for f in pseudo:
    name = os.path.basename(f).replace(".png", "")
    src = None
    for base in [r"B:\mode\pinterest\outfit_imgs", r"B:\mode\yt_outfit", r"B:\mode\pet\lora_data_v3"]:
        for ext in [".jpg", ".png"]:
            if os.path.exists(os.path.join(base, name)):
                src = os.path.join(base, name); break
        if src: break
    if not src:
        continue
    img = Image.open(src).convert("RGB")
    # u2net
    im = img.resize((320, 320))
    x = (np.asarray(im, np.float32)/255 - MEAN)/STD
    x = x.transpose(2,0,1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0][0]
    u2m = Image.fromarray(((out>0.5).astype(np.uint8)*255)).resize(img.size, Image.LANCZOS)
    # BiRefNet
    bm = Image.open(f).convert("L")
    if bm.size != img.size:
        bm = bm.resize(img.size)
    # 三联对比
    W, H = img.size
    canvas = Image.new("RGB", (W, H*3), (20,20,20))
    canvas.paste(img, (0, 0))
    canvas.paste(u2m.convert("RGB"), (0, H))
    canvas.paste(bm.convert("RGB"), (0, H*2))
    canvas.resize((W//2, H*3//2)).save(rf"B:\mode\models\cut\cmp_{name[:12]}.jpg", quality=88)
    # 数字:软边对比
    a = np.asarray(bm, np.float32)/255
    soft_b = ((a>0.1)&(a<0.9)).mean()
    a2 = np.asarray(u2m, np.float32)/255
    soft_u = ((a2>0.1)&(a2<0.9)).mean()
    print(f"{name[:16]}: 软边 u2net {soft_u*100:.1f}% → BiRefNet {soft_b*100:.1f}%")
print("CMP_DONE")
