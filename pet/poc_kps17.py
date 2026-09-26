
# 宠物17点POC(onnxruntime CPU)
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

sess = ort.InferenceSession(r"B:\mode\models\pet\rtmpose_animal_ap10k.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("输入:", inp.name, inp.shape)

img = Image.open(r"B:\mode\pet\sample.png").convert("RGB")
# 中心裁方形→256
w, h = img.size
s = min(w, h)
img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((256, 256))
x = np.asarray(img, dtype=np.float32)
# ImageNet norm(模型卡说明)
mean = np.array([123.675, 116.28, 103.53], np.float32)
std = np.array([58.395, 57.12, 57.375], np.float32)
x = ((x - mean) / std).transpose(2,0,1)[None]
outs = sess.run(None, {inp.name: x})
sx, sy = outs[0][0], outs[1][0]   # [17,512]
kx = sx.argmax(-1) / 2.0
ky = sy.argmax(-1) / 2.0
conf = sx.max(-1)
print("17点(256系):")
NAMES = ["左眼","右眼","鼻","颈","尾根","左肩","左肘","左前爪","右肩","右肘","右前爪","左髋","左膝","左后爪","右髋","右膝","右后爪"]
d = ImageDraw.Draw(img)
for i, (a, b, c) in enumerate(zip(kx, ky, conf)):
    print(f"{i:2d} {NAMES[i]}: ({a:.0f},{b:.0f}) conf={c:.2f}")
    if c > 2:
        d.ellipse((a-3,b-3,a+3,b+3), fill=(255,0,0))
img.save(r"B:\mode\pet\poc_kps.png")
print("POC_KPS_DONE")
