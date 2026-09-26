# INT8抠图质量验收:同图FP32 vs INT8,IoU对比
import numpy as np
import onnxruntime as ort
from PIL import Image
import glob, io

src = sorted(glob.glob(r"B:\mode\vton_training\whitebg_50\*.jpg"))[0]
img = Image.open(src).convert("RGB")
# rembg预处理:resize 320 + normalize
im = img.resize((320, 320))
x = np.asarray(im, dtype=np.float32)/255.0
x = ((x - np.array([0.485,0.456,0.406],np.float32)) / np.array([0.229,0.224,0.225],np.float32)).transpose(2,0,1)[None]

def mask_of(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0][0]
    return (out > 0.5)

m_fp = mask_of(r"B:\mode\models\cut\u2net_fp32.onnx")
m_int = mask_of(r"B:\mode\models\cut\vyton_cut_int8.onnx")
inter = (m_fp & m_int).sum(); union = (m_fp | m_int).sum()
iou = inter/max(union,1)
print(f"IoU INT8-vs-FP32: {iou:.4f} ({'PASS≥0.98' if iou>=0.98 else 'FAIL<0.98'})")
# 可视化对比
W = 660; H = 360
canvas = Image.new("RGB", (W, H), (10,10,14))
a = Image.fromarray((m_fp*255).astype(np.uint8)).resize((320,320))
b = Image.fromarray((m_int*255).astype(np.uint8)).resize((320,320))
canvas.paste(Image.merge("RGB",(a,a,a)), (8, 26))
canvas.paste(Image.merge("RGB",(b,b,b)), (332, 26))
from PIL import ImageDraw
d = ImageDraw.Draw(canvas)
d.text((10,6), f"FP32 mask  |  INT8 mask   IoU={iou:.4f}", fill=(255,255,80))
canvas.save(r"B:\mode\models\cut\quant_compare.png")
print("CMP_SAVED")
