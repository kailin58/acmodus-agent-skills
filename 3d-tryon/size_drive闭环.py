
# 尺寸驱动体态闭环:照片→测量头v2→体型参数→3D渲染
import os, io
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
from pose-lib-mp.tasks import python as mp_python
from pose-lib-mp.tasks.python import vision as mp_vision
from pose-lib-mp import Image as MPImage, ImageFormat
import body_engine as anny

# 模型
class MeasureHeadV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(66,128), torch.nn.ReLU(), torch.nn.Linear(128,64), torch.nn.ReLU())
        self.reg = torch.nn.Linear(64,2)
        self.gen = torch.nn.Linear(64,2)
    def forward(self, x):
        f = self.backbone(x)
        return self.reg(f), self.gen(f)
mh = MeasureHeadV2()
mh.load_state_dict(torch.load(r"B:\mode\measure\measure_head_v2.pt", weights_only=True))
mh.eval()

det = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\pose_landmarker_lite.task")))
model = body_engine.Engine(local_changes="default")

t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
for i in [0, 3]:
    b = t.column('image')[i].as_py()['bytes']
    img = Image.open(io.BytesIO(b)).convert("RGB"); img.thumbnail((440,440))
    arr = np.asarray(img)
    r = det.detect(MPImage(image_format=ImageFormat.SRGB, data=arr))
    if not r.pose_landmarks: continue
    wl = np.array([[p.x,p.y] for p in r.pose_landmarks[0]], dtype=np.float32)
    c = (wl[23]+wl[24])/2
    span = np.linalg.norm(wl[10]-wl[31]) + 1e-6
    x = torch.tensor(((wl-c)/span).reshape(-1)[None])
    with torch.no_grad():
        yr, yc = mh(x)
    h, w = yr[0].tolist()
    g = yc.argmax(1).item()
    gender = "female" if g==1 else "male"
    kw = {"height": float(np.clip(h,0.05,0.95)), "weight": float(np.clip(w,0.05,0.95)),
          "gender": 0.1 if g==1 else 0.9}
    res = model(None, kw)
    V = np.asarray(res["vertices"])[0]
    W2, H2 = 300, 620
    rim = Image.new("RGB", (W2,H2), (24,24,28))
    d = ImageDraw.Draw(rim)
    x0,x1 = V[:,0].min(), V[:,0].max(); z0,z1 = V[:,2].min(), V[:,2].max()
    px = ((V[:,0]-x0)/(x1-x0+1e-9)*(W2-60)+30).astype(int)
    py = ((z1-V[:,2])/(z1-z0+1e-9)*(H2-110)+70).astype(int)
    for k in np.argsort(V[:,1]):
        d.ellipse((px[k]-1,py[k]-1,px[k]+1,py[k]+1), fill=(150,190,255))
    try: font=ImageFont.truetype("arial.ttf",15)
    except: font=ImageFont.load_default()
    d.text((12,20), f"{gender} h={kw['height']:.2f} w={kw['weight']:.2f}", fill=(240,240,240), font=font)
    combo = Image.new("RGB", (img.width+300, max(img.height,620)), (14,14,18))
    combo.paste(img,(0,0)); combo.paste(rim,(img.width,0))
    combo.save(rf"B:\mode\vton_training\garment_poc\size_drive_{i}.png")
    print(f"IMG{i}: {gender} h={kw['height']:.3f} w={kw['weight']:.3f}", flush=True)
print("SIZE_DRIVE_DONE", flush=True)
