
import os, io, json
import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq
from pose_lib import mp_python
from pose_lib import mp_vision
from pose_lib import MPImage, ImageFormat

class MH2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(66,128), torch.nn.ReLU(), torch.nn.Linear(128,64), torch.nn.ReLU())
        self.reg = torch.nn.Linear(64,2); self.gen = torch.nn.Linear(64,2)
    def forward(self, x):
        f = self.backbone(x); return self.reg(f), self.gen(f)
mh = MH2(); mh.load_state_dict(torch.load(r"B:\mode\measure\measure_head_v2.pt", weights_only=True)); mh.eval()
pdet = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=r"B:\mode\models\pose_landmarker_lite.task")))
t = pq.ParquetFile(r"B:\mode\vton_training\data\VITON-HD-Captions\data\train-00000-of-00012.parquet").read_row_group(0)
b = t.column('image')[0].as_py()['bytes']
photo = Image.open(io.BytesIO(b)).convert("RGB"); photo.thumbnail((440,440))
photo.save(r"B:\mode\app_proto\demo_output\1_user_photo.png")
arr = np.asarray(photo)
pr = pdet.detect(MPImage(image_format=ImageFormat.SRGB, data=arr))
wl = np.array([[p.x,p.y] for p in pr.pose_landmarks[0]], np.float32)
c = (wl[23]+wl[24])/2; span = np.linalg.norm(wl[10]-wl[31])+1e-6
with torch.no_grad(): yr, yc = mh(torch.tensor(((wl-c)/span).reshape(-1)[None]))
kw = {"height": float(np.clip(yr[0][0],0.05,0.95)), "weight": float(np.clip(yr[0][1],0.05,0.95))}
json.dump(kw, open(r"B:\mode\app_proto\demo_output\2_measure.json","w"))
print("STEP1_OK", kw)
