import torch, torch.nn as nn
from torchvision import models as tvm
class BackboneV2(nn.Module):
    def __init__(s):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=None)
        s.feat = m.features
    def forward(s, x): return s.feat(x)
class HeadV2(nn.Module):
    def __init__(s, inch=576):
        super().__init__()
        s.up = nn.Sequential(nn.Conv2d(inch,64,1), nn.ReLU(True), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False), nn.Conv2d(64,32,3,padding=1), nn.ReLU(True), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False), nn.Conv2d(32,16,3,padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(16,1,3,padding=1))
    def forward(s, f): return s.up(f)
bb, hd = BackboneV2(), HeadV2()
bb.load_state_dict(torch.load(r"B:/mode/models/cut/vyton_cut_v21_backbone.pt", map_location="cpu"))
hd.load_state_dict(torch.load(r"B:/mode/models/cut/vyton_cut_v21_head.pt", map_location="cpu"))
bb.eval(); hd.eval()
x = torch.randn(1, 3, 320, 320)
torch.onnx.export(bb, x, r"B:/mode/models/cut/pack/capability_cutout_v2/backbone320.onnx", opset_version=13, input_names=["image"], output_names=["feat"], dynamic_axes={"image": {0: "b"}})
with torch.no_grad():
    f = bb(x)
torch.onnx.export(hd, f, r"B:/mode/models/cut/pack/capability_cutout_v2/head320.onnx", opset_version=13, input_names=["feat"], output_names=["mask"], dynamic_axes={"feat": {0: "b"}})
print("ONNX320_OK")
