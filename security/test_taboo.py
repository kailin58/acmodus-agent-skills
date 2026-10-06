import torch, torch.nn as nn, numpy as np
from PIL import Image
class TabooNet(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        s.net = nn.Sequential(blk(3,32), blk(32,64), blk(64,128), blk(128,192), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(192, 12)
    def forward(s, x): return s.fc(s.net(x).flatten(1))
CLASSES = ["funeral","religious_sacred","political","flag_misuse","exposure","gesture_offense","animal_omen","number_taboo","food_violation","material_risk","gender_mismatch","SAFE"]
m = TabooNet().cuda()
m.load_state_dict(torch.load(r"B:\mode\models\vyton_taboo_net.pt", map_location="cuda", weights_only=True))
m.eval()
for p, name in [(r"B:\mode\ethnic_data\Japan\x_00.jpg", "日本民族服(应SAFE)"),
                (r"B:\mode\vton_training\garment_poc\virtual_self_v9.png", "虚拟自己(应SAFE)")]:
    try:
        img = Image.open(p).convert("RGB").resize((128,128))
        x = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()
        with torch.no_grad():
            out = torch.softmax(m(x),1)[0]
        top = out.argmax().item()
        print(name, "->", CLASSES[top], round(float(out[top]),3))
    except Exception as e:
        print(name, "跳过:", str(e)[:40])
