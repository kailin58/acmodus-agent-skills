
import numpy as np
from PIL import Image
import glob
for f in sorted(glob.glob(r"B:\mode\models\cut\test_out\*.png"))[:3]:
    a = np.asarray(Image.open(f))[:, :, 3].astype(np.float32) / 255
    soft = ((a > 0.1) & (a < 0.9)).mean()  # 软边像素占比
    import os
    print(f"{os.path.basename(f)[:25]}: 软边{soft*100:.1f}% 纯前景{(a>0.9).mean()*100:.0f}% 纯背景{(a<0.1).mean()*100:.0f}%")
