
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
from body3d.photo_to_self import photo_to_self
import numpy as np, glob
# 3张图(模拟三视图不同人图)
imgs = sorted(glob.glob(r"B:\\mode\\pinterest\\images\\*.jpg"))[:3]
mesh, rep = photo_to_self(imgs, height=178, weight=72)   # 用户给真实值=校准
pm = rep["photo_measure"]
print(f"三照融合: 测高{pm['pred_height']}m 测重{pm['pred_weight']}kg 性别{pm['pred_sex']} 有效照{pm['n_photos']}/3")
print(f"校准后: {rep['params']['height']}cm {rep['params']['weight']}kg(用户值优先✓)")
print(f"网格: 高{(mesh[:,2].max()-mesh[:,2].min())*100:.0f}cm 精度{rep['precision']}")
