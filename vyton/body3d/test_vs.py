
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
import numpy as np
from PIL import Image, ImageDraw
from body3d.virtual_self import gen_virtual_self

# 形式②基础数据
m1, r1 = gen_virtual_self(age=28, sex="f", height=163, weight=52, ethnic="南方东亚")
print(f"②基础: 高{(m1[:,2].max()-m1[:,2].min())*100:.0f}cm 精度{r1['precision']} 推算{len(r1['inferred'])}项")
# 形式③明细
m2, r2 = gen_virtual_self(age=35, sex="m", height=180, weight=82, country="德国", bust=100, waist=88, hips=98)
print(f"③明细: 高{(m2[:,2].max()-m2[:,2].min())*100:.0f}cm 精度{r2['precision']} 推算{r2['inferred']}")
# 形式④数据+脸照(合成圆脸)
img = Image.new("RGB", (200, 260), (40, 40, 50))
d = ImageDraw.Draw(img)
d.ellipse([20, 20, 180, 240], fill=(224, 172, 140))
m3, r3 = gen_virtual_self(age=25, sex="f", height=168, weight=55, face_img=np.asarray(img))
print(f"④数据+脸: 脸应用={r3.get('face', {}).get('applied')} 精度{r3['precision']}")
# 形式①模拟:只给身高体重(照片测量后续接)
m4, r4 = gen_virtual_self(height=170, weight=60)
print(f"①极简: {r4['params']['sex']},{r4['params']['age']}岁默认 推算{r4['inferred'][:3]}...")
# 同输入确定性
m5, _ = gen_virtual_self(age=28, sex="f", height=163, weight=52, ethnic="南方东亚")
assert np.array_equal(m1, m5)
print("确定性PASS | 4形式全通")
