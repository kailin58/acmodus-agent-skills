
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
from body3d.complete import complete_params
cases = [
    dict(),                                     # 全空!
    dict(sex="f"),                              # 只有性别
    dict(height=175),                           # 只有身高
    dict(age=6, sex="f"),                       # 只有年龄性别
    dict(age=30, sex="m", height=178, weight=75, ethnic="北欧"),  # 全的
    dict(country="巴西", sex="f"),              # 国家推民族
]
for c in cases:
    p, inf = complete_params(**c)
    print(f"输入{list(c.keys()) or '空'} → {p['sex']},{p['age']}岁,{p['height']:.0f}cm,{p['weight']}kg,{p['ethnic']},胸{p['bust']}腰{p['waist']}臀{p['hips']} | 推算:{len(inf)}项")
