"""vyton.body3d.complete — 数据补全引擎(高内聚:只管补全)
规则(用户定):数据越细越接近;数据不全→模拟完善
输入任意子集→输出全参数+confidence标记(哪些是推算)
"""
import json, os

_DIR = os.path.dirname(__file__)
STAGES = json.load(open(os.path.join(_DIR, "anthro_stages.json"), encoding="utf-8"))
ETHNIC = json.load(open(os.path.join(_DIR, "ethnic_adj.json"), encoding="utf-8"))

# BMI参考(年龄,性别):儿童15-17,成人男22-24女21-23,老人23-25
def _ref_bmi(age, sex):
    if age < 18: return 16 + age * 0.15
    if age < 65: return 23.5 if sex == "m" else 21.5
    return 24.0

def complete_params(age=None, sex=None, height=None, weight=None,
                    ethnic=None, country=None, bust=None, waist=None, hips=None):
    """任意子集→全参数。返回(params, inferred) inferred=被推算的字段"""
    inf = []
    if sex is None:
        sex = "m"; inf.append("sex")
    if ethnic is None and country is None:
        ethnic = "东亚"; inf.append("ethnic")
    elif ethnic is None:
        # 国家→民族映射
        m = {"中国": "东亚", "日本": "东亚", "韩国": "东亚", "印度": "南亚", "美国": "北美",
             "德国": "西欧", "法国": "西欧", "英国": "西欧", "巴西": "拉美", "尼日利亚": "撒哈拉以南"}
        ethnic = m.get(country, "东亚"); inf.append("ethnic(由国家推)")
    adj = 0
    for k, v in ETHNIC.items():
        if k in ethnic: adj = v; break
    if age is None:
        age = 30; inf.append("age")
    if height is None:
        # 身高:由体重+BMI反推,否则阶段中位+民族
        if weight:
            h0 = (weight / _ref_bmi(age, sex)) ** 0.5
            # 用阶段范围约束
            st = _stage(age, sex)
            lo, hi = STAGES[st]["range"]
            height = float(np.clip(h0 * 100, lo, hi))
        else:
            height = STAGES[_stage(age, sex)]["h"] + adj
        inf.append("height")
    if weight is None:
        h_m = height / 100
        weight = round(_ref_bmi(age, sex) * h_m * h_m, 1)
        inf.append("weight")
    # 围度:由身高体重推(BMI大→围度大)
    h_m = height / 100
    bmi = weight / (h_m * h_m)
    if bust is None:
        bust = round(h_m * (49 + (bmi - 22) * 0.9) if sex == "f" else h_m * (52 + (bmi - 22) * 0.95), 1)
        inf.append("bust")
    if waist is None:
        waist = round(h_m * (41 + (bmi - 22) * 1.4), 1)
        inf.append("waist")
    if hips is None:
        hips = round(h_m * (50 + (bmi - 22) * 1.0) if sex == "f" else h_m * (51 + (bmi - 22) * 0.9), 1)
        inf.append("hips")
    p = dict(age=age, sex=sex, height=height, weight=weight, ethnic=ethnic,
             country=country, bust=bust, waist=waist, hips=hips,
             bmi=round(bmi, 1), inferred=inf)
    return p, inf

def _stage(age, sex):
    if age < 1: return "baby_0"
    if age < 2: return "baby_1"
    if age < 4: return "child_3"
    if age < 7: return "child_6"
    if age < 10: return "child_9"
    if age < 13: return "teen_12"
    if age < 16: return "teen_15"
    if age < 65: return "adult_m" if sex == "m" else "adult_f"
    return "senior"

import numpy as np
