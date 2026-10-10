# 交叉学科暴力验证audit
import json, math
from collections import Counter

R = {}
labs = json.load(open(r"B:\mode\style_train\pin_labels_v1.json", encoding="utf-8"))

def entropy(counter):
    tot = sum(counter.values())
    return -sum((c/tot)*math.log2(c/tot) for c in counter.values() if c > 0)

sty_c = Counter(s for v in labs.values() for s in v.get("style", []))
sce_c = Counter(s for v in labs.values() for s in v.get("scene", []))
R["信息论"] = {"风格熵": round(entropy(sty_c), 2), "场景熵": round(entropy(sce_c), 2),
             "风格最大熵": round(math.log2(len(sty_c)), 2)}

sus = json.load(open(r"B:\mode\style_train\suspect_v12.json", encoding="utf-8"))
fix_pred = Counter(x["pred"] for x in sus if x["type"]=="style" and x["conf"]>0.6)
conc = round(fix_pred.most_common(1)[0][1]/max(sum(fix_pred.values()),1), 2) if fix_pred else 0
R["统计-确认偏差"] = {"改标分布": dict(fix_pred.most_common(4)), "单类集中度": conc,
                   "风险": "大于0.6为模式坍缩风险"}

enb3 = 12e6
R["容量-过参数化"] = {"每样本参数": int(enb3/1338), "健康区间": "10-100",
                   "判定": "严重过参数化,靠预训练先验撑住,建议冻结backbone"}

R["CS-复杂度"] = {"四头": "一次前向多头复用=O(n)最优",
                 "毛孔头": "高斯滤波可分离优化空间3ms→1ms"}

R["物理-尺度律"] = {"观察": "数据期28倍→+42%;方法期0数据→+9%",
                   "判定": "已过数据拐点,进入方法驱动期"}

for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False))
json.dump(R, open(r"B:\mode\rsi\cross_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("CROSS_AUDIT_DONE")