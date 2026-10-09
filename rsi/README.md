# RSI Loop系统(递归自改进)

## 架构(2026-10-09 v2)
```
任务开始 → pre_check(查历史经验/最佳版本/坑)
  ↓
执行(训练/采集/实验)
  ↓
质量门5关:G1数据/G2指标/G3证据/G4回归/G5效率
  ↓ (FAIL=打回重做)
分值评估:0-100(指标60%+门禁25%+经验15%)
  ↓
rsi_log自动记账(GREEN/RED自动提取入库)
  ↓
失败3次同签名 → fail_track强制换路线
```

## 使用
```python
from rsi_hook import pre_check, rsi_log, fail_track
hist = pre_check("style")        # 任务前必查
rsi_log("style", "v13", 0.853, note="GREEN:自评环")  # 完成记账
fail_track("style", "OOM")       # 失败计数
from quality_gate import quality_gate, score
ok, gates = quality_gate("style", 0.853, ev)
s = score("style", 0.853, gates)
```

## 方法论来源
Anthropic Agent Skills(渐进披露/自进化/组织记忆)+TMLR 2026自改进闭环
落地实证:Style v13自评环 0.782→0.853(+9.1%)
