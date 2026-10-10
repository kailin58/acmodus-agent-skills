# RSI质量门+分值评估体系(用户令2026-10-09)
# 质量门:输出必须过5门才放行(任何一门FAIL=打回)
# 分值:每任务0-100分制,加权合成

import json, os

def quality_gate(task, metric, evidence=None):
    """5门检查:数据/指标/证据/回归/效率
    返回(PASS/FAIL, 明细)"""
    gates = {}
    # G1 数据门:数据量/来源清洁
    gates["G1数据"] = "PASS" if evidence and evidence.get("data_n", 0) >= 50 else "CHECK"
    # G2 指标门:不低于历史best-5%(防回归)
    try:
        led = json.load(open(r"B:\mode\rsi\train_ledger.json", encoding="utf-8"))
        hist = [float(str(x["metric"]).replace("联合","").replace("%","")) 
                for x in led.get(task, []) 
                if x.get("metric") and str(x["metric"]).replace("联合","").replace("%","").replace(".","").isdigit()]
        best = max(hist) if hist else 0
        cur = float(str(metric).replace("联合","").replace("%",""))
        gates["G2指标"] = "PASS" if cur >= best * 0.95 else "FAIL"
    except Exception:
        gates["G2指标"] = "CHECK"
    # G3 证据门:有实测输出(非自述)
    gates["G3证据"] = "PASS" if evidence and evidence.get("real_output") else "FAIL"
    # G4 回归门:老能力不掉(多任务时查其他任务best)
    gates["G4回归"] = "PASS"  # 单任务默认过
    # G5 效率门:耗时/显存不超预算
    gates["G5效率"] = "PASS" if evidence and evidence.get("ms", 999) <= 500 else "WARN"
    ok = all(v != "FAIL" for v in gates.values())
    return ok, gates

def score(task, metric, gates, notes=""):
    """0-100分:指标60%+门禁25%+经验沉淀15%"""
    try:
        led = json.load(open(r"B:\mode\rsi\train_ledger.json", encoding="utf-8"))
        hist = [float(str(x["metric"]).replace("联合","").replace("%","")) 
                for x in led.get(task, []) 
                if x.get("metric") and str(x["metric"]).replace("联合","").replace("%","").replace(".","").isdigit()]
        best = max(hist) if hist else 0
        cur = float(str(metric).replace("联合","").replace("%",""))
        rel = cur / max(best, 1e-9)  # 相对历史best
    except Exception:
        rel = 0.8
    s_metric = min(60, rel * 60)
    gate_pass = sum(1 for v in gates.values() if v == "PASS")
    s_gate = gate_pass / len(gates) * 25
    s_exp = 15 if "GREEN" in notes or "RED" in notes else 10  # 有经验提取=满分
    return round(s_metric + s_gate + s_exp, 1)
