# RSI Loop v2:任务前查/失败自动写/经验继承(2026-10按Anthropic方法论强化)
import json, os, time

RSI_DIR = r"B:\mode\rsi"
LEDGER = os.path.join(RSI_DIR, "train_ledger.json")
RULES = os.path.join(RSI_DIR, "rules.json")
FAILLOG = os.path.join(RSI_DIR, "fail_counter.json")

def _load(p, default):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default

def _save(p, d):
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def pre_check(task):
    """规则1:任务前强制查询——返回该任务全部GREEN/RED经验+最佳版本"""
    led = _load(LEDGER, {})
    rules = _load(RULES, {})
    hist = led.get(task, [])
    r = rules.get(task, {})
    greens = r.get("greens", [])
    reds = r.get("reds", [])
    return {
        "history_n": len(hist),
        "best": r.get("best_version"),
        "best_metric": r.get("best_metric"),
        "greens": greens,
        "reds": reds,
        "last3": hist[-3:] if hist else [],
    }

def rsi_log(task, version, metric, data="", note=""):
    """训练/实验完成自动记账"""
    led = _load(LEDGER, {})
    led.setdefault(task, []).append({
        "v": version, "metric": metric, "data": data, "note": note,
        "t": time.strftime("%m-%d %H:%M"),
    })
    # 自动提取GREEN/RED到rules
    if "GREEN:" in note or "RED:" in note:
        rules = _load(RULES, {})
        r = rules.setdefault(task, {"greens": [], "reds": []})
        for kw, bucket in [("GREEN:", "greens"), ("RED:", "reds")]:
            if kw in note:
                item = note.split(kw)[1].split(";")[0].strip()[:60]
                if item and item not in r[bucket]:
                    r[bucket].append(item)
                    # GREEN且metric更优→更新best
                    if kw == "GREEN:":
                        try:
                            m = float(str(metric).replace("联合", "").replace("%", ""))
                            cur = float(str(r.get("best_metric", 0)).replace("联合", "").replace("%", "") or 0)
                            if m > cur:
                                r["best_version"] = version
                                r["best_metric"] = metric
                        except Exception:
                            pass
        _save(RULES, rules)
    _save(LEDGER, led)
    return True

def fail_track(task, fail_sig):
    """规则2:失败自动计数——同签名3次=自动升级为技能条目并提示"""
    fc = _load(FAILLOG, {})
    key = f"{task}|{fail_sig[:50]}"
    fc[key] = fc.get(key, 0) + 1
    _save(FAILLOG, fc)
    if fc[key] >= 3:
        rsi_log(task, "fail_pattern", 0, note=f"RED:3次同失败[{fail_sig[:40]}]→必须换路线或写技能")
        return f"⚠️ 3次失败触发: {fail_sig[:60]} — 已记RED,强制换思路"
    return f"失败{fc[key]}/3"

def inherit(*skill_names):
    """规则3:技能继承——汇总多个技能的坑到一段提示"""
    out = []
    for sn in skill_names:
        for kw in ["坑", "RED", "禁止", "勿", "必须"]:
            rules = _load(RULES, {})
            for task, r in rules.items():
                for g in r.get("reds", []):
                    if sn.lower() in str(r).lower():
                        out.append(f"[{task}] {g}")
                        break
    return "\n".join(dict.fromkeys(out))[:500]
