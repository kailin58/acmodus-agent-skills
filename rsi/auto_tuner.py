# -*- coding: utf-8 -*-
# RSI规则引擎:账本→规则提取→参数建议→失败拦截
import json, os, sys

RDIR = r"B:\mode\rsi"
LEDGER = os.path.join(RDIR, "train_ledger.json")
RULES = os.path.join(RDIR, "rules.json")

def load_rules():
    if os.path.exists(RULES):
        return json.load(open(RULES, encoding="utf-8"))
    return {}

def extract_rules():
    """从账本提取GREEN/RED规则"""
    led = json.load(open(LEDGER, encoding="utf-8"))
    rules = load_rules()
    for line, exps in led.items():
        greens = [e["note"].split(":",1)[1] for e in exps if e.get("note","").startswith("GREEN")]
        reds = [e["note"].split(":",1)[1] for e in exps if e.get("note","").startswith("RED")]
        best = max(exps, key=lambda e: e.get("acc", e.get("score", 0)))
        rules[line] = {
            "best_version": best["v"],
            "best_metric": best.get("acc", best.get("score")),
            "greens": greens, "reds": reds,
        }
    json.dump(rules, open(RULES, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rules

def advise(line):
    """给指定产品线的下轮参数建议"""
    rules = load_rules()
    r = rules.get(line)
    if not r: return f"{line}: 无历史,从预训练微调起步"
    return {
        "line": line,
        "inherit_from": r["best_version"],
        "must_do": r["greens"],
        "must_avoid": r["reds"],
    }

FAIL_PATTERNS = [
    {"name": "GroupNorm不整除", "pat": "ch=52", "fix": "ch必须被8整除(48/64)"},
    {"name": "负向语义词", "pat": "not scarf|not a", "fix": "负向只加具体物(drool/blurry),禁not X语义"},
    {"name": "权重过高", "pat": ":1.4|:1.5", "fix": "prompt权重≤1.3"},
    {"name": "从零训练分类", "pat": "weights=None", "fix": "必须预训练微调"},
    {"name": "caption在循环内def", "pat": "for d in v4:\n    def ", "fix": "函数定义提循环外"},
]

def lint_script(path):
    """④失败模式拦截:扫脚本命中已知坑"""
    code = open(path, encoding="utf-8").read()
    hits = []
    for fp in FAIL_PATTERNS:
        if fp["pat"].replace("\\n", chr(10)) in code:
            hits.append(fp)
    return hits

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rules"
    if cmd == "rules":
        r = extract_rules()
        for k, v in r.items():
            print(f"[{k}] best={v['best_version']}({v['best_metric']}) GREEN={len(v['greens'])} RED={len(v['reds'])}")
    elif cmd == "advise":
        print(json.dumps(advise(sys.argv[2]), ensure_ascii=False, indent=1))
    elif cmd == "lint":
        hits = lint_script(sys.argv[2])
        if hits:
            for h in hits: print("⛔", h["name"], "→", h["fix"])
            sys.exit(1)
        print("✓ 无已知坑")
