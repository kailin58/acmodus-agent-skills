# RSI自动记录钩子:训练脚本import后在结束时调用
import json, os
RSI = r"B:\mode\rsi\train_ledger.json"

def rsi_log(task, version, metric, data="", note=""):
    """训练完成自动记账:rsi_log('style','v13',0.85,'数据X','GREEN:某技巧')"""
    try:
        led = json.load(open(RSI, encoding="utf-8"))
        led.setdefault(task, []).append({"v": version, "metric": metric, "data": data, "note": note})
        json.dump(led, open(RSI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return True
    except Exception:
        return False

def rsi_query(task=None):
    """查询:rsi_query('style')或全部"""
    led = json.load(open(RSI, encoding="utf-8"))
    return led.get(task) if task else led
