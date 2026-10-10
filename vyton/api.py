# vyton.api:高内聚门面——外部只import这一个
"""
from vyton.api import collect, train, evaluate, rsi
collect("youtube", kws=[...])       # 采集
train("style", data=..., epochs=..) # 训练
evaluate("style", model, metric)    # 质量门+分值
rsi("style").pre_check()            # 经验查询
"""
import sys, os
sys.path.insert(0, r"B:\mode\rsi")
from rsi_hook import pre_check, rsi_log, fail_track
from quality_gate import quality_gate, score

def collect(source, **kw):
    """统一采集入口:source∈{pinterest,instagram,youtube,bilibili,brandsites}"""
    from .data import dispatch
    return dispatch(source, **kw)

def evaluate(task, metric, evidence=None):
    ok, gates = quality_gate(task, metric, evidence or {})
    s = score(task, metric, gates)
    return {"pass": ok, "gates": gates, "score": s}
