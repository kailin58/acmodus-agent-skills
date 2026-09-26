# Vyton多帧融合测量(App侧集成件)v1
# 用法:拍3-5张→各帧跑Measure→中值聚合→输出稳定6维
import json
import numpy as np

def fuse_measurements(frame_preds):
    """frame_preds: list of dict(height/weight/shoulder/hip/torso/leg,gender_logit)
    返回:中值6维+稳定性指标(各维IQR=不确定度)"""
    keys = ["height","weight","shoulder","hip","torso","leg"]
    out = {}
    for k in keys:
        vals = np.array([f[k] for f in frame_preds], dtype=np.float32)
        out[k] = float(np.median(vals))
        out[k+"_unc"] = float(np.percentile(vals,75)-np.percentile(vals,25))  # IQR=置信
    g = np.array([f.get("gender_logit",0.5) for f in frame_preds])
    out["gender"] = float(np.median(g))
    out["gender_conf"] = float(min(g.max()-g.min(), 1.0))  # 越小越稳
    out["n_frames"] = len(frame_preds)
    # 质量门:肩宽IQR>0.03(≈5cm抖动)建议重拍
    out["retake_advice"] = bool(out["shoulder_unc"] > 0.03)
    return out

if __name__ == "__main__":
    # 自测:模拟5帧(v3.1典型误差±0.012归一≈±2cm)
    rng = np.random.default_rng(42)
    base = dict(height=0.68, weight=0.62, shoulder=0.44, hip=0.51, torso=0.55, leg=0.58, gender_logit=0.8)
    frames = [{k: v + rng.normal(0, 0.012) for k, v in base.items()} for _ in range(5)]
    r = fuse_measurements(frames)
    print(json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in r.items()}, ensure_ascii=False, indent=1))
    # 理论:5帧中值误差≈单帧/√n×0.8≈±0.004→肩宽±0.7cm
