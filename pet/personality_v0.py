# Vyton-Pet性格引擎v0:5维性格向量+行为参数映射+演化
import numpy as np

# 5维:活泼energy/粘人affection/好奇curiosity/胆小timidity/独立independence
DIMS = ["energy", "affection", "curiosity", "timidity", "independence"]

def gen_personality(seed=None):
    """照片哈希/用户选择→性格种子(算法随机,同图可重roll)"""
    rng = np.random.default_rng(seed)
    # Beta分布:大多数性格温和,极端性格少
    v = rng.beta(2.5, 2.5, size=5)
    v[4] = 1 - abs(v[4] - v[1])  # 独立与粘人负相关
    return {k: round(float(x), 2) for k, x in zip(DIMS, v)}

def behavior_params(p):
    """性格→行为参数(App动画层用)"""
    return {
        "move_speed": 0.3 + p["energy"]*1.2,          # 走动速度
        "follow_dist": 1.5 - p["affection"]*1.2,       # 跟随距离(粘人=近)
        "explore_radius": 2 + p["curiosity"]*8,        # 探索范围
        "startle_prob": p["timidity"]*0.6,             # 惊吓概率(碰撞/大音量)
        "solo_time": p["independence"]*40,             # 独处时长(分钟)
        "pet_response": 0.2 + p["affection"]*0.8,      # 撸宠反馈强度
    }

def evolve(p, interaction, dt=1.0):
    """陪伴养成:互动历史塑造性格(操作-条件-结果,本地零上传)
    interaction: feed/walk/play/pet/scold/ignore"""
    lr = 0.002 * dt
    if interaction == "play":   p["energy"] += lr; p["affection"] += lr*0.5
    elif interaction == "feed": p["affection"] += lr
    elif interaction == "walk": p["curiosity"] += lr; p["energy"] += lr*0.5
    elif interaction == "pet":  p["affection"] += lr*1.5; p["timidity"] -= lr
    elif interaction == "scold":p["timidity"] += lr*2; p["affection"] -= lr
    elif interaction == "ignore": p["independence"] += lr*0.5; p["affection"] -= lr*0.3
    return {k: round(float(np.clip(v, 0.01, 0.99)), 3) for k, v in p.items()}

if __name__ == "__main__":
    p = gen_personality(seed=2026)
    print("初始性格:", p)
    print("行为参数:", {k: round(v,2) for k,v in behavior_params(p).items()})
    for _ in range(100): p = evolve(p, "pet")   # 撸100次
    print("撸100次后:", p)
