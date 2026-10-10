"""vyton.intent — 意图中枢 v1.1(高内聚:只管理解,窄输出)
语言意图:文本→动作ID | 环境意图:文本→场景→穿搭+动作建议
端侧:纯规则引擎,<1MB,零依赖
"""
import re

ACTIONS = {
    "walk": {"frames": "walk12", "loop": True, "desc": "行走"},
    "turn": {"frames": "turn36", "loop": False, "desc": "转身"},
    "raise_hand": {"frames": "raise_arm", "loop": False, "desc": "抬手"},
    "stand": {"frames": "f0_stand", "loop": True, "desc": "站立"},
    "sit": {"frames": "sit", "loop": False, "desc": "坐下"},
    "jump": {"frames": "jump8", "loop": False, "desc": "跳"},
    "wave": {"frames": "wave12", "loop": True, "desc": "挥手"},
    "clap": {"frames": "clap10", "loop": True, "desc": "鼓掌"},
    "squat": {"frames": "squat8", "loop": False, "desc": "蹲下"},
    "point": {"frames": "point", "loop": False, "desc": "指"},
    "bow": {"frames": "bow8", "loop": False, "desc": "鞠躬"},
    "dance": {"frames": "dance24", "loop": True, "desc": "跳舞"},
}

PATTERNS = [  # 顺序=优先级(长模式在前防误吞)
    (r"跳[个]?舞|跳舞|舞蹈|dance", "dance"),
    (r"跳[一]?下|跳远|跳高|jump", "jump"),
    (r"走|walk|前进|过来|go", "walk"),
    (r"转身|转圈|turn|spin", "turn"),
    (r"抬手|举手|伸手|raise|hand up", "raise_hand"),
    (r"站|stand|停|stop", "stand"),
    (r"坐|sit", "sit"),
    (r"挥手|招手|wave", "wave"),
    (r"鼓掌|拍手|clap", "clap"),
    (r"蹲|squat", "squat"),
    (r"指|point", "point"),
    (r"鞠躬|bow", "bow"),
]

SCENES = {
    "office":     {"kw": r"办公|上班|公司|开会|office|work", "outfit": "smart_casual", "act": "stand"},
    "gym":        {"kw": r"健身|运动|跑步|gym|sport|workout", "outfit": "sportswear", "act": "squat"},
    "date":       {"kw": r"约会|吃饭|晚餐|date|dinner", "outfit": "casual_chic", "act": "wave"},
    "wedding":    {"kw": r"婚礼|结婚|wedding|bride", "outfit": "formal", "act": "bow"},
    "interview":  {"kw": r"面试|interview", "outfit": "business", "act": "stand"},
    "party":      {"kw": r"派对|聚会|party|club", "outfit": "evening", "act": "dance"},
    "travel":     {"kw": r"旅行|旅游|travel|trip", "outfit": "comfort", "act": "walk"},
    "home":       {"kw": r"在家|家里|home|relax", "outfit": "loungewear", "act": "sit"},
    "outdoor":    {"kw": r"户外|爬山|徒步|outdoor|hike", "outfit": "outdoor_gear", "act": "walk"},
    "school":     {"kw": r"学校|上学|校园|school|class", "outfit": "campus", "act": "stand"},
}


def parse(text):
    """语言→意图。窄输出dict:action或scene二选一"""
    t = text.lower()
    for pat, act in PATTERNS:
        if re.search(pat, t):
            return {"type": "action", "action": act, "params": ACTIONS[act]}
    for scene, cfg in SCENES.items():
        if re.search(cfg["kw"], t):
            return {"type": "scene", "scene": scene, "outfit": cfg["outfit"], "action": cfg["act"]}
    return {"type": "unknown"}
