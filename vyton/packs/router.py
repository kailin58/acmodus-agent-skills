"""vyton.packs.router — 能力路由器v1(MoE条件计算落地)
输入:intent(意图中枢输出) → 输出:应加载的能力包清单
宿主常驻:基座+意图中枢(<80MB);包按需载入
"""
import json, os

# 注册表:包路由表(随包manifest的routes字段自动聚合)
REGISTRY = {
    # capability: {routes(激活键), deps, ram_mb}
    "cutout":  {"routes": ["photo_input", "cutout"], "deps": [], "ram": 30},
    "style":   {"routes": ["outfit_query", "scene.*"], "deps": ["cutout"], "ram": 60},
    "measure": {"routes": ["photo_input", "measure"], "deps": ["cutout"], "ram": 20},
    "avatar":  {"routes": ["create_avatar", "photo_input"], "deps": ["cutout", "measure"], "ram": 40},
    "pet":     {"routes": ["pet_input", "pet_tryon"], "deps": ["cutout"], "ram": 35},
    "tryon":   {"routes": ["outfit_query", "tryon"], "deps": ["cutout", "style"], "ram": 50},
    "glow":    {"routes": ["makeup"], "deps": ["cutout"], "ram": 10},
}

def route(intent):
    """意图→包清单(含依赖闭包+内存预算)
    intent: 意图中枢输出dict {type, action/scene/...}"""
    need = set()
    t = intent.get("type", "")
    if t in ("photo_input", "photo") or intent.get("photos"):
        need.add("cutout")
    if t == "action" or intent.get("action"):
        need.add("avatar")  # 动作需要3D分身
    if t == "scene" or intent.get("scene"):
        need.add("style")   # 场景→穿搭建议
        need.add("tryon")
    if t in ("makeup",) or intent.get("outfit") == "makeup":
        need.add("glow")
    if intent.get("pet"):
        need.add("pet")
    if intent.get("measure"):
        need.add("measure")
    # 依赖闭包
    for _ in range(3):
        for c in list(need):
            for d in REGISTRY.get(c, {}).get("deps", []):
                need.add(d)
    total_ram = sum(REGISTRY.get(c, {}).get("ram", 30) for c in need)
    return {"load": sorted(need), "ram_mb": total_ram,
            "idle_caps": sorted(set(REGISTRY) - need)}

# 自测
if __name__ == "__main__":
    cases = [
        {"type": "action", "action": "walk"},              # 走两步→avatar+cut
        {"type": "scene", "scene": "wedding"},             # 婚礼→style+tryon+cut
        {"type": "photo_input", "photos": ["a.jpg"]},      # 拍照→cut(+measure按需)
        {"type": "photo_input", "pet": True},              # 宠物照→pet+cut
        {"type": "unknown"},
    ]
    for c in cases:
        r = route(c)
        print(f"{str(c)[:40]:42s} → 载入{r['load']} {r['ram_mb']}MB | 省掉{r['idle_caps']}")
