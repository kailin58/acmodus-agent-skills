
# GLB自验证:pygltflib重载+结构检查
from pygltflib import GLTF2
import numpy as np
g = GLTF2.load(r"B:\mode\models\body3d\avatar_rigged.glb")
print("场景:", len(g.scenes), "| 节点:", len(g.nodes), "| 皮肤:", len(g.skins), "| 动画:", len(g.animations))
sk = g.skins[0]
print("蒙皮关节数:", len(sk.joints), "| 骨架根:", g.nodes[sk.skeleton].name)
an = g.animations[0]
print("动画:", an.name, "| 通道:", len(an.channels), "| 采样:", len(an.samplers))
names13 = ["Hips","LeftUpLeg","LeftLeg","LeftFoot","RightUpLeg","RightLeg","RightFoot","LeftArm","LeftForeArm","LeftHand","RightArm","RightForeArm","RightHand"]
allnodes = [n.name for n in g.nodes if n.name]
hit = [b for b in names13 if b in allnodes]
print(f"App 13槽位: {len(hit)}/13", "✓合规" if len(hit)==13 else "缺:"+str([b for b in names13 if b not in allnodes]))
print("骨骼树样例:", [f"{n.name}<-{g.nodes[n.children[0]].name if n.children else '叶'}" for n in g.nodes[1:4] if n.name])
