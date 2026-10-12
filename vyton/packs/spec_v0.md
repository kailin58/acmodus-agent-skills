# 能力包规范 v0(模型内部·基座分离架构)
> 2026-10-12 · 依用户战略令(基座可换/能力插拔/MoE四思想)

## 1.包结构(zip)
```
capability_<name>_v<ver>.zip
├── manifest.json       # 包清单(见§2)
├── backbone.onnx       # 基座(可选,可共享宿主已装基座)
├── head.onnx           # 头(能力核心,必带)
├── labels.json         # 标签映射
├── postprocess.py      # 后处理(纯函数)
└── meta/               # 校准数据/证据
```

## 2.manifest v1.1
```json
{
  "schemaVersion": "1.1",
  "capability": "style",
  "version": "2.0.0",
  "backbone": {"id": "enb3", "file": "backbone.onnx", "sha256": "...", "feat_dim": 1536, "shared": true},
  "heads": [{"id": "style", "file": "head.onnx", "sha256": "...", "out": [8]}],
  "deps": ["backbone:enb3>=1.0"],
  "ram_mb": 60, "lazy_load": true,
  "routes": ["outfit_query", "scene_wedding"]   # 路由器激活键
}
```

## 3.路由协议(条件计算)
- 输入:intent.json(意图中枢输出)
- 规则:routes[]命中→包载入;未命中→不加载
- 宿主启动只载基座+意图中枢(常驻<80MB)
- 包切换<2s(内存映射载入)

## 4.重蒸SOP(换基座标准流程)
1.新基座过评估基准(标准测集:5990张抽1000)
2.基准达标→提特征(全数据,一次GPU前向)
3.全部头重蒸(逐包,小时级)
4.过各包验收门(IoU/召回不降)
5.新backbone.onnx+重蒸heads打包发版

## 5.懒加载协议
- App侧ModelManager按manifest.lazy_load
- 同基座多包:基座只载一次(shared:true)
- 内存压力→LRU卸载(意图中枢不可卸)

## 6.验收门(每包)
- 模型质量:任务指标不低于现役
- 包校验:model-pack-verify 18项PASS
- 路由:激活/休眠正确
- 卸载:内存回落验证

## 7.第一个包
vyton_cut_v2(训练中):backbone=MobileNetV3-small
+head=FPN上采样,int8后目标≤15MB
