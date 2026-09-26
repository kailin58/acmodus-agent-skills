# Vyton端侧资产盘点 v1(2026-09-27)

## 核心结论:500MB红线内!现在全量235MB,余265MB

## 按模块(端侧选用,非全部27个文件)
| 模块 | 端侧模型 | 大小 | 状态 |
|---|---|---|---|
| Cut抠图 | rembg u2net | ~170MB→INT8可压88MB | 待量化 |
| 3D深度 | depth_v2_small_quantized | 26MB | ✅已量化 |
| 测量 | measure_v3.1三件+v6.2 | 4MB | ✅极轻 |
| 性别脸 | gender_v3 | 0.9MB | ✅ |
| 风格 | style_v4 | 0.9MB | ✅ |
| 质量门 | noul_v1 | 0.3MB | ✅ |
| Me脸关键点 | face_landmarker | 3.6MB | ✅ |
| 姿态 | pose_landmarker_lite | 5.5MB | ✅ |
| 品种识别 | resnet34 | **81.4MB→蒸馏MobileNet~10MB** | 待蒸馏 |
| Pet17点 | tflite端侧版 | 27.5MB | ✅(已备) |
| SR增强 | (撤链,Lanczos替代) | 0 | — |
| Try教师 | SD1.5 | 4.3GB | ❌不上端侧(蒸馏学生~100-200MB待训) |

## 端侧合计(选用后):~160MB(不含Try学生)
+Try学生蒸馏后估100-200MB → 总~300MB ✅红线内

## P0行动
1. 品种:resnet34→MobileNetV3蒸馏(81→10MB)
2. Cut:rembg INT8量化(170→88MB)
3. Try学生蒸馏(SD1.5→小模型)——最大件,基础建设核心
