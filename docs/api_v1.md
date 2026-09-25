# Vyton模型侧 API 对接文档 v1.0(2026-09-25)

> 给APP开发侧(用户)的模型接口定义。模型=Vyton家族,全离线,图进图出。
> 每个接口:输入(图片+参数)→输出(图片/mesh/数值/JSON)。无文字通道。

## 六模块统一调用约定
- 输入:图片文件(jpg/png,≤20MB)+ JSON参数
- 输出:按模块schema(见下)
- 延迟目标:T1设备<2s/模块
- 全部本地推理,零上传

---

## 模块1:穿搭(Vyton-Style + 颜色引擎)
```
in : { image, season?, occasion?, culture_profile? }
out: { colors: [{name,hex,confidence}], categories: [{name,confidence}],
       outfit_suggestion: {items:[{category,color,reason_score}], taboo_filtered: bool},
       quality_score: 0-1 }
```
模型:vyton_style_v3.pt(颜色12类+品类9类)+color_rules_v1(39条)+禁忌库(19组)

## 模块2:换装(Vyton-Try)
```
in : { person_image, garment_image }
out: { tryon_image(png), quality_score }
```
管线:抠图→白底→深度→mesh→贴身投影。教师sd_lora_v5(蒸馏前APP用云端/演示)

## 模块3:虚拟自己/换脸(Vyton-Me)
```
in : { face_image, target_pose_image? }
out: { avatar_3d(glbin), face_score:0-1 }
```
478关键点→Anny头部形变+纹理投影

## 模块4:化妆(Vyton-Glow)
```
in : { face_image, makeup_style: "commute|date|stage|daily", shade: "red|bean|coral" }
out: { makeup_image, quality_score(Noul门0-1) }
```
vyton_glow(缎面gloss算法)+Noul门v1。**只做静态照片,无实时视频流(专利规避)**

## 模块5:抠图(Vyton-Cut)
```
in : { image }
out: { cutout_image(png, RGBA), foreground_ratio }
```

## 模块6:3D动态+测量(Vyton-3D + Vyton-Measure)
```
in : { person_image, frames?: 12, scene?: "studio|street|stage" }
out: { mesh(glbin,13718顶点/104骨骼),
       walk_gif, scene_render,
       measurements: {height_cm±,weight_kg±,shoulder_cm±,hip_cm±,torso_cm±,leg_cm±,
                      gender(0-1+conf), confidence_each} }
```
vyton_measure_v3.1.pt(6维集成)+性别头v2(75.8%自动预填,用户可改)

---

## 设备适配
- T1(iPhone13+/2022安卓):3D 36帧
- T1中档:24帧 / T2:2.5D 15帧
- 设备档案首次检测持久化,30天复核

## 升级
- 模型热更新(差异下载+签名+A/B槽)
- 本地个性化:操作-条件-结果记忆(零上传)

## 安全
- 只收图/无文字通道/输出白名单六schema/破解乱码
