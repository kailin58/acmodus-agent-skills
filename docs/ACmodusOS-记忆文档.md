# ACmodus 项目记忆文档(长期维护,接手即读)

> 建立:2026-09-17 · 用途:归档用户全部要求与项目状态,避免反复询问 · 每次会话有新要求/新进展立即更新本档

---

## 〇、项目总纲(永远第一优先级)
**训练端侧模型是唯一目的。** 6模块端侧模型(穿搭/换装/换脸/化妆补妆补光/抠图/真3D动态+拍照测量),蒸馏≤500MB、内存≤3GB、装进手机离线跑、隐私零上传,支撑ACmodus 3D衣帽间完整体验(找衣→取衣→换装→搭配→购买)。Pinterest采集=给穿搭+化妆模块喂风格公式/妆容素材。
**训练目标:足够轻量 + 性能10倍好**(蒸馏后端侧模型:体积足够轻量、推理性能10倍提升,对齐Mobile-VTON路线:415M/INT8)。

## 一、用户明确要求(已归档,不再重复确认)

### 1. 六大模块(全部=端侧模型训练标准)
| # | 模块 | 含子能力 | 现状资产 |
|---|---|---|---|
| 1 | 穿搭 | 搭配推荐/风格公式 | 6个JSON知识库就绪;Pinterest采集管线就绪 |
| 2 | 换装 | VTON虚拟试穿 | M1✅;M2 LoRA待跑 |
| 3 | 换脸 | 人脸替换 | 待启动 |
| 4 | 化妆 | 化妆+补妆+补光 | 妆容映射知识库就绪 |
| 5 | 抠图 | 人像/衣服分离 | rembg就绪 |
| 6 | 图片变3D | 真3D可动态(Avatar+骨骼绑定) | 待启动 |
| + | 拍照测量 | 尺寸/身材测量 | ✅调研完成:B:\mode\调研-拍照测量人体数据.md(MediaPipe新Task API+Anny人体模型+身高锚点,全Apache2.0,双端2020机型可跑)→下一步POC |

### 2. 工作方式
- **全自主执行不问确认**(用户原话:你自己干活训练不要我点确认/自己想办法)
- 所有要求必须归档不反复问;维护本记忆文档;遇阻自行换路线;结果归档汇报

### 5. 拍照测量模块调研指令(语音 2026-09-17)
- **先做充分调研报告再动手**:选最优方案
- 要求:开源+可商用(Apache/MIT,排除NC);相机参数(距离/焦距/物距)处理方案;苹果+2020安卓都可用
- 产出:调研报告归档 B:\mode\ + GitHub技能库

### 3. 目标机型(硬约束)
- **iPhone 13(A15/4GB)+2020年起的安卓(骁龙865级)必须可用**:真3D穿搭(动态)/换脸/拍照测量人体数据(语音补充:2020安卓也要能跑,两大阵营都覆盖)
- 部署:CoreML(ANE)+NNAPI/Hexagon,INT8,模型≤500MB,内存≤3GB

## 二、进度状态(2026-09-17 02:00)

| 事项 | 状态 |
|---|---|
| 仓库克隆 B:\mode\ACmodusOS-modeisavailable | ✅ 9.3MB/33文件 |
| 记忆文档+归档 | ✅ 本文档 |
| GitHub技能库 | ✅ kailin58/acmodus-agent-skills(commit eb41404,60文件) |
| Pinterest采集 | ✅ 10关键词53图原图,PIL全通过(449~1179px),B:\mode\pinterest\images |
| 本机训练环境 | ✅ torch2.6.0+cu124 CUDA可用(**须带PYTHONUSERBASE跑**,见坑4) |
| M2 LoRA训练 | ⏸️ 未开跑(卡:底模+数据集缺失,见困难1) |

## 三、遇到的困难与对策

1. **移交资产不在本机**:D:\harnessAgnes\vton-training\(54GB底模+5.58GB数据集)不存在——上台机器路径。⇒从魔搭重新拉(国内直连免代理)
2. **8GB显存装不下20B底模**:Qwen-Image-Edit(20B)LoRA必须上魔搭GPU;本机只做:学生模型(415M)训练/量化/蒸馏验证/抠图/测量小模型
3. **Hermes内置浏览器工具坏**:Chromium检测只认Playwright缓存目录+常驻缓存,已绕开(直调Chrome headless --dump-dom;细节在技能库skills/pinterest-collect/SKILL.md)
4. **Python环境混乱+torch坑**:4个Python;锁定`py -3.13`;**torch装在Administrator user-site,而hermes会话HOME=systemprofile,import torch必须先设`PYTHONUSERBASE=C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages`**;pip输出必须重定向到文件(GBK解码崩)
5. **网络**:全靠SOCKS5 127.0.0.1:7897;git全局10809死(勿改全局,命令级`-c http.proxy=socks5://127.0.0.1:7897`);魔搭国内直连不用代理;**terminal工具偶发WSL报错崩**(报WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED),用execute_code兜底
6. **Pinterest匿名限额**:每词约6pin。风格参考够用,训练数据不够——主力=VITON-HD 13k对;更多Pinterest需登录态cookie(未做)

## 3.5 端侧体积总红线(2026-09-17语音指令)
- App安装总占用≤1GB红线/≤600MB目标;运行内存≤3GB;用户数据+缓存≤1.5GB自动封顶
- 六模块预算:VTON350/换脸80/3D基底80/化妆30/测量15/穿搭15/抠图10/预留20(MB)
- 衣服3D资产按需下载不打包;生成图LRU 500MB;详见 B:\mode\端侧体积红线v2.1-六模块版.md

## 四、下一步(自主推进)
1. 魔搭拉VITON-HD-Captions(5.58GB)+Qwen-Image-Edit(**唯一等用户的点:魔搭手机号注册**)
2. M2 smoke test:10对样本500步(魔搭A10;或本机换小底模SD1.5先验管线)
3. 并行:抠图模块(rembg批处理)/拍照测量(MediaPipe体测POC)/学生模型蒸馏框架
4. 每完成一项更新本档+推GitHub技能库
