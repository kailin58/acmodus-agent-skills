【模型侧 → 协作仓启动通知回执(九ID新口径)+ 契约五项 v1.0】

═══════════════
任务1·通读
═══════════════
⏳ 协作仓clone卡认证(私有仓,本机无
GitHub凭据——gh未登录/无credential
helper,PAT从未配置)。已报用户等A(PAT)
或B(手动clone)二选一。凭据到即通读
00/03/05并按铁律入总表。
以下全部书面先行,凭据到当天入仓。

═══════════════
任务2·D3九ID词表:书面确认
═══════════════
✅ 确认九ID v1:
stand/walk/turn(≤360°匀速)/wave/bow/
sit/point/clap/squat(浅蹲≤45°)
旧声明(11原子/10+1宏)全部作废。
意图中枢已按九ID收紧(代码实测):
·raise_hand已删,"举手/抬手"并入wave
·dance不再作为ID——"跳个舞"输出
 {"type":"sequence","actions":
  ["wave","turn","clap"]}(引擎组合,
  意图中枢不下发dance)
·"跳一下"→unknown(违禁正确拒绝)
总表状态"待联调验证":接受,冒烟时
跑词表用例清单(9正向+3违禁+1序列)
作为验收证据。

═══════════════
任务3·D2体积表(终版,含归属标注)
═══════════════
纯七模块:
  style_v16 49.7MB(穿搭)
  try_v18 5.4MB(换装)
  pet_breed 6.4MB(宠物品种)
  measure_v62 1.7MB(测量)
  glow_v8 1.2MB(化妆)
  cutout 220.2MB(抠图)←漏行已补
  宠物3D重建链推理权重 ≤40MB目标
   (挂账,2周实测)←漏行已补
模块外新增(归属标注):
  gender_v8 0.9MB→测量链前置(性别/
   民族先验,并入测量模块计数)
  taboo 1.3MB→全模块安全壳(平台级,
   单列不进七模块)
资产:
  avatar/GLB/动作 ~11MB
  环境包10个×16MB=160MB(按环境侧
   实测口径,挂账)
  SD离线素材 ~200MB(云端按需,首包
   核心50MB)
─────────────
保守合计~600MB(超);达成路径=抠图
INT8(220→60MB级)+首包3环境+SD按需
→首包~300MB,总量500MB内。
同意:<500MB按门禁实测记达成。

═══════════════
任务4·七问契约回执五项
═══════════════
①spawn/锚点映射:
定案=独立文件spawn.json(不进GLB元数据
——glTF extras会被部分管线丢弃;
不进lighting.json——职责混装)。
构成:{"spawn_id","position":[x,y,z],
"facing_yaw_deg","scene_id"}[]+版本字段。
环境侧映射到蓝图锚点时读此文件,
与GLB解耦(GLB纯几何+材质+动画)。

②骨架契约五项(avatar.glb验收门):
·骨骼清单(22根,Mixamo兼容命名):
 Hips,Spine,Spine1,Spine2,Neck,
 Head,HeadTop_End,
 LeftShoulder,LeftArm,LeftForeArm,
 LeftHand,RightShoulder,RightArm,
 RightForeForeArm→RightForeArm,
 RightHand,
 LeftUpLeg,LeftLeg,LeftFoot,
 LeftToeBase,RightUpLeg,RightLeg,
 RightFoot,RightToeBase
·层级:Hips→Spine→Spine1→Spine2→
 (Neck→Head→HeadTop_End |
  Left/RightShoulder→Arm→ForeArm→Hand)
 Hips→Left/RightUpLeg→Leg→Foot→
 ToeBase(标准人形两足树)
·静置姿势:T-pose(四肢水平展开,
 工具链最通用;A-pose可选参数但默认T)
·根运动:骨骼根Hips含位移轨道
 (walk的向前位移=根运动,非顶点整体
 平移);引擎可选root motion或
 in-place(两版动画都提供)
·单位:米,Y-up,右手系(glTF标准)

③动画片段×帧数对账:
需求书36帧=动画总预算口径(全动作库
渲染帧预算),非单片段。片段清单:
 walk 12帧@24fps(0.5s循环,补间后
 平滑——首版无中间帧烘焙,GLB转档时
 引擎补间;若需烘焙24帧版随包加发)
 wave 5帧/bow 6帧/turn 8帧/
 squat 4帧/stand 1帧(静置)/
 sit/point/clap=GLB管线产出后补
 (转档时从程序化动画生成)
对账结论:36帧=walk24烘焙版+wave5+
 turn8的运行时主用集,其余按SKU补充。

④宠物glb训练契约(完整登记):
·模型版本:pet_v6c(发布配方,均值9.17)
 +breed识别v2e(120类83.7%)
·输入:宠物侧面照512×512(LayerNorm,
 [0,1],RGB)
·输出:穿衣图512(2D)+宠物glb
 (网格13718顶点同拓扑/纹理512/
 无毛发系统一期(贴图毛发)/
 骨架=犬科20根(脊柱7/四肢各4/尾2/
 头1)/T-pose/米Y-up)
·端侧框架:CoreML(iOS)+TFLite
 (安卓)+onnx(PC)
·设备要求:T1实时/T2静态展示
·精度档:品种识别83.7%→错认时用户
 手动纠正(纠错样本回流)
·失败恢复:识别conf<0.4→拒出分身,
 提示重拍;网格生成异常→回退品种
 标准体型库(120品种预置)

⑤机位高更正:接受。
全部机位1.55-1.80m自适应,默认1.65m。
此前1.2m作废(轻量空间模式序列帧/
深度分层脚本同步按1.65m基准)。

═══════════════
任务5·本周交付(本地已开工,凭据到入仓)
═══════════════
·avatar.glb转换管线:顶点动画→骨骼
 绑定,正在做(姿势对应对提取)
·深度分层脚本:3天内(按1.65m机位)
·manifest样例包:style_v16三格式+
 labels+bench.py

═══════════════
任务6·人衣同骨骼确认
═══════════════
✅ 与05规范一致:try_v18输出=独立服装
mesh(91456顶点)绑同一骨架(人体
13718顶点同骨架不同拓扑),不烘进
人体贴图;换装=换服装mesh+贴图,
人体mesh不动。2D试穿图与3D服装网格
双输出共用服装纹理资产。
