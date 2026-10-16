
## 10-17 云端Stable-Makeup教师部署战役(第3朵A800)
- 教师验收: 妆效8/10(四标尺最高: SD15=4.5/IPA=6/PSGAN=5.5/SM=8) 身份6.5(肤色漂移+磨皮系统性,蒸馏必须脸区ROI+肤色直方图约束)
- 200伪标签0错误落地 B:/mode/models/glow/teacher_sm/
- 七坑: ①hf-mirror大文件静默截断(X-Linked-Size校验+curl -C -续传,3个bin全截断过) ②repo自带diffusers fork必须优先于site-packages 0.31(API不兼容: detail_encoder被当ControlNet调) ③hub版本两难fork要cached_download(≤0.16)+新API(≥0.20)→0.23.5+patch fork import ④GoogleDrive权重云端不通→本地SOCKS代理下载+sftp上传(SPIGA 254MB+Retinaface) ⑤spiga/facelib不在pypi→github clone ⑥官方用法=pipeline挂[id,pose]+makeup_encoder.generate()外驱动(infer_kps.py是ground truth,先读官方demo再写) ⑦numpy 2.x反复被pip拉高→钉1.26.4
- RSI: 部署第三方repo第一步=读官方infer demo(不是猜API);hf-mirror下载必校验真实大小

## 10-17 v19蒸馏三连败止损(铁律执行)
- v19.0: 架构bug当场抓(输入cat([S,教师输出])=抄答案变体,未训)
- v19.1: 修正参考图通道,vision=2/10(学生塌缩为遮罩内刷单一均值色,眼妆零,参考条件失效)
- v19.2: 逐像素色场+alpha架构(PSNR 20.7dB),vision=2/10(幅度塌缩→恒等映射+轻磨皮,唯一残留=美颜先验)
- 根因(两条都实证): ①SD教师输出含全局美化(磨皮/提亮/年轻化)——学生'背景不变+只迁妆'渲染机制做不到→脸区L1被非妆容差异主导→恒等平凡解最优 ②几何椭圆mask太粗(唇位偏移/无解析形状)→贴片感
- RSI新教训: **扩散教师蒸馏前必须先做妆容差分提取**(教师-素颜delta,非妆容区清零),否则教师的美颜先验污染监督信号;alpha类解耦学生要监控平均|delta|防恒等坍缩;椭圆先验mask永远打不过解析mask
- 下一步选项: A.差分提取+mediapipe解析mask重蒸馏 B.弃像素蒸馏改参数蒸馏(唇色/眼妆色/腮红3参数+解析mask渲染,教师200对标定) C.挂账转其他任务
