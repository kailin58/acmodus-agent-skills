
## 10-17 云端Stable-Makeup教师部署战役(第3朵A800)
- 教师验收: 妆效8/10(四标尺最高: SD15=4.5/IPA=6/PSGAN=5.5/SM=8) 身份6.5(肤色漂移+磨皮系统性,蒸馏必须脸区ROI+肤色直方图约束)
- 200伪标签0错误落地 B:/mode/models/glow/teacher_sm/
- 七坑: ①hf-mirror大文件静默截断(X-Linked-Size校验+curl -C -续传,3个bin全截断过) ②repo自带diffusers fork必须优先于site-packages 0.31(API不兼容: detail_encoder被当ControlNet调) ③hub版本两难fork要cached_download(≤0.16)+新API(≥0.20)→0.23.5+patch fork import ④GoogleDrive权重云端不通→本地SOCKS代理下载+sftp上传(SPIGA 254MB+Retinaface) ⑤spiga/facelib不在pypi→github clone ⑥官方用法=pipeline挂[id,pose]+makeup_encoder.generate()外驱动(infer_kps.py是ground truth,先读官方demo再写) ⑦numpy 2.x反复被pip拉高→钉1.26.4
- RSI: 部署第三方repo第一步=读官方infer demo(不是猜API);hf-mirror下载必校验真实大小
