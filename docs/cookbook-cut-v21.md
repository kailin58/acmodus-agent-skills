# cut_v2.1 能力包接入手册(Cookbook)

## 概述
cut_v2.1 = 轻量抠图能力包(两件套:骨干+头),单图输出alpha遮罩。
- 体积: 4.1MB | 推理: 14ms(256px, 骁龙888) | IoU: 0.883 | 软边比v1提升15.8倍
- 基座分离: 骨干可换(manifest.backbone字段), 头部自有资产

## 依赖
- ONNX Runtime ≥1.16 (Android/iOS均有官方包)
- 输入: RGB uint8, 任意尺寸(内部resize到320)
- 输出: alpha [H,W] float32 0-1

## 接口(Python示例)
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("cut_v21.onnx", providers=["CPUExecutionProvider"])
img = Image.open("input.jpg").convert("RGB")
x = np.asarray(img.resize((320,320)), dtype=np.float32) / 255.0
x = x.transpose(2,0,1)[None]
alpha = sess.run(None, {"input": x})[0][0,0]
alpha = np.array(Image.fromarray((alpha*255).astype(np.uint8)).resize(img.size)) / 255.0
rgba = np.dstack([np.asarray(img), (alpha*255).astype(np.uint8)])
Image.fromarray(rgba, "RGBA").save("out.png")
```

## Android(Kotlin)要点
- 放assets/, ONNX Runtime加载
- 输入NCHW float32, 归一化/255
- 输出[1,1,320,320]→resize回原尺寸

## 常见坑
1. 输入必须RGB不是BGR(转一下)
2. resize用双线性(PIL默认)不用最近邻
3. 模型是FP16版——不要INT8量化(误差34.5%,实证翻车)
4. 软边场景(毛发)输出alpha有渐变,别二值化

## manifest字段
```json
{"id":"cut", "version":"2.1", "backbone":"enb3", "head":"cut_head_v21",
 "size_mb":4.1, "latency_ms":14, "metrics":{"IoU":0.883}}
```

## 验收基准
- 42项vyton.core测试常驻,接入后跑一遍
- 视觉验收:毛发边缘≥6/10
