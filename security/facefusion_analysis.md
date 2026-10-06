
=== FaceFusion借鉴分析(2026-10-05) ===

【它是什么】MIT开源换脸管线:检测→身份编码→生成→混合修复,28.4k星最活跃换脸项目

【四阶段管线拆解】(2026行业事实标准)
1.检测landmarks:InsightFace buffalo_l(极端角度/遮挡/小脸是主要失败源)
2.身份编码:ArcFace向量(单照one-shot,无需per-identity训练!)
3.生成:InSwapper(ArcFace条件化StyleGAN2编解码器,AdaIN融合身份+属性)
4.混合修复:羽化边缘mask合成+CodeFormer/GFPGAN复原(2026效果飞跃的最大功臣)

【对我们③换脸/虚拟自己模块的4个借鉴点】
1.**身份向量解耦思想**:我们的虚拟自己现在是"点彩风重建"——可引入ArcFace身份embedding作为条件(保身份不变,换姿态/光照/风格)。我们已有blaze_face,补ArcFace类身份头即可
2.**修复通道**:我们Try学生v11r的"条纹/发丝保真"思路=它们的CodeFormer位——但FaceFusion证明了"生成后独立修复pass"价值巨大→我们v8r回归学生天然适合做这个修复头(单步前向!)
3.**one-shot理念**:单参考照即可——与我们"用户拍一张生成虚拟自己"产品定义完全一致,验证了路线
4.**实时模式工程**:检测一次+跨帧跟踪(不重复检测)/ROI最小化/显存预分配——虚拟自己动态展示时的端侧优化清单

【许可证判定】
-FaceFusion本身MIT=可借鉴架构思想(我们零代码引用,全自研✓铁律)
-⚠ InSwapper模型=InsightFace商用需授权(非MIT!)→我们不用它的权重,自己训练身份条件生成头
-CodeFormer/GFPGAN:CodeFormer可商用(S-Lab)/GFPGAN Apache——修复头可参考

【行动项】
虚拟自己v15:加ArcFace身份条件头(自训)+v8r回归式修复通道——预计8.5→9+(身份稳定性质变)
