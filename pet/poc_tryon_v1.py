
# Vyton-Pet-Try贴装POC:狗图+毛衣→贴身穿着效果
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

img = Image.open(r"B:\mode\pet\poc_kps.png").convert("RGB")
W, H = img.size
# POC关键点(256系,已验证的)
KPS = {"左眼":(152,64),"右眼":(111,58),"鼻":(134,76),"颈":(132,134),"左肩":(154,182),
       "右肩":(107,180),"左髋":(160,197),"右髋":(94,200),"尾根":(126,36)}
neck = np.array(KPS["颈"]); ls = np.array(KPS["左肩"]); rs = np.array(KPS["右肩"])
lh = np.array(KPS["左髋"]); rh = np.array(KPS["右髋"])

# 服饰=红色毛衣(程序生成POC,真实场景用商品白底图)
# 覆盖区:颈下→胸腹(肩线到髋线),沿身体轮廓
top_w = np.linalg.norm(ls-rs)         # 肩宽
bot_w = np.linalg.norm(lh-rh) * 1.1   # 下摆
cx = (ls[0]+rs[0]+lh[0]+rh[0])/4
y_top = neck[1] + top_w*0.35
y_bot = min(lh[1], rh[1]) + 8

# 躯干轮廓(左右曲线:肩→髋外扩)
def edge(t):  # t:0顶1底
    half = (top_w/2*(1-t) + bot_w/2*t) * 0.78
    return cx-half, cx+half
lay = Image.new("RGBA", (W,H), (0,0,0,0))
d = ImageDraw.Draw(lay)
# 主体衣身(梯形+圆角)
steps = 24
pts_l, pts_r = [], []
for i in range(steps+1):
    t = i/steps
    y = y_top + (y_bot-y_top)*t
    l, r = edge(t)
    pts_l.append((l, y)); pts_r.append((r, y))
poly = pts_l + pts_r[::-1]
d.polygon(poly, fill=(200,30,40,215), outline=(150,15,25,255))
# 领口(椭圆挖空,透明)
d.ellipse((cx-top_w*0.22, y_top-top_w*0.10, cx+top_w*0.22, y_top+top_w*0.16), fill=(0,0,0,0))
# 袖子(左右臂沿肩→肘方向)
d.line([tuple(ls), (ls[0]+14, ls[1]+26)], fill=(200,30,40,215), width=int(top_w*0.30))
d.line([tuple(rs), (rs[0]-14, rs[1]+26)], fill=(200,30,40,215), width=int(top_w*0.30))
# 纹理:针织竖线
for i in range(1, 8):
    x0 = edge(0)[0] + (edge(0)[1]-edge(0)[0])*i/8
    d.line([(x0, y_top+4), (x0+(edge(1)[0]-edge(0)[0])*0.9, y_bot-4)], fill=(160,20,30,120), width=1)

lay = lay.filter(ImageFilter.GaussianBlur(1.2))
out = Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")
d2 = ImageDraw.Draw(out)
d2.text((10,10), "Pet-Try POC v1: sweater fitted by 17-kps", fill=(255,255,255))
out.save(r"B:\mode\pet\poc_tryon.png")
print("PET_TRY_POC_DONE")
