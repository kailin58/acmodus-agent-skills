---
name: pinterest-collect
description: 用无头Chrome采集Pinterest公开搜索页的穿搭/美妆素材,免登录,断点续跑,自动下载原图
version: 0.1.0
---

# Pinterest 公开采集技能

## 用法
```bash
python collect.py "关键词1|关键词2|..." hashes.json   # 抓搜索页提取pin
python download.py                                    # 下载原图(需SOCKS5代理127.0.0.1:7897)
```

## 要点(踩坑记录)
1. **免登录路径**:用 `/search/pins/?q=`,不用 `/search/my_pins/`(要登录)。匿名页每词约6个pin可见
2. **浏览器**:`chrome --headless=new --dump-dom --virtual-time-budget=10000`,DOM约1.2MB
3. **数据在DOM**:pin图URL形如 `i.pinimg.com/236x/ab/cd/ef/<hash>.jpg`,hash是唯一ID;改URL前缀为 `originals` 得原图
4. **Pinterest API(BaseSearchResource)匿名403**,别走API
5. **图床i.pinimg.com需代理**(SOCKS5 7897),直连超时
6. 搜索联想词(截图下拉)是现成关键词池:女夏/歐美/肉肉女/海報設計/韓國夏天/ins/男成熟/街頭風/ootd
7. 关键词间 sleep 1s,单页 timeout 70s,防封
