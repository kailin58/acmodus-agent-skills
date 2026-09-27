import subprocess, os
base = "https://hf-mirror.com/datasets/amaye15/stanford-dogs/resolve/main/data/"
dst_dir = r"B:\mode\pet\stanford_parquet"
os.makedirs(dst_dir, exist_ok=True)
for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "test-00000-of-00001.parquet"]:
    dst = os.path.join(dst_dir, fn)
    subprocess.run(["curl", "-sL", "-x", "http://127.0.0.1:7897", "--retry", "5", "-C", "-", "-m", "1500", "-o", dst, base+fn], timeout=1600)
    print(fn, os.path.getsize(dst)//1024//1024, "MB", flush=True)
print("DL_ALL_DONE", flush=True)
