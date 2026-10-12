
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForImageSegmentation
m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
m.save_pretrained(r"B:\mode\models\cut\teacher_birefnet")
print("TEACHER_SAVED", flush=True)
