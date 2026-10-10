
import sys
sys.path.insert(0, r"B:\\mode\\vyton")
from body3d.photo_to_self import photo_to_self
import numpy as np
img = r"B:\\mode\\pinterest\\images\\0486a67d21a3376b92fef7ee5f91d012.jpg"
mesh, rep = photo_to_self([img])
print("照片测量:", rep.get("photo_measure"))
print("参数:", {k: rep["params"][k] for k in ["sex", "height", "weight"]})
print("网格高:", round((mesh[:,2].max()-mesh[:,2].min())*100), "cm")
print("推算:", rep["inferred"][:4])
