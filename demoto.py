from skimage import measure
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure

# 计算切圆直径的函数
def compute_max_diameter(region_labels):
    coords = np.transpose(np.nonzero(region_labels))
    dm = pdist(coords)
    if dm.size != 0:
        max_value = np.max(dm)
    else:
        max_value = 0 # 或其他你想要的默认值
    return max_value

# 生成示例3D数据
# data = np.zeros((10, 10, 10), dtype=np.uint8)
# data[1:4, 1:4, 1:4] = 1
# data[6:9, 6:9, 6:9] = 1

# 进行连通域标记
# labeled_data, num_features = measure.label(data, return_num=True)

# 提取每个连通域的切圆直径
# for i in range(1, num_features + 1):
#     region_labels = labeled_data == i
#     diameter, (x, y, z) = compute_max_diameter(region_labels)
#     print(f"Region {i}: diameter={diameter:.2f}, center=({x}, {y}, {z})")


def demotoo(a):
    vsb = []
    for data in a:
        labeled_data, num_features = measure.label(data, return_num=True)
        vs = []
        for i in range(1, num_features + 1):
            region_labels = labeled_data == i
            diameter= compute_max_diameter(region_labels)
            vs.append(diameter)
        vs = np.array(vs)
        idex = np.nonzero(vs)
        vb = vs[idex]
        tt = np.mean(vb)
        bb = np.std(vb)
        vsb.append([tt,bb])
    return np.array(vsb)

def demotoo1(data):
    vsb = []
    labeled_data, num_features = measure.label(data, return_num=True)
    vs = []
    for i in range(1, num_features + 1):
        region_labels = labeled_data == i
        diameter= compute_max_diameter(region_labels)
        vs.append(diameter)
    vs = np.array(vs)
    idex = np.nonzero(vs)
    vb = vs[idex]
    tt = np.mean(vb)
    bb = np.std(vb)
    vsb.append([tt,bb])
    return np.array(vsb)