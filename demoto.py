from skimage import measure
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure

def compute_max_diameter(region_labels):
    coords = np.transpose(np.nonzero(region_labels))
    dm = pdist(coords)
    if dm.size != 0:
        max_value = np.max(dm)
    else:
        max_value = 0 
    return max_value




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
