import os
import numpy as np
from demoto import demotoo1
path = "/root/ddpm/ourdata/"
path_new = "/root/ddpm/data1/"

for filename in os.listdir(path):
    a = os.path.join(path, filename)
    data = 1-np.load(a)
    label = np.squeeze(demotoo1(data))
    a = np.around(label[0],3)
    b =  np.around(label[1],3)
    np.save(path_new + str(a)+"_" +str(b)+".npy",data )