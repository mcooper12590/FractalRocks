import imageio
import numpy as np
import matplotlib.pyplot as plt
from c_corr2d import ts_corr as corr
# Load image to check. Img should just be thin section, with rest
# masked off as zeros.
iname = "PZ103_Initial_SqCrop.tif"
img = imageio.imread(iname).astype(np.float32) # Type should have number of bits higher than original data type, e.g. 16bit uint -> 32bit float
print("Loaded img: " + iname)

ofile="correlation_data/c_e01.dat"
efile="correlation_data/e01.dat"

mask = img>0 # Mask of indices where core is located
g_bar = np.mean(img[mask]) # mean grayscale value in thin section 
img[mask] -= g_bar
norm_f = np.mean(img[mask]**2) # second moment for corr.

#n_e = 50 # Number of distances to try
#e = np.arange(n_e)
e = np.array([1250,1500])
n_e = e.size
ei = np.arange(n_e)
c_e = np.zeros(n_e) # Correlations
#c_e[0] = np.mean(img[mask]*g_r_e[mask])/norm_f
for i in ei:
    g_r_e = corr(img, e[i])
    c_e[i] = np.mean(img[mask]*g_r_e[mask])/norm_f # Autocorrelation
    print(c_e[i])
    e.tofile(efile, ",")
    c_e.tofile(ofile, ",")

