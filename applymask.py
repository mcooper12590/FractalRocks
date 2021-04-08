import imageio
from numpy import array
from tifffile import imsave
mfile = "Mask.tif"
mask = ~((imageio.volread(mfile)-1).astype(bool))
print(mask.shape)
ifile = "PZ101_InitialCropped.tif"
img = imageio.imread(ifile)
print(img.shape)
res = mask*img

imsave("Masked.tif", res)
