# https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import numpy as np
from skimage import data
from math import sqrt
from skimage.feature import blob_log, peak_local_max
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed
#############
# IMAGE
#############
img_stars = data.hubble_deep_field()[0:500, 0:500]
stars_gray = rgb2gray(img_stars)

#############
# BLOB DETECTION
#############
blobs_log = blob_log(stars_gray, min_sigma = 1, max_sigma=30, num_sigma=50, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
numrows = len(blobs_log)
print("Number of blobs counted : " ,numrows)

#############
# LABEL DETECTION
#############
stars_thresh = threshold_otsu(stars_gray)
#stars_binary = (stars_gauss > stars_thresh)
stars_binary = (stars_gray > stars_thresh)
distance = ndi.distance_transform_edt(stars_binary)
local_maxi = peak_local_max(distance, indices=False, labels=stars_binary)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=stars_binary)

props = regionprops(labels)
num_basins = len(props)
print("Number of basins counted : " ,num_basins)

#############
# FIGURES
#############
fig,  axs = plt.subplots(2,2)

axs[0,0].imshow(stars_gray, cmap=plt.cm.gray)
axs[0,0].set_title('Original')
axs[0,0].axis('off')

for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
    axs[0,1].add_patch(c)
axs[0,1].imshow(stars_gray, cmap=plt.cm.gray)
axs[0,1].set_title('LoG')
axs[0,1].axis('off')

axs[1,0].imshow(labels, cmap=plt.cm.gray)
axs[1,0].set_title('Watershed')
axs[1,0].axis('off')

axs[1,1].imshow(labels, cmap=plt.cm.gray)
axs[1,1].set_title('Watershed')
axs[1,1].axis('off')

plt.tight_layout()
plt.show()
