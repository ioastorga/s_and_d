# https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

from matplotlib import pyplot as plt
from math import sqrt
import imageio
import numpy as np
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu

#############
# IMAGE
#############

img_stars = data.hubble_deep_field()[0:500, 0:500]
stars_gray = rgb2gray(img_stars)
# Original image
plt.imshow(stars_gray, cmap=plt.cm.gray)

#############
# BLOB DETECTION
#############

blobs_log = blob_log(stars_gray, min_sigma = 1, max_sigma=30, num_sigma=50, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
numrows = len(blobs_log)
print("Number of blobs counted : " ,numrows)
# Show image with stars detected
fig, ax = plt.subplots(1,1)
plt.imshow(stars_gray, cmap = plt.cm.gray)
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
    ax.add_patch(c)

#############
# LABEL DETECTION
#############

stars_gauss = gaussian(stars_gray)
#thresh = threshold_otsu(stars_gauss)
#stars_thresh = stars_gauss > thresh


thresholds = threshold_multiotsu(stars_gauss, classes=2)
stars_thresh = np.digitize(stars_gray, bins=thresholds)

label_img = label(stars_thresh)
regions = regionprops(label_img)
numregions = len(regions)
print("Number of regions counted : " ,numregions)

fig_label, ax_label = plt.subplots(1,1)
plt.imshow(stars_thresh, cmap = plt.cm.gray)

for props in regions:
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax_label.plot(bx, by, '-g', linewidth=0.5)


# Show images
plt.show()
