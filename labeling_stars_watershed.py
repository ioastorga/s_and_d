# https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

from matplotlib import pyplot as plt
from math import sqrt
import imageio
import numpy as np
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops
from skimage.filters import gaussian, sobel
from skimage.segmentation import slic, join_segmentations,watershed
from scipy import ndimage as ndi
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
mask = np.zeros_like(stars_gray)
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
    ax.add_patch(c)
plt.imshow(mask, cmap = plt.cm.gray)

#############
# LABEL DETECTION
#############

# Make segmentation using edge-detection and watershed.
elevation = sobel(stars_gray)
plt.figure()
plt.imshow(elevation, cmap=plt.cm.gray)

# Identify some background and foreground pixels from the intensity values.
# Unsure region is labeled 0
markers = np.zeros_like(stars_gray)
markers[stars_gray < 30.0] = 1 #background
markers[stars_gray > 150.0] = 2 #foreground

stars_segmented = watershed(mask, markers)
#seg1 = label(ws == foreground)
#regions_seg = regionprops(seg1)

#fig_label, ax_label = plt.subplots(1,1)
#color1 = label2rgb(seg1, image=stars_gray, bg_label=0)
#ax_label.imshow(edges,cmap=plt.cm.nipy_spectral)
#ax_label.set_title('Sobel+Watershed')
plt.figure()
plt.imshow(stars_segmented, cmap=plt.cm.gray)
stars_segmented = ndi.binary_fill_holes(stars_segmented - 1)
labeled_stars_segmented, _ = ndi.label(stars_segmented)
image_label_overlay = label2rgb(labeled_stars_segmented, image=stars_gray, bg_label=0)

fig_s, axes_s = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes_s[0].imshow(stars_gray, cmap=plt.cm.gray)
axes_s[0].contour(stars_segmented, [0.5], linewidths=1.2, colors='y')
axes_s[1].imshow(image_label_overlay)

for a in axes_s:
    a.axis('off')

plt.tight_layout()




# Show images
plt.show()
