# https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

from matplotlib import pyplot as plt
from math import sqrt
import imageio
import numpy as np
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops,find_contours
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import dilation, erosion
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
plt.figure()
#n, bins, patches = plt.hist(blobs_log[:, 2], 50)
#print("Number of sigma counted : " ,bins)
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

plt.figure()
n, bins, patches = plt.hist(blobs_log[:, 2])
#############
# LABEL DETECTION
#############

#dilate to avoid loosing small elements on edges
#image_dil = dilation(stars_gray, np.ones((5,5)))
#make the image smooth
#image_gauss = gaussian(stars_gray, sigma=1.5)
#automatic threshold
image_th = threshold_otsu(stars_gray)
#find contour
contours = find_contours(stars_gray, level = 0.1)

# Find contours at a constant value of 0.8
#contours = find_contours(stars_gray, 0.3)

# Display the image and plot all contours found
fig_c, ax_c = plt.subplots()
ax_c.imshow(stars_gray, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax_c.plot(contour[:, 1], contour[:, 0], linewidth=1)
print("Number of contours counted : " ,len(contours))
ax_c.axis('image')
ax_c.set_xticks([])
ax_c.set_yticks([])
plt.show()
