# https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import data
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
# Original image
plt.imshow(stars_gray, cmap=plt.cm.gray)

#############
# LABEL DETECTION
#############
stars_thresh = threshold_otsu(stars_gray)
#stars_binary = (stars_gauss > stars_thresh)
stars_binary = (stars_gray > stars_thresh)
distance = ndi.distance_transform_edt(stars_binary)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                        #    labels=stars_binary)
local_maxi = peak_local_max(distance, indices=False, labels=stars_binary)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=stars_binary)

props = regionprops(labels)
num_basins = len(props)
print("Number of basins counted : " ,num_basins)

plt.figure()
plt.imshow(labels, cmap=plt.cm.gray)

# Show images
plt.show()
