import numpy as np
import cv2
import matplotlib.pyplot as plt
from equal_his import equal_hist


# Read input
img = cv2.imread("img.jpg")


# Equalization Histogram
# equal_img = equal_hist(img)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal by Morphology
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)


# Finding sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0


# Watershed
watershed = cv2.watershed(img,markers)
watershed = watershed.astype(np.uint8)


# Canny Edge
canny = cv2.Canny((watershed+1), 10, 20)


# Draw Canny Edge to original image
img[canny == 255] = 255


# Show putput
plt.imshow(img, cmap='gray')
plt.show()