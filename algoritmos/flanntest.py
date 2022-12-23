from __future__ import print_function
import cv2 as cv
import numpy as numpy
import matplotlib.pyplot as plt
import time

img1_path = './Photos/original/img_0605.jpg' # queryImage
img2_path = './Photos/original/img_0584.jpg' # trainImage
img1 = cv.imread(img1_path,cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread(img2_path,cv.IMREAD_GRAYSCALE) # trainImage

orb = cv.ORB_create(500)
start=time.time()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

'''# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

print(matches[1])
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()'''
# FLANN parameters
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=12,
                    key_size=12,
                    multi_probe_level=2)
search_params = dict(checks=100)  # or pass empty dictionary
flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
matches = flann_matcher.knnMatch(des1, des2, k=2)
good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
print(len(good))
print(len(matches))
similary = len(good) / len(matches)
print(similary)

