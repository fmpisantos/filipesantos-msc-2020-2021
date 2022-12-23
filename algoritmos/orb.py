import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import imagehash


def test():
    img1 = cv.imread('../Photos/original/img_0576.jpg',3)          # queryImage
    img2 = cv.imread('../Photos/original/img_0577.jpg',3) # trainImage

    b,g,r = cv.split(img1)           # get b, g, r
    img1 = cv.merge([r,g,b])     # switch it to r, g, 

    b,g,r = cv.split(img2)           # get b, g, r
    img2 = cv.merge([r,g,b])     # switch it to r, g, 

    method = 'ORB' # 'SIFT'
    lowe_ratio = 0.89
    magic_number = 0.75

    if method   == 'ORB':
        finder = cv.ORB_create()
    elif method == 'SIFT':
        finder = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    average = 0
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            print(abs(m.distance-n.distance))
            average += abs(m.distance-n.distance)
            good.append([m])

    average /= len(good)

    print(average)
    msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
    msg2 = 'there are %d good matches' % (len(good))

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img3,msg1,(10, 250), font, 0.5,(255,255,255),1,cv.LINE_AA)
    cv.putText(img3,msg2,(10, 270), font, 0.5,(255,255,255),1,cv.LINE_AA)
    fname = 'output_%s_%.2f.png' % (method, magic_number)
    cv.imwrite(fname, img3)

    plt.imshow(img3),plt.show()

img = Image.open('../Photos/original/img_0593.jpg')
image_one_hash = imagehash.whash(img)

img2 = Image.open('../Photos/original/img_0581.jpg')
image_two_hash = imagehash.whash(img2)

similarity = image_one_hash - image_two_hash
print(similarity)