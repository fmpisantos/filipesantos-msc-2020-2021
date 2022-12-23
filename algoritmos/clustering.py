import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os, shutil, glob, os.path
from PIL import Image as pil_image

def compareImages(img1,img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1,des2)
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv.drawMatches(img1, kp1, 
    img2, kp2, matches[:20],None)
    
    final_img = cv.resize(final_img, (1000,650))
    
    # Show the final image
    cv.imshow("Matches", final_img)
    cv.waitKey(10000)


def describeWOrb(images):
    orb = cv.ORB_create()
    detector = cv.ORB()
    features = []
    for im in images:
        kp1, des1 = orb.detectAndCompute(im["frame"],None)
        im["des"] = des1
        features.append(des1)
    return images

def flannScore(des1,des2):
    # FLANN parameters
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=12,
                        key_size=12,
                        multi_probe_level=2)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.knnMatch(des1, des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    similary = len(good) / len(matches)
    return similary

def groupImages(images,threshold):
    images = describeWOrb(images)
    groups = []
    for im in images:
        if len(groups) == 0:
            groups.append([im])
        else:
            inserted = 0
            for group in groups:
                for image in group:
                    avscore = 0
                    avscore = avscore + flannScore(im["des"],image["des"])
                if (avscore/ len(group) > threshold):
                    group.append(im)
                    inserted = inserted + 1
            if inserted == 0:
                groups.append([im])
    return groups

def kerasClustering(images, path, output): 
    image.LOAD_TRUNCATED_IMAGES = True 
    model = VGG19(weights='imagenet', include_top=False)

    # Variables
    imdir = path
    targetdir = output
    number_clusters = 5

    # Loop over files and get features
    #filelist = glob.glob(os.path.join(imdir, '*.jpg'))
    #filelist.sort()
    featurelist = []
    for i, im in enumerate(images):
    # for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" %(i, len(images)), end="\r")
        #img = image.load_img(imagepath, target_size=(224, 224))
        img = im["img"]
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())

    # Clustering
    clustered = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

    #clustered = DBSCAN(eps=0.5,metric='euclidean', min_samples=2).fit(np.array(featurelist))

    # Copy images renamed by cluster 
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(clustered.labels_):
        print("    Copy: %s / %s" %(i, len(clustered.labels_)), end="\r")
        shutil.copy(path + '/' + images[i]["image_id"], targetdir + "/" + str(m) + "_" + str(i) + ".jpg")

def groupImagesByLabels(images, labels):
    print("Grouping Images By Labels")
    
