import sys
import threading
from algoritmos.testClusters import organizeByLabelsAndObjects, testOrganizationByLabelsAndObjects
import algoritmos.SlideshowMaker as sl
from os import path
import os
import algoritmos.nima as nima
import argparse
from algoritmos.myresize import resize2
import algoritmos.utils as util
import time
import algoritmos.visonApi as visonApi
import numpy as np
import algoritmos.brisque2 as bq
from multiprocessing import Pool
import warnings
from algoritmos.similarityTest import identifyIdentical, _checkSimilairs, clearGroups, fromGroupToHtml, removeIdenticalImages, _calculateHashs
from functools import partial
from algoritmos.mythread import myThread, runParallel
import cv2
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        from datetime import datetime

        now = datetime.now()  # current date and time
        logpath = path.abspath("./logs/")
        util.makeDirs(logpath)
        logpath = path.abspath(
            f"./logs/LOG_{now.strftime('%m_%d_%Y_%H_%M_%S')}.log")
        self.log = open(
            f"{logpath}", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()
# Variables
# kerasClustering
# Paths:

# original = path.abspath("./Photos/500Photos")
original = path.abspath("./Photos/original")
# original = path.abspath("./Photos/SkyDive")
outputPath = path.abspath("./output")
tecnicalChanged = path.abspath("./Photos/TecnicalChanged")
kerasOutput = path.abspath("./outputPath/")
orbOutput = path.abspath("./orboutput/")
brisqueTest = path.abspath("./Photos/TecnicalChangedByTheme")

# Slideshow variables:
# fps = frames per second
# tF = transition frames - number must be an int > 0
# imgF = image frames - number must be an int > 0
# imgSec = seconds per image
# tSec = seconds per image transiction

fps = 10
imgSec = 1
tSec = 0.5
nImages = 15
models = ['mobilenet_aesthetic', 'mobilenet_technical']
percent = 0.5

def _loadImages(path, runNima, runHash, create=True):
    startT = time.time()
    # Get images path & generate slideshow information
    if create:
        imgF, tF, images = util.createSlideShow(True, path)
    # Get images
    j = len(images)
    if runHash:
        images = runParallel(_calculateHashs,images)
    print(f"Time per image Load ({str((time.time()-startT)/ j)})")
    print("\n")
    return imgF, tF, images, j


def _brisque(images, j):
    print("Get brisque score")
    start = time.time()
    images = bq.brisqueThread(images,j)
    end = time.time()-start
    print(f"Brisque eval per image ({str(end/j)})")
    print("\n")
    return images


def _calculateNima(image, string="aesthetic", model=models[0]):
    image[string] = 100 - (10 * nima.getImageQuality(nima.preProcessImage(image["path"]), model))
    return image

def nimaThreaded(images, j, string="aesthetic", model=models[0]):
    print("Get nima score")
    return runParallel(_calculateNima,images)


def _nima(images):
    print("Get nima score")
    with Pool(6) as p:
        ret = p.map(_calculateNima, images)
    return ret


def _orderByQuality(i, quality1, quality2=False, percent=percent):
    ret = util.orderList(i, quality2 != False,
                         quality1, quality2, percent)
    ret.reverse()
    return ret


def _nima(images, j, string="aesthetic", model=models[0]):
    print("Get nima score")
    with Pool(6) as p:
        ret = p.map(_calculateNima, images)
    return ret


def _imageLabelingAndClustering(images, path, j, writeToFile=False, getLabels=True, getObjects=True):
    print("Vision API labeling")
    start = time.time()
    objects, labels, labeledImages = visonApi._defineImageContent(
        path, images, getLabels=getLabels, getObjects=getObjects)
    if writeToFile:
        util.writeToFile(list(objects), f"{outputPath}/objects.json")
        util.writeToFile(list(labels), f"{outputPath}/labels.json")
    end = time.time()-start
    print(f"Image content identification per image ({str(end/j)})")
    print("\n")
    print("Clustering images based on their content")
    start = time.time()
    groups = organizeByLabelsAndObjects(objects, labels, labeledImages)
    end = time.time()-start
    print(f"Time per image grouping ({str(end/j)})")
    print("\n")
    return groups


def _getTotalNumberofFrames(tF, imgF, nImages):
    return nImages * (tF+imgF)-(tF*2)


def clearEmptyRowInMatrix(matrix):
    return [arr for arr in matrix if len(arr) > 0]


def _selectImagesToDisplay(groups, quality1, quality2, percent, ratio):
    images = list()
    groupQuality = list()
    groups = clearEmptyRowInMatrix(groups)
    for group in groups:
        tmp = round(len(groups) * ratio) + 1
        images = np.concatenate((images, group[:tmp]))
        if quality2 == "aesthetic" or quality1 == "aesthetic":
            groupQuality.append({'group': group[0]['group'], 'quality': sum(
                c['aesthetic'] for c in group[:tmp])/len(group[:tmp])})
    images = _orderByQuality(images, quality1, quality2, percent)
    if quality2 == "aesthetic" or quality1 == "aesthetic":
        groupQuality = sorted(
            groupQuality, key=lambda x: x['quality'], reverse=True)
        returnList = list()
        for group in groupQuality:
            for img in images:
                if img["group"] == group["group"]:
                    returnList.append(img)
        return returnList
    return images


def _removeIndenticalPhotos(images, orderFunction, similarityScore, outputPath):
    output = path.abspath(outputPath)
    ret = list()
    for i in images:
        ret.append(_checkSimilairs(images,similarityScore, i))
    rret = list()
    for l in ret:
        if len(l) > 0:
            rret.append(l)
    ret = clearGroups(rret)
    fromGroupToHtml(ret, output)
    ret = removeIdenticalImages(ret, orderFunction, images)
    return ret, len(ret)


def removeIdenticalPhotos(images, nImages, orderFunction):
    startT = time.time()
    print("Similar photos identification")
    for i in range(15, 0, -5):
        tempRet = list()
        tempRet, lenght = _removeIndenticalPhotos(
            images, orderFunction, i, f"./output/similarityTest({i}%similarity).html")
        print(
            f"Similar photos identification time ({str(time.time()-startT)})")
        if lenght > nImages:
            return tempRet, lenght
    print(f"Similar photos identification time ({str(time.time()-startT)})")
    return tempRet, -1


def _identyfyIndentical(groups, nImages, orderFunction):
    for i in range(15, 0, -5):
        tempRet = list()
        tempTotal = 0
        for group in groups:
            nGroup, lenght = identifyIdentical(
                group, orderFunction, i, outputpath=f"./output/similarityTest({i}%similarity).html")
            tempRet += nGroup
            tempTotal += lenght
        if tempTotal > nImages:
            return tempRet, tempTotal
    return tempRet, -1


def _getQualityScore(q1, q2, p):
    if q2:
        return(p * q1 + q2 * (1-p))
    else:
        return q1


def _generateSlideShow(images, totalNumberOfFrames, tF, imgF, quality1, quality2=False, percent=percent, fileName="output"):
    # Get images for slideshow
    #images, w, h = sl.loadCV2Img(images)
    w = 0
    h = 0
    startT = time.time()
    print("Calculating video width and height")
    for img in images:
        img["frame"] = cv2.imread(img["path"], cv2.IMREAD_UNCHANGED)
        h1, w1 = img["frame"].shape[:2]
        if w < w1:
            w = w1
        if h < h1:
            h = h1
    print(
        f"Time elapsed per slideshow width and height calculations ({str(time.time()/startT)})")
    print("\n")
    images = resize2(images, w, h)
    # Generate slideshow
    sl.write_video(outputPath + "/out.mp4",
                   images[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)
    util.writeToFile([{f"{quality1}{((' '+quality2) if quality2 != False else '')}": _getQualityScore(key[quality1], key[quality2], percent), "objects": key["objects-score"], "labels": key["labels-score"], "features": key["features"].tolist(), "image_id":key["image_id"],
                     "imagePath": "file://" + key["path"]}for key in images], outputPath+"/"+fileName+".json")
    print('Saving image quality eval to : ' +
          outputPath+"/"+fileName+".json")
    print('Saving .mp4 slideshow to : ' +
          outputPath + "/out.mp4")


if __name__ == "__main__":
    sys.setrecursionlimit(2097152)    # adjust numbers
    threading.stack_size(134217728)   # for your
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-fps', '--fps', dest='fps', default=fps,
                        help='Frames per sec (default = 15).', type=int,  required=False)
    parser.add_argument('-is', '--imgSec', dest='imgSec', default=imgSec,
                        help='Seconds per image (default = 1).', type=float, required=False)
    parser.add_argument('-ts', '--tSec', dest='tSec', default=tSec,
                        help='Seconds per transiction (default = 0.5).', type=float, required=False)
    parser.add_argument('-ni', '--nImage', dest='nImages', default=nImages,
                        help='Number of images to show (default = 15).', type=int, required=False)
    parser.add_argument('-qp', '--qualityPercent', dest='qualityPercent', default=percent,
                        help='Percentage associated with the tecnical quality of every photo (used in the balancing between technical and aesthetic quality) 0-1', type=float, required=False)
    parser.add_argument('-a', '--alg', action='append', dest='alg', default=[],
                        help='''Select the algoritms to use from the following list (note: this flag can be omitted to use the recomended algoritms): (default Runs all algoritms)
    "brisque" or "b" for tecnical photo assessment
    "nima" or "n" for aesthetic photo assessment
    "labels" or "l" for image label identification
    "objects" or "o" for objects identification
    "identical" or "ii" for identical photos identification
    "slideshow" or "s" to create a slideshow''', required=False)
    parser.add_argument('-p', '--path', dest='path', default=original,
                        help="Path to folder holding the photos (default = './Photos/original').", type=str, required=False)
    parser.add_argument('-out', '--output', dest='showOutput', default=False,
                        help='Boolean value that represents whether or not to generate debug outputs', type=bool, required=False)
    parser.add_argument('-wsl', '--wsl', dest='wsl', default=False,
                        help='To be True if program is beeing runned in wsl', type=bool, required=False)
    parser.print_help()
    args = parser.parse_args()
    fps = args.fps
    imgSec = args.imgSec
    tSec = args.tSec
    nImages = args.nImages
    algs = args.alg
    wsl = args.wsl
    _path = path.abspath(args.path)
    percent = args.qualityPercent
    showOutput = args.showOutput
    if(algs == None):
        algs = list()
    if len(algs) > 0:
        algs = algs[0].strip().split(" ")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./thesis-312720-c0663e5d4e21.json"
    runNima = "nima" in algs or "n" in algs or len(algs) == 0
    runBrisque = "brisque" in algs or "b" in algs or len(algs) == 0
    runLabels = "labels" in algs or "l" in algs or len(algs) == 0
    runObjects = "objects" in algs or "o" in algs or len(algs) == 0
    runSlideShow = "slideshow" in algs or "s" in algs or len(algs) == 0
    runIdenticalIdentification = "identical" in algs or "ii" in algs or len(
        algs) == 0
    print("\n")
    beginTime = time.time()
    totalPrintingOutputTime = 0
    imgF, tF, images, j = _loadImages(
        _path, runNima, runIdenticalIdentification)
    #imgF, tF, images, w, h, j = _loadImages(_path, runNima, runIdenticalIdentification)
    if runBrisque:
        images = _brisque(images, j)
    if runNima:
        start = time.time()
        # images = _nima(images, j)
        images = nimaThreaded(images, j)
        end = time.time() - start
        print(f"Nima eval per image ({str(end/j)})")
        print("\n")
    if runNima:
        if runBrisque:
            images = [{**img, "id": idx, "quality": percent * img["brisque"] +
                       img["aesthetic"] * (1-percent), "idx": idx} for idx, img in enumerate(images)]
            if showOutput:
                tstart = time.time()
                tempImages = [{"path": img["path"], "brisque": img["brisque"], "aesthetic":img["aesthetic"],
                               "quality": percent * img["brisque"] + img["aesthetic"] * (1-percent)} for img in images]
                tempImages = _orderByQuality(tempImages, "brisque")
                util.fromImagesArrayToHtml(tempImages, path.abspath(
                    "./output/brisque.html"), "Brisque eval", ["brisque"],wsl)
                tempImages = _orderByQuality(tempImages, "aesthetic")
                util.fromImagesArrayToHtml(tempImages, path.abspath(
                    "./output/nima.html"), "Nima eval", ["aesthetic"],wsl)
                totalPrintingOutputTime += time.time() - tstart
        else:
            images = [{**img, "quality": img["aesthetic"]} for img in images]
    elif runBrisque:
        images = [{**img, "quality": img["brisque"]} for img in images]
    images = _orderByQuality(images, "quality")
    images = [{**img, "id": idx, "idx": idx} for idx, img in enumerate(images)]
    if showOutput:
        tstart = time.time()
        util.fromImagesArrayToHtml(images, path.abspath(
            "./output/quality.html"), "Quality order", ["quality"],wsl)
        totalPrintingOutputTime += time.time() - tstart
    #images, total = _identyfyIndentical([images], nImages, lambda g: _orderByQuality(g, "quality", False, percent))
    images, total = removeIdenticalPhotos(
        images, nImages, lambda g: _orderByQuality(g, "quality", False, percent))

    images = _orderByQuality(images, "quality")
    if showOutput:
        tstart = time.time()
        util.fromImagesArrayToHtml(images, path.abspath(
            "./output/afterremoval.html"), "Quality order", ["quality"],wsl)
        totalPrintingOutputTime += time.time() - tstart

    if runObjects or runLabels:
        groups = _imageLabelingAndClustering(
            images, _path, j, getLabels=runLabels, getObjects=runObjects)
    # groups = testOrganizationByLabelsAndObjects()
    totalNumberOfFrames = _getTotalNumberofFrames(tF, imgF, nImages)
    if runIdenticalIdentification:
        if total != -1:
            j = total
    images = _selectImagesToDisplay(
        groups, "quality", False, percent, nImages/j)

    # images = _selectImagesToDisplay(
    #     groups, "brisque", "aesthetic", percent, nImages/j)
    if showOutput:
        startT = time.time()
        util.fromImagesArrayToHtml(images, path.abspath(
            "./output/selected.html"), "Selected images", ["image_id"],wsl)
        totalPrintingOutputTime += time.time() - tstart
        print(
            f"Total time creating and saving output for degug ({totalPrintingOutputTime})")
    if runSlideShow:
        _generateSlideShow(images, totalNumberOfFrames, tF,
                           imgF, "brisque", "aesthetic", percent)
    print(f"Total time ({time.time()-beginTime})")
