from algoritmos.mythread import myThread
from brisq.Python.libsvm.python import brisquequality as b
# import imquality.brisque as brisque
from multiprocessing import Pool
from os import listdir, path
from os.path import isfile, join
import algoritmos.SlideshowMaker as sl
import algoritmos.utils as util
import matplotlib.pyplot as plt
from algoritmos.mythread import myThread, runParallel
from functools import partial
import cv2


def multipleBrisque(imagesArray):
    brisq = BRISQUE()
    i = 0
    j = len(imagesArray)
    for im in imagesArray:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        im['brisque'] = brisq.get_score(im['frame'])
        i = i + 1
    print("\n")
    return imagesArray
    # print('Reference image: %s' % brisq.get_score(path))


def score(imgArr):
    i = 0
    j = len(imgArr)
    print()
    for im in imgArr:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        #im['brisque'] = b.test_measure_BRISQUE(im['path'])
        im['brisque'] = b.test_measure_BRISQUE(im['frame'], False)
        i = i + 1
    print("\n")
    return imgArr


def brisqueThread(images, j):
    loadedModel = b.loadModel()
    func = partial(_scoreSingle, model=loadedModel)
    return runParallel(func,images)

def _scoreSingle(img, model=None):
    img["brisque"] = b.test_measure_BRISQUE(sl.scaledResize(cv2.imread(img["path"], cv2.IMREAD_UNCHANGED), 25), False, model)
    return img

def scoreSingle(img, model=None):
    img['brisque'] = b.test_measure_BRISQUE(
        sl.scaledResize(img['frame'], 25), False, model)
    return img


def multipleBrisque2(imagesArray):
    brisq = BRISQUE()
    i = 0
    j = len(imagesArray)
    for im in imagesArray:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        im['brisque'] = b.test_measure_BRISQUE(
            im['frame'], brisq._scale_feature(brisq.get_feature(im['frame'])))
        i = i + 1
    print("\n")
    return imagesArray
    # print('Reference image: %s' % brisq.get_score(path))


def cvbrisque(imgArr):
    i = 0
    j = len(imgArr)
    for im in imgArr:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        #im['brisque'] = b.test_measure_BRISQUE(im['path'])
        try:
            im['brisque'] = brisque.score(im['img'])
        except:
            im['brisque'] = 0
        print(im['image_id'] + " " + str(im['brisque']))
        i = i + 1
    print("\n")
    return imgArr


def doMedia(arr, i):
    for idx, obj in enumerate(arr):
        arr[idx] = obj/i
    return arr


def testBrisque(outputPath):
    toTest = path.abspath("./Photos/TecnicalChangedByTheme")
    folders = ["cidade", "fotogrupo", "objecto", "paisagem", "retrato"]
    folderTypes = ["blur", "noise", "underExposure", "overExposure"]
    images = []
    for t in folderTypes:
        for folder in folders:
            p = "%s/%s/%s/" % (toTest, folder, t)
            for f in listdir(p):
                if isfile(join(p, f)):
                    images.append({
                        "path": "%s%s" % (p, f),
                        "image_id": f,
                        "type": t,
                        "group": folder
                    })
                    # print("%s%s"%(p,f))
    images, w, h = sl.loadCV2Img(images, True, 25)
    #images = brisqueThread(images)
    images = score(images)
    images = util.orderList(images, False, "brisque", "")
    util.writeToFile([{
        "brisque": key["brisque"],
        "image_id":key["image_id"],
        "imagePath": "file://" + key["path"],
        "type": key["type"],
        "group": key["group"],
        "intensity": key["image_id"].split(".jp")[0]
    }for key in images], outputPath+"/"+"brisque"+".json")
    medias = list()
    for tt in folderTypes:
        cidade = [0] * 6
        fotogrupo = [0] * 6
        objecto = [0] * 6
        paisagem = [0] * 6
        retrato = [0] * 6
        media = [0] * 6
        i = 0
        for t in [{
            "image_id": d["image_id"],
            "brisque": d["brisque"],
            "type": d["type"],
            "group": d["group"],
            "intensity": d["image_id"].split(".jp")[0]
        } for d in images if d["type"] == tt]:
            print(t)
            index = int(int(t["intensity"])/2)
            if t["group"] == folders[0]:
                i = i + 1
                cidade[index] = t["brisque"]
            if t["group"] == folders[1]:
                fotogrupo[index] = t["brisque"]
            if t["group"] == folders[2]:
                objecto[index] = t["brisque"]
            if t["group"] == folders[3]:
                paisagem[index] = t["brisque"]
            if t["group"] == folders[4]:
                retrato[index] = t["brisque"]
            media[index] = media[index] + t["brisque"]
        media = doMedia(media, i)
        medias.append({
            "media": media,
            "label": tt
        })
        c1, = plt.plot(cidade, label=folders[0])
        f1, = plt.plot(fotogrupo, label=folders[1])
        o1, = plt.plot(objecto, label=folders[2])
        p1, = plt.plot(paisagem, label=folders[3])
        r1, = plt.plot(retrato, label=folders[4])
        m1, = plt.plot(media, label="media")
        plt.legend(handles=[c1, f1, o1, p1, r1, m1])
        # plt.show()
        plt.savefig('%s.png' % (tt))
        plt.close("all")
    m0, = plt.plot(medias[0]["media"], label=medias[0]["label"])
    m1, = plt.plot(medias[1]["media"], label=medias[1]["label"])
    m2, = plt.plot(medias[2]["media"], label=medias[2]["label"])
    m3, = plt.plot(medias[3]["media"], label=medias[3]["label"])

    plt.legend(handles=[m0, m1, m2, m3])
    plt.savefig('%s.png' % ("medias"))
    plt.close("all")

# testBrisque(path.abspath("./Photos/TecnicalChangedByTheme/Result"))
