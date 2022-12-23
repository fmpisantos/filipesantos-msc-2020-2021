from PIL import Image, ImageFile
import imagehash
from os import listdir, path
from os.path import isfile, join
from multiprocessing import Pool
import numpy as np
from functools import partial
import algoritmos.utils as util
ImageFile.LOAD_TRUNCATED_IMAGES = True

hashfunc = imagehash.phash
hashfunc2 = imagehash.average_hash


def getSimilarityScore(hash1, hash2):
    return abs(hash1 - hash2)


def _calculateHashs(image):
    hash1,hash2 = calculateHashs(image["path"])
    image["hash"] = hash1
    image["hash2"] = hash2
    return image

def calculateHashs(image):
    img = Image.open(image)
    return(hashfunc(img), hashfunc2(img))


def loadImages(testpath, method):
    return [{"path": f"{testpath}/{f}", "hash": method(Image.open(f"{testpath}/{f}")), "id": i} for i, f in enumerate(listdir(testpath)) if isfile(join(testpath, f))]


def _loadImages(testpath, method, method2):
    return [{"path": f"{testpath}/{f}", "hash": method(Image.open(f"{testpath}/{f}")), "hash2": method2(Image.open(f"{testpath}/{f}")), "id": i} for i, f in enumerate(listdir(testpath)) if isfile(join(testpath, f))]


def _checkSimilairs(images, similarityMax, image):
    ret = list()
    for img in images[image["id"]+1:]:
        if (getSimilarityScore(image['hash'], img['hash']) + getSimilarityScore(image['hash2'], img['hash2']))/2 < similarityMax:
            ret.append(img)
    if len(ret) > 0:
        ret.append(image)
    return ret


def checkSimilairs(image):
    ret = list()
    for img in images[image["id"]+1:]:
        if getSimilarityScore(image['hash'], img['hash']) < similarityMax:
            ret.append(img)
    if len(ret) > 0:
        ret.append(image)
    return ret


def clearGroups(images, id = "idx"):
    ret = list()
    flag = False
    for idx, group in enumerate(images):
        for prevGroups in ret[:idx]:
            for img in group:
                # if img in prevGroups:
                if next((i[id] for i in prevGroups if img[id] == i[id]), -1) != -1:
                    prevGroups = prevGroups + group
                    flag = True
                    break
        if not flag:
            ret.append(group)
        flag = False
    return ret


def fromGroupToHtml(groups, path):
    from xml.etree import ElementTree as ET
    html = ET.Element('html')
    head = ET.Element('head')
    # <link rel="stylesheet" type="text/css" href="mystyles.css" media="screen" />
    style = ET.Element(
        'link', attrib={"rel": "stylesheet", "type": "text/css", "href": "style.css"})
    head.append(style)
    html.append(head)
    body = ET.Element('body')
    html.append(body)
    div = ET.Element('div', attrib={
                     "style": "overflow-x: hidden;overflow-y: auto;"})
    body.append(div)
    i = 0
    for group in groups:
        if len(group):
            p = ET.Element('p')
            p.text = f"Group {i}:"
            div.append(p)
            p = ET.Element('div')
            for img in group:
                container = ET.Element("div", attrib={"class": "container"})
                _img = ET.Element('img', attrib={'src': img["path"].replace(
                    "/mnt/d", "d:"), "class": "image"})
                overlay = ET.fromstring(
                    f"<div class='overlay'>{img['id']}</div>")
                container.append(_img)
                container.append(overlay)
                p.append(container)
            div.append(p)
            i += 1
    util.makeDirs("/".join(path.split("/")[:-1]))
    ET.ElementTree(html).write(path, encoding='unicode',
                               method='html')


def isInAnyGroup(image, groups):
    for group in groups:
        # if image in group:
        if next((i["idx"] for i in group if image["idx"] == i["idx"]), -1) != -1:
            return True
    return False


# def _clearGroups(ret, orderFunction):
#     tempList = list()
#     for image in images:, groups):
#     for group in groups:
#         # if image in group:
#         if next((i["idx"] for i in group if image["idx"] == i["idx"]), -1) != -1:
#             return True
#     return False


def removeIdenticalImages(ret, orderFunction, imagens):
    tempList = list()
    for image in imagens:
        if not isInAnyGroup(image, ret):
            tempList.append(image)
    for group in ret:
        if len(group) > 0:
            tempList.append(orderFunction(group)[0])
    return tempList


def _clearGroups(ret, orderFunction):
    tempList = list()
    for image in images:
        if isInAnyGroup(image, ret):
            tempList.append(image)
    for group in ret:
        if len(group) > 0:
            tempList.append(orderFunction(group)[0])
    return ret


def identifyIdentical(group, orderFunction, similarityScore, outputpath=f"./output/similarityTest.html"):
    output = path.abspath(outputpath)
    global images
    images = group
    global similarityMax
    similarityMax = similarityScore
    func = partial(_checkSimilairs, group)
    with Pool(6) as p:
        ret = p.map(func, group)
    ret = clearGroups(ret)
    fromGroupToHtml(ret, output)
    ret = _clearGroups(ret, orderFunction)
    return ret, len(ret)


if __name__ == "__main__":
    import time
    hashmethod = "crop-resistant"
    similarityMax = 15
    testpath = path.abspath("./Photos/original")
    output = path.abspath(f"./output/similarityTest({hashmethod}).html")

    if hashmethod == 'averagehash':
        hashfunc = imagehash.average_hash
    elif hashmethod == 'phash':
        hashfunc = imagehash.phash
    elif hashmethod == 'dhash':
        hashfunc = imagehash.dhash
    elif hashmethod == 'whash-haar':
        hashfunc = imagehash.whash
    elif hashmethod == 'whash-db4':
        hashfunc = lambda img: imagehash.whash(img, mode='db4')
    elif hashmethod == 'colorhash':
        hashfunc = imagehash.colorhash
    elif hashmethod == 'crop-resistant':
        hashfunc = imagehash.crop_resistant_hash
    if hashmethod == 'combine':
        hashfunc = imagehash.phash
        hashfunc2 = imagehash.average_hash
        images = _loadImages(testpath, hashfunc, hashfunc2)
        func = partial(_checkSimilairs, images, similarityMax)
    else:
        images = loadImages(testpath,hashfunc)
        func = partial(checkSimilairs)

    pairs = list()
    start = time.time()
    with Pool(6) as p:
        ret = p.map(func, images)
    if hashmethod != "combine":
        print(f"HashFunction = {hashmethod} ({time.time()-start}ms)")
    else:
        print(f"HashFunction = [phash & averagehash] ({time.time()-start}ms)")
    rret = list()
    for l in ret:
        if len(l) > 0:
            rret.append(l)
    ret = clearGroups(rret, "id")
    print(output)
    fromGroupToHtml(ret, output)
