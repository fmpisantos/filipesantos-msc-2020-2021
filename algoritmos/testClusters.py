import algoritmos.utils as util
from os import path, mkdir
import numpy as np

outputPath = path.abspath("./output")


def loadArrays():
    return util.loadFromFile(f"{outputPath}/objects.json"), util.loadFromFile(f"{outputPath}/labels.json"), util.loadFromFile(f"{outputPath}/imagesArray.json")


def prepareData(featuresLabels, featureString, images):
    arrlength = len(featuresLabels)
    preparedImages = list()
    for img in images:
        tempArr = [0.0]*arrlength
        for labelAndScore in img[featureString]:
            label = labelAndScore.split(" - ")
            score = round(float(label[1]))
            label = label[0]
            tempArr[featuresLabels.index(label)] = score
        preparedImages.append(tempArr)
    return preparedImages


def prepareDataCombined(objects, labels, images):
    arrlength = len(objects) + len(labels)
    global preparedImages
    preparedImages = list()
    for img in images:
        tempArr = [0.0]*arrlength
        # for labelAndScore in img["objects-score"][:min(5, len(img["objects-score"]))]:
        for labelAndScore in img["objects-score"][:len(img["objects-score"])]:
            label = labelAndScore.split(" - ")
            score = round(float(label[1]))
            label = label[0]
            tempArr[objects.index(label)] = score
        # for labelAndScore in img["labels-score"][:min(5, len(img["labels-score"]))]:
        for labelAndScore in img["labels-score"][:len(img["labels-score"])]:
            label = labelAndScore.split(" - ")
            score = round(float(label[1]))
            label = label[0]
            tempArr[labels.index(label)+len(objects)] = score
        preparedImages.append(tempArr)
    # sum, avg = sumColumn(preparedImages)
    # deleteLowFrequencyCollumns(sum, preparedImages, avg)
    return preparedImages


def deleteLowFrequencyCollumns(sum, prepared, avg):
    for idx, s in enumerate(sum):
        if s < avg:
            np.delete(prepared, idx, axis=1)


def sumColumn(matrix):
    sum = np.sum(matrix, axis=0)
    return sum, np.mean(sum)


def MeanShift(preparedData, images):
    from sklearn.cluster import MeanShift
    import numpy as np
    X = np.array(preparedData)
    clustering = MeanShift().fit(X)
    labelSet = set()
    for idx, label in enumerate(clustering.labels_):
        labelSet.add(str(label))
        images[idx]["group"] = str(label)
        images[idx]["idx"] = idx
    util.writeToFile(images, f"{outputPath}/meanshift.json")
    print(clustering.labels_)
    return labelSet


def DBScan(preparedData, images):
    from sklearn.cluster import DBSCAN
    import numpy as np
    X = np.array(preparedData)
    clustering = DBSCAN(eps=1, min_samples=2).fit(X)
    labelSet = set()
    for idx, label in enumerate(clustering.labels_):
        labelSet.add(str(label))
        images[idx]["group"] = str(label)
        images[idx]["idx"] = idx
    util.writeToFile(images, f"{outputPath}/dbscan.json")
    print(clustering.labels_)
    return labelSet


def Optics(preparedData, images):
    from sklearn.cluster import OPTICS
    import numpy as np
    X = np.array(preparedData)
    global clustering
    clustering = OPTICS(min_samples=2, algorithm="kd_tree").fit(X)
    labelSet = set()
    for idx, label in enumerate(clustering.labels_):
        labelSet.add(str(label))
        images[idx]["group"] = str(label)
        images[idx]["idx"] = idx
    return labelSet


def groupByLabel(images, labelSet):
    groups = list()
    for label in labelSet:
        tmpArr = list()
        for img in images:
            if img["group"] == label:
                i = img
                i['features'] = np.concatenate((img["objects-score"], img["labels-score"]))
                tmpArr.append(i)
                # tmpArr.append({"image_id": img["image_id"], "brisque": img["brisque"], "aesthetic": img["aesthetic"], "idx": img["idx"], "path": img["path"], "group": img["group"],
                #               "features": np.concatenate((img["objects-score"], img["labels-score"])), "objects-score": img["objects-score"], "labels-score": img["labels-score"]})
        groups.append(tmpArr)
    return groups


def fromGroupToHtml(groups, path):
    from xml.etree import ElementTree as ET
    html = ET.Element('html')
    body = ET.Element('body')
    html.append(body)
    body.append(ET.Element('div', attrib={
                "id": "imageInfo", "style": "position:fixed;top:0;width:100%;z-index:100;height: 15%;overflow-x: hidden;overflow-y: auto;"}))
    div = ET.Element('div', attrib={
                     "style": "margin-top: 20%;height: 75%;overflow-x: hidden;overflow-y: auto;"})
    body.append(div)
    for group in groups:
        p = ET.Element('p')
        p.text = f"Group {group[0]['group']}:"
        div.append(p)
        p = ET.Element('p')
        tempIndex = 0
        for img in group:
            img["id"] = tempIndex
            i = ET.Element('img', attrib={'src': img["path"].replace("/mnt/d","d:"), "width": "200", "height": "200", "title": "Smile more",
                             "onclick": f"document.getElementById('imageInfo').innerHTML = {np.array2string(img['features'],separator = ' , ')}"})
            #  {img['brisque']} {img['aesthetic']}"})
            #np.array2string(img['features'],separator = ' , ')
            p.append(i)
            tempIndex += 1
        div.append(p)
    util.makeDirs("/".join(path.split("/")[:-1]))
    ET.ElementTree(html).write(path, encoding='unicode',
                               method='html')
    return groups


def SVM(groups, X, y, test, tmpGroup):
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X, y)
    labels = clf.predict(test)
    for idx, lbl in enumerate(labels):
        for group in groups:
            if group[0]["group"] == lbl:
                group.append(tmpGroup[idx])
    return groups


def treatFails(groups):
    train_x = list()
    train_y = list()
    test = list()
    tmpGroup = list()
    # Preparing data to new cluster
    i = False
    for idx, group in enumerate(groups):
        for img in group:
            if img["group"] == "-1":
                i = idx
                test.append(preparedImages[img["idx"]])
                tmpGroup.append(img)
                # group.remove(img)
            else:
                train_x.append(preparedImages[img["idx"]])
                train_y.append(img["group"])
    groups.remove(groups[i])
    SVM(groups, train_x, train_y, test, tmpGroup)
    return groups


def organizeByLabelsAndObjects(objects, labels, images):
    # keysToExclude = ['img']
    # for img in images:
    #     for key in img:
    #         if key in keysToExclude:
    #             img[key] = ""
    #         elif isinstance(img[key], np.ndarray):
    #             img[key] = img[key].tolist()
    # util.writeToFile(images, f"{outputPath}/imagesArray.json")
    return fromGroupToHtml(treatFails(fromGroupToHtml(groupByLabel(images, Optics(
        prepareDataCombined(list(objects), list(labels), images), images)),f"{outputPath}/optics.html")), f"{outputPath}/opticsAndSVM.html")


def testOrganizationByLabelsAndObjects():
    objects, labels, images = loadArrays()
    return fromGroupToHtml(treatFails(groupByLabel(images, Optics(
        prepareDataCombined(objects, labels, images), images))), f"{outputPath}/optics.html")


# objects, labels, images = loadArrays()
# util.writeToFile(groupByLabel(images,MeanShift(prepareData(objects,"objects",images),images)), f"{outputPath}/meanshift.json")
# util.writeToFile(groupByLabel(images,DBScan(prepareData(objects,"objects",images),images)), f"{outputPath}/dbscan.json")
# util.writeToFile(groupByLabel(images,Optics(prepareData(labels,"labels",images),images)), f"{outputPath}/optics.json")
# fromGroupToHtml(groupByLabel(images,Optics(prepareDataCombined(objects,labels,images),images)),f"{outputPath}/optics-old.html")
# fromGroupToHtml(treatFails(groupByLabel(images, Optics(prepareDataCombined(objects, labels, images), images))), f"{outputPath}/optics.html")
