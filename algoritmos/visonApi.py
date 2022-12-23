from algoritmos.mythread import myThread, runParallel, runThreadedLoop
from google.cloud import vision
from multiprocessing import Pool


def getImageContent(path, client, getLabels=True, getObjects=True):
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations
    objSet = set()
    objs = []
    for o in objects:
        if o.name not in objSet:
            objs.append(f"{o.name} - {o.score}")
            objSet.add(o.name)
    labels = client.label_detection(image=image).label_annotations
    lblSet = set()
    lbls = []
    for l in labels:
        if l.description not in lblSet:
            lbls.append(f"{l.description} - {l.score}")
            lblSet.add(l.description)
    return objSet, objs, lblSet, lbls
    # return list(set([o.name for o in objects])), list(set([label.description for label in labels]))


def threadImageAnotation(img):
    client = vision.ImageAnnotatorClient()
    img["objects"], img["objects-score"], img["labels"], img["labels-score"] = getImageContent(
        f"{img['path']}", client)
    img["objects"] = list(set(img["objects"]))
    img["labels"] = list(set(img["labels"]))
    return img


def threadImageLabeling(img):
    client = vision.ImageAnnotatorClient()
    img["labels"], img["labels-score"] = getImageContent(
        f"{img['path']}", client, getObjects=False)
    img["labels"] = list(set(img["labels"]))
    return img


def threadImageObjects(img):
    client = vision.ImageAnnotatorClient()
    img["objects"], img["objects-score"] = getImageContent(
        f"{img['path']}", client, getLabels=False)
    img["objects"] = list(set(img["objects"]))
    return img


def _defineImageContent(path, images, getLabels=True, getObjects=True):
    j = len(images)
    labels = set()
    objects = set()
    if getLabels:
        if getObjects:
            images = runParallel(threadImageAnotation,images)
            for img in images:
                labels = labels.union(set(img["labels"]))
                objects = objects.union(set(img["objects"]))
            return objects, labels, images
        else:
            images = runParallel(threadImageLabeling,images)
            for img in images:
                labels = labels.union(set(img["labels"]))
            return list(), labels, images
    else:
        images = runParallel(threadImageObjects,images)
        for img in images:
            objects = objects.union(set(img["objects"]))
        return objects, list(), images
    

def defineImageContent(path, images, getLabels=True, getObjects=True):
    j = len(images)
    labels = set()
    objects = set()
    ret = list()
    if getLabels:
        if getObjects:
            with Pool(6) as p:
                ret = p.map(threadImageAnotation, images)
            for img in ret:
                labels = labels.union(set(img["labels"]))
                objects = objects.union(set(img["objects"]))
            return objects, labels, ret
        else:
            with Pool(6) as p:
                ret = p.map(threadImageLabeling, images)
            for img in ret:
                labels = labels.union(set(img["labels"]))
            return list(), labels, ret
    else:
        with Pool(6) as p:
            ret = p.map(threadImageObjects, images)
        for img in ret:
            objects = objects.union(set(img["objects"]))
        return objects, list(), ret


def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations
    return list(set([o.name for o in objects]))

    '''print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))'''


def identifyObjects(path, images):
    for img in images:
        img["objects"] = localize_objects(path + "/" + img["image_id"])


def detect_labels(path):
    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    return list(set([label.description for label in labels]))
    """for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))"""


def defineLabels(path, images):
    for img in images:
        img["labels"] = detect_labels(path + "/" + img["image_id"])

# localize_objects(path.abspath("./Photos/original/img_0610.jpg"))
# localize_objects(path.abspath("./Photos/original/img_0610.jpg"))
