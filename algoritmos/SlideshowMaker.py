#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir, remove
from os.path import isfile, join
# from PIL import Image
import cv2
# import pyheif
# import io
from algoritmos.myresize import resize
import time
from algoritmos.similarityTest import calculateHashs

# Note: That the fps and tf(Transition frames) are fixed --- standardized for best performance vs quality
# input: {
#           sec: seconds per image
#        }
# return: {
#            imgF: frames per image
#         }


def generateFramesFromImgTime(img, t, fps):
    return int(img*fps), int(t*fps/2)


def read_heic(path, filename):
    metadados = []
    """if path.lower().find(".heic") != -1:
        print("Decoding .heic to .jpg (" + filename + ")", end="\r")
        with open(path, "rb") as file:
            newFile = path + "/" + filename.lower().replace(".heic", ".jpg")
            i = pyheif.read_heif(io.BytesIO(file.read()))
            for metadata in i.metadata or []:
                if metadata["type"] == "Exif":

                    # analisar EXIF metadata

                    metadados.append(metadados)
                    # fstream = io.BytesIO((metadata["data"])[6:])

            # now just convert to jpeg

            s = io.BytesIO()
            pi = Image.frombytes(
                i.mode,
                i.size,
                i.data,
                "raw",
                i.mode,
                i.stride,
            )
            pi.save(newFile, format="jpeg")
            # tags = exifread.process_file(fstream)
            remove(path)
        # return pi
        print("\n")
        return newFile
    else:
        # return Image.open(path)
        """
    return path


def decodeHEIF(path, script):
    print("Getting img path")
    if not script:
        return [
            read_heic((path + "/" + f), f)
            for f in listdir(path)
            if isfile(join(path, f))
        ]
    else:
        return [
            {"path": read_heic((path + "/" + f), f), "image_id": f}
            for f in listdir(path)
            if isfile(join(path, f)) and ":Zone.Identifier" not in f
        ]


def scaledResize(img, scale_percent=100):
    # scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def _loadCV2Img(f, runSimilarity=True):
    if ":Zone.Identifier" not in f:
        frame = cv2.imread(f["path"], cv2.IMREAD_UNCHANGED)
        if runSimilarity:
            hashes = calculateHashs(f["path"])
            f["hash"] = hashes[0]
            f["hash2"] = hashes[1]
        f["frame"] = frame
        h, w = frame.shape[:2]
        f["h"] = h
        f["w"] = w
    # print(f["path"])
    return f


def loadCV2Img(images, flag=False, perc=100, runSimilarity=True):
    print("From path to img")
    im1 = images[0]
    if flag:
        im1["frame"] = scaledResize(cv2.imread(
            im1["path"], cv2.IMREAD_UNCHANGED), perc)
    else:
        im1["frame"] = cv2.imread(im1["path"], cv2.IMREAD_UNCHANGED)
    h, w = im1["frame"].shape[:2]
    i = 0
    j = len(images)
    timeI = 0
    fullTime = 0
    for f in images:
        if ":Zone.Identifier" not in f:
            print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
            start = time.time()
            timeI = timeI + 1
            i = i + 1
            if flag:
                frame = scaledResize(cv2.imread(
                    f["path"], cv2.IMREAD_UNCHANGED), perc)
            else:
                frame = cv2.imread(f["path"], cv2.IMREAD_UNCHANGED)
            if runSimilarity:
                hashes = calculateHashs(f["path"])
                f["hash"] = hashes[0]
                f["hash2"] = hashes[1]
            h1, w1 = frame.shape[:2]
            if w1 > w:
                w = w1
            if h1 > h:
                h = h1
            # ret.append({"frame":frame, "image_id":((f.split("/original/")[1]).split(".")[0])})
            f["frame"] = frame
            fullTime = fullTime + time.time() - start
    print("\n")
    fullTime = fullTime / timeI
    print("Time per image Load = " + str(fullTime))
    return images, w, h


def fromPathToCV2Img(pathArr):
    print("From path to img")
    ret = []
    im1 = pathArr.pop(0)
    ret.append({"frame": cv2.imread(im1), "image_id": im1})
    h, w = ret[0]["frame"].shape[:2]
    i = 0
    j = len(pathArr)
    timeI = 0
    fullTime = 0
    for f in pathArr:
        if ":Zone.Identifier" not in f:
            print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
            start = time.time()
            timeI = timeI + 1
            i = i + 1
            frame = cv2.imread(f)
            h1, w1 = frame.shape[:2]
            if w1 > w:
                w = w1
            if h1 > h:
                h = h1
            # ret.append({"frame":frame, "image_id":((f.split("/original/")[1]).split(".")[0])})
            ret.append({"frame": frame, "image_id": f})
            fullTime = fullTime + time.time() - start

    print("\n")
    fullTime = fullTime / timeI
    print("Time per image Load = " + str(fullTime))
    return ret, w, h


def loadImagesToSys(file_path, frames, fps, tF, imgF, nImages):
    # file_path, frames, fps, tF, imgF
    # fps = frames per second
    # tF = transition frames - number must be an int > 0
    # imgF = image frames - number must be an int > 0

    # Loads img from url to cv2.image and returns the max width and height of the images
    # This (w,h) value is used to resize images when needed
    # If we implement more templates for slideshow the number of images that might need resizing can decrease
    frames, w, h = fromPathToCV2Img(frames)

    totalNumberOfFrames = nImages * (tF+imgF)-(tF*2)

    # Images resize (we use a filling resise so we dont ruin an image by extending it)
    frames = resize(frames, w, h)
    print("Finish image loading and resizing")
    return frames, w, h, totalNumberOfFrames


def write_video(file_path, frames, w, h, totalNumberOfFrames, fps, tF, imgF):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    # Saves 1st image, this needs to be before the loop so we get a reference image for the next transition
    im1 = frames.pop(0)
    # Used to calculate elapsed time
    timeI = 0
    imgTime = 0
    frameTime = 0
    timeF = 0

    start = time.time()
    for z in range(imgF - 1):
        startF = time.time()
        timeF = timeF + 1
        print("Frame: (" + str(z+1) + ", " +
              str(totalNumberOfFrames) + ")", end="\r")
        writer.write(im1["frame"])
        frameTime = frameTime + time.time() - startF
    imgTime = imgTime + time.time() - start
    timeI = timeI + 1

    # debuging index
    i = z
    for frame in frames:
        # debuging string
        print("Frame: (" + str(i+1) + ", " +
              str(totalNumberOfFrames) + ")", end="\r")
        im2 = frame["frame"]
        start = time.time()
        for y in range(tF):
            print("Frame: (" + str(i+1) + ", " +
                  str(totalNumberOfFrames) + ")", end="\r")
            startF = time.time()
            timeF = timeF + 1
            i = i + 1
            beta = 1.0 - y * 1 / tF
            dst = cv2.addWeighted(im2, 1.0 - beta, im1["frame"], beta, 0.0)
            writer.write(dst)
            frameTime = frameTime + time.time() - startF
        for z in range(imgF):
            print("Frame: (" + str(i+1) + ", " +
                  str(totalNumberOfFrames) + ")", end="\r")
            startF = time.time()
            timeF = timeF + 1
            i = i + 1
            writer.write(im2)
            frameTime = frameTime + time.time() - startF
        im1 = frame
        imgTime = imgTime + time.time() - start
        timeI = timeI + 1
    print("Time elapsed per Image print ("+str(imgTime/timeI)+")")
    print("Time elapsed per frame print ("+str(frameTime/timeF)+")")
    print("\n")
    writer.release()
