#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir, path, remove
from os.path import isfile, join
from PIL import Image
import cv2
import numpy as np
# import pyheif
import io
# import exifread

myPath = path.abspath('./Photos/original')
newPath = path.abspath('./Photos/new')
outputPath = path.abspath('./output')


def read_heic(path, filename):
    metadados = []
    # if path.find('.heic') != -1:
    #     with open(path, 'rb') as file:
    #         newFile = myPath + '/' + filename.replace('.heic', '.jpg')
    #         i = pyheif.read_heif(io.BytesIO(file.read()))
    #         for metadata in i.metadata or []:
    #             if metadata['type'] == 'Exif':
    #
    #                 # analisar EXIF metadata
    #
    #                 metadados.append(metadados)
    #                 fstream = io.BytesIO((metadata['data'])[6:])
    #
    #         # now just convert to jpeg
    #
    #         s = io.BytesIO()
    #         pi = Image.frombytes(
    #             i.mode,
    #             i.size,
    #             i.data,
    #             'raw',
    #             i.mode,
    #             i.stride,
    #             )
    #         pi.save(newFile, format='jpeg')
    #         tags = exifread.process_file(fstream)
    #         remove(path)
    #     # return pi
    #     return newFile
    # else:
        # return Image.open(path)
    return path

def decodeHEIF(path):
    return [read_heic((myPath + '/' + f).lower(), f.lower()) for f in
            listdir(myPath) if isfile(join(myPath, f))]


def getPhotosLocation():
    return [myPath + '/' + f for f in listdir(myPath)
            if isfile(join(myPath, f))]


def associateTimeToImage(onlyfiles):
    ret = []
    for i in onlyfiles:
        ret.append([1, i])
    return ret

def checkImage(f,w,h):
    # frame = cv2.imread(f, cv2.COLOR_RGB2BGR)
    frame = Image.open(f)
    return [frame, f]

def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    (w, h, c) = cv2.imread(frames[0], cv2.COLOR_RGB2BGR).shape
    newFrames = []
    for f in frames:
        frame = cv2.imread(f, cv2.COLOR_RGB2BGR)
        (w1,h1,c1) = frame.shape
        if w1 > w:
            w = w1
        if h1 > h:
            h = h1
        newFrames.append([frame, f])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    print((w,h))
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    writer.write(newFrames[0][0])
    im1 = newFrames.pop(0)
    i = 1
    for frame in newFrames:
        print('Frame: (' + str(i) + ', ' + str(len(newFrames)+1) + ')' \
            + ' image: ' + frame[1])
        i = i + 1

        # im2 = frame.convert(im1.mode)
        # im2 = im2.resize(im1.size)
        # diff = Image.blend(im1, im2, 0.0)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # diff = Image.blend(im1, im2, 0.2)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # diff = Image.blend(im1, im2, 0.4)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # diff = Image.blend(im1, im2, 0.6)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # diff = Image.blend(im1, im2, 0.8)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # diff = Image.blend(im1, im2, 1)
        # writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
        # im1 = frame
        # for j in range(60):
        writer.write(frame[0])
            # cv2.imshow('video',frame[0])
    # cv2.destroyAllWindows()
    writer.release()


images = decodeHEIF(myPath)

# imagesTimed = associateTimeToImage(images)

write_video(outputPath + '/out.mp4', images, 1)
