#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir, path, remove
from os.path import isfile, join
from PIL import Image
import cv2
import numpy as np
import pyheif
import io
import exifread

myPath = path.abspath('./Photos/original')
newPath = path.abspath('./Photos/new')
outputPath = path.abspath('./output')


def read_heic(path, filename):
    metadados = []
    if path.find('.heic') != -1:
        with open(path, 'rb') as file:
            newFile = myPath + '/' + filename.replace('.heic', '.jpg')
            i = pyheif.read_heif(io.BytesIO(file.read()))
            for metadata in i.metadata or []:
                if metadata['type'] == 'Exif':

                    # analisar EXIF metadata

                    metadados.append(metadados)
                    fstream = io.BytesIO((metadata['data'])[6:])

            # now just convert to jpeg

            s = io.BytesIO()
            pi = Image.frombytes(
                i.mode,
                i.size,
                i.data,
                'raw',
                i.mode,
                i.stride,
                )
            pi.save(newFile, format='jpeg')
            tags = exifread.process_file(fstream)
            remove(path)
        return newFile
        #return pi
    else:
        return path
        #return Image.open(path) 


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


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    w, h = Image.open(frames[0]).size
    frames = [cv2.imread(f,cv2.COLOR_RGB2BGR) for f in frames]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    writer.write(frames[0])
    im1 = frames.pop(0)
    i=0
    for frame in frames:
    	print(i)
    	i=i+1
    	#im2 = frame.convert(im1.mode)
    	#im2 = im2.resize(im1.size)
    	#diff = Image.blend(im1, im2, 0.0)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#diff = Image.blend(im1, im2, 0.2)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#diff = Image.blend(im1, im2, 0.4)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#diff = Image.blend(im1, im2, 0.6)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#diff = Image.blend(im1, im2, 0.8)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#diff = Image.blend(im1, im2, 1)
    	#writer.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))
    	#im1 = frame
    	writer.write(frame)
    writer.release() 

def slideshow(imagesTimed):
    FPS = 1  # Sets the FPS of the entire video
    currentFrame = 0  # The animation hasn't moved yet, so we're going to leave it as zero
    startFrame = 0  # The animation of the "next" image starts at "startFrame", at most
    trailingSeconds = 5  # Sets the amount of time we give our last image (in seconds)
    blendingDuration = 0.5  # Sets the amount of time that each transition should last for

                           # This could be more dynamic, but for now, a constant transition period is chosen

    blendingStart = 10  # Sets the time in which the image starts blending before songFile

    for i in imagesTimed:
        i[0] = i[0] * FPS  # Makes it so that iterating frame-by-frame will result in properly timed slideshows

    im1 = Image.open(imagesTimed[-1][1])  # Load the image in
    im2 = im1  # Define a second image to force a global variable to be created

    current = imagesTimed[-1][1]  # We're going to let the script know the location of the current image's location
    previous = current  # And this is to force/declare a global variable

    (height, width, layers) = np.array(im1).shape  # Get some stats on the image file to create the video with
    video = cv2.VideoWriter(outputPath + '/slideshow2.avi', -1, 60, (width, height),
                            True)

    while currentFrame < imagesTimed[0][0] + FPS * 60 * trailingSeconds:  # RHS defines the limit of the slideshow
        for i in imagesTimed:  # Loop through each image timing
            if currentFrame >= i[0] - blendingStart * FPS:  # If the image timing happens to be for the

                                                             # current image, the continue on...
                                                             # (Notice how imagesTimed is reversed)

                # The print statement adds some verbosity to the program

                print(str(currentFrame) + ' - ' + str(imagesTimed[0][0] + FPS * 60 * trailingSeconds) + ' - ' + i[1])
                if not current == i[1]:  # Check if the image file has changed
                    previous = current  # We'd want the transition to start if the file has changed
                    current = i[1]
                    startFrame = i[0] - blendingStart * FPS

                    # The two images in question for the blending is loaded in

                    im1 = Image.open(previous)
                    im2 = Image.open(current)
                break

        # See: http://blog.extramaster.net/2015/07/python-pil-to-mp4.html for the part below
        im2 = im2.convert(im1.mode)
        im2 = im2.resize(im1.size)
        diff = Image.blend(im1, im2, min(1.0, (currentFrame
                           - startFrame) / float(FPS)
                           / blendingDuration))
        video.write(cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2BGR))

        currentFrame += 1  # Next frame

    # At this point, we'll assume that the slideshow has completed generating, and we want to close everything off to prevent a corrupted output.

    video.release()


images = decodeHEIF(myPath)
imagesTimed = associateTimeToImage(images)

slideshow(imagesTimed)

#write_video(outputPath+"/out.mp4",images,0.1)