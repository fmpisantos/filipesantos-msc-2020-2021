import json
import skimage.io
import skimage.filters
import numpy as np
from PIL import Image
from skimage.util import random_noise
from skimage import exposure, img_as_ubyte
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
import sys

# import imquality.brisque as brisque
sys.path.insert(1, "/mnt/c/Users/filip/Documents/filipesantos-msc-2020-2021")
# from algoritmos.mythread import runParallel
from brisq.Python.libsvm.python import brisquequality
import algoritmos.nima as nima
from algoritmos.piqe import piqe
import os
import tensorflow.keras as keras

models = ["mobilenet_aesthetic", "mobilenet_technical"]
useModel = 0


def addBlurToImage(image, percent):
    blurred = skimage.filters.gaussian(
        image, sigma=(percent, percent), truncate=3.5, multichannel=True
    )

    blurred = np.array(255 * blurred, dtype="uint8")
    print("addBlurToImage")
    return blurred


def addNoiseToImage(image, percent):
    noise_img = random_noise(image, mode="s&p", amount=percent)
    noise_img = np.array(255 * noise_img, dtype="uint8")
    print("addNoiseToImage")
    return noise_img


def changeExposuteToImage(image, percent):
    gamma_corrected = exposure.adjust_gamma(image, percent, percent)
    gamma_corrected = np.array(255 * gamma_corrected, dtype="uint8")
    print("changeExposuteToImage")
    return gamma_corrected


def decrease_brightness(img, percent):
    gamma_corrected = exposure.adjust_gamma(img, percent)
    # gamma_corrected = np.array(255 * gamma_corrected, dtype="uint8")
    print("decreaseBrightnessToImage")
    return gamma_corrected


def increase_brightness(img, percent):
    gamma_corrected = exposure.adjust_gamma(img, percent)
    print("increaseBrightnessToImage")
    return gamma_corrected


def saveSkimage(fileName, image):
    skimage.io.imsave(fileName, image)


def scaledResize(img, scale_percent=100):
    # scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def addStringBeforeFileExtention(p, string):
    return (
        p.replace(".jpg", f"{string}.jpg")
        .replace(".jpeg", f"{string}.jpeg")
        .replace(".png", f"{string}.png")
        .replace(".gif", f"{string}.gif")
        .replace(".JPG", f"{string}.jpg")
        .replace(".JPEG", f"{string}.jpeg")
        .replace(".PNG", f"{string}.png")
        .replace(".GIF", f"{string}.gif")
    )


def prepare_image(img, size=(224, 224)):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize(size)
    img_array = image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(image_array_expanded)
nima.preProcessImage

def fromPathToMatrix(
    paths, intensities, saveImages=False, percent=25, usePiqe=False, useNima=False
):
    blur = list()
    noise = list()
    over = list()
    under = list()
    arr = list()
    # func = partial(_fromPathToMatrix, saveImages, percent)
    # runParallel(func, paths, arr)
    for p in paths:
        img = skimage.io.imread(fname=p)
        b = list()
        n = list()
        o = list()
        u = list()
        loadedModel = brisquequality.loadModel()
        # save diferent intensities of blur

        for i in intensities:
            temp = addBlurToImage(img, i / 10)
            cv2Im = img_as_ubyte(temp)
            resized = scaledResize(cv2Im, percent)
            if usePiqe:
                pi = piqe(resized)[0]
            if useNima:
                ni = 100 - (
                    10 * nima.getImageQuality(prepare_image(cv2Im), models[useModel])
                )
            brisq = abs(
                brisquequality.test_measure_BRISQUE(
                    resized,
                    False,
                    loadedModel,
                )
            )
            if brisq > 100:
                brisq -= 100
            # elif brisq < 0:
            #     brisq = abs(brisq)
            b.append(
                {
                    **({"img": temp} if saveImages else {}),
                    **{
                        "piqe": pi if usePiqe else {},
                        "nima": ni if useNima else {},
                        "brisque": brisq,
                        "name": addStringBeforeFileExtention(p, f"_blur_{i}"),
                    },
                }
            )
        blur.append(b)
        # save diferent intensities of noise
        for i in intensities:
            temp = addNoiseToImage(img, i / 100)
            cv2Im = img_as_ubyte(temp)
            resized = scaledResize(cv2Im, percent)
            if usePiqe:
                pi = piqe(resized)[0]
            if useNima:
                ni = 100 - (
                    10 * nima.getImageQuality(prepare_image(cv2Im), models[useModel])
                )
            brisq = abs(
                brisquequality.test_measure_BRISQUE(
                    resized,
                    False,
                    loadedModel,
                )
            )
            if brisq > 100:
                brisq -= 100
            # elif brisq < 0:
            #     brisq = abs(brisq)
            n.append(
                {
                    **({"img": temp} if saveImages else {}),
                    **{
                        "piqe": pi if usePiqe else {},
                        "nima": ni if useNima else {},
                        "brisque": brisq,
                        "name": addStringBeforeFileExtention(p, f"_noise_{i}"),
                    },
                }
            )
        noise.append(n)
        # save diferent intensities of overexposure
        for i in intensities:
            temp = increase_brightness(img, 1 - i / 100)
            cv2Im = img_as_ubyte(temp)
            resized = scaledResize(cv2Im, percent)
            if usePiqe:
                pi = piqe(resized)[0]
            if useNima:
                ni = 100 - (
                    10 * nima.getImageQuality(prepare_image(cv2Im), models[useModel])
                )
            brisq = abs(
                brisquequality.test_measure_BRISQUE(
                    resized,
                    False,
                    loadedModel,
                )
            )
            if brisq > 100:
                brisq -= 100
            # elif brisq < 0:
            #     brisq = abs(brisq)
            o.append(
                {
                    **({"img": temp} if saveImages else {}),
                    **{
                        "piqe": pi if usePiqe else {},
                        "nima": ni if useNima else {},
                        "brisque": brisq,
                        "name": addStringBeforeFileExtention(p, f"_overexposure_{i}"),
                    },
                }
            )
        over.append(o)
        # save diferent intensities of underexposure
        for i in intensities:
            temp = decrease_brightness(img, i / 10)
            cv2Im = img_as_ubyte(temp)
            resized = scaledResize(cv2Im, percent)
            if usePiqe:
                pi = piqe(resized)[0]
            if useNima:
                ni = 100 - (
                    10 * nima.getImageQuality(prepare_image(cv2Im), models[useModel])
                )
            brisq = abs(
                brisquequality.test_measure_BRISQUE(
                    resized,
                    False,
                    loadedModel,
                )
            )
            if brisq > 100:
                brisq -= 100
            # elif brisq < 0:
            #     brisq = abs(brisq)
            u.append(
                {
                    **({"img": temp} if saveImages else {}),
                    **{
                        "piqe": pi if usePiqe else {},
                        "nima": ni if useNima else {},
                        "brisque": brisq,
                        "name": addStringBeforeFileExtention(p, f"_underexposure_{i}"),
                    },
                }
            )
        under.append(u)
    return blur, noise, over, under


def saveImagesFromMatrix(matrix):
    for row in matrix:
        for img in row:
            saveSkimage(img["name"], img["img"])
            img.pop("img")
    return matrix


def plotMatrix(matrix, path, name, intensities):
    ret = list()
    for i in range(len(intensities)):
        ret.append(0)
    for imgs in matrix:
        for i in range(len(intensities)):
            ret[i] += imgs[i][name]
            temp = list()
            for idx, img in enumerate(imgs):
                if idx == 0:
                    temp.append(0)
                else:
                    temp.append(abs(imgs[0][name] - img[name]))
            # plt.plot(intensities, [img[name] for img in imgs])
            plt.plot(intensities, temp)
    # plt.show()
    plt.savefig(path)
    plt.close("all")
    l = len(matrix)
    for i in range(len(intensities)):
        ret[i] = ret[i] / l
    return ret


def plotMedia(medias, path, intensities):
    for idx, media in enumerate(medias):
        plt.plot(intensities, media, label=f"{idx}")
    plt.legend(loc="upper left")
    plt.savefig(path)
    plt.close("all")


def writeToFile(jsonFile, path):
    jsonString = json.dumps(jsonFile)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    folderPath = sys.argv[1]
    dir = os.listdir(folderPath)
    paths = [
        f"{folderPath}/{x}"
        for x in dir
        if ".png" in x
        or ".jpg" in x
        or ".jpeg" in x
        or ".gif" in x
        or ".JPG" in x
        or ".JPEG" in x
        or ".PNG" in x
        or ".GIF" in x
    ]
    intensities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    saveImages = False
    usePique = False
    useNima = True
    # special = "defaultIs0"
    special = ""
    blur, noise, over, under = fromPathToMatrix(
        paths, intensities, saveImages, 25, usePique, useNima
    )
    if saveImages:
        # Save new images (altered)
        blur = saveImagesFromMatrix(blur)
        noise = saveImagesFromMatrix(noise)
        over = saveImagesFromMatrix(over)
        under = saveImagesFromMatrix(under)
    # Save json
    output = f"{folderPath}/output"
    writeToFile(blur, f"{output}/blur.json")
    writeToFile(noise, f"{output}/noise.json")
    writeToFile(over, f"{output}/overexposure.json")
    writeToFile(under, f"{output}/underexposure.json")
    # Plot results
    graphsOutput = f"{folderPath}/graphs"
    media = list()
    media.append(
        plotMatrix(blur, f"{graphsOutput}/blur-brisque{f'-{special}'}.png", "brisque", intensities)
    )
    media.append(
        plotMatrix(
            noise, f"{graphsOutput}/noise.png-brisque{f'-{special}'}.png", "brisque", intensities
        )
    )
    media.append(
        plotMatrix(
            over, f"{graphsOutput}/overexposure-brisque{f'-{special}'}.png", "brisque", intensities
        )
    )
    media.append(
        plotMatrix(
            under, f"{graphsOutput}/underexposure-brisque{f'-{special}'}.png", "brisque", intensities
        )
    )
    if usePique:
        media.append(
            plotMatrix(blur, f"{graphsOutput}/blur-piqe.png", "piqe", intensities)
        )
        media.append(
            plotMatrix(noise, f"{graphsOutput}/noise.png-piqe.png", "piqe", intensities)
        )
        media.append(
            plotMatrix(
                over, f"{graphsOutput}/overexposure-piqe.png", "piqe", intensities
            )
        )
        media.append(
            plotMatrix(
                under, f"{graphsOutput}/underexposure-piqe.png", "piqe", intensities
            )
        )
    if useNima:
        media.append(
            plotMatrix(blur, f"{graphsOutput}/blur-nima.png", "nima", intensities)
        )
        media.append(
            plotMatrix(noise, f"{graphsOutput}/noise.png-nima.png", "nima", intensities)
        )
        media.append(
            plotMatrix(
                over, f"{graphsOutput}/overexposure-nima.png", "nima", intensities
            )
        )
        media.append(
            plotMatrix(
                under, f"{graphsOutput}/underexposure-nima.png", "nima", intensities
            )
        )
    plotMedia(media, f"{graphsOutput}/media.png", intensities)
