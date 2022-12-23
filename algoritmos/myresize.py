import numpy as np
import time


def resize(img, w , h):
    ret = []
    black_image = np.zeros((h,w,3), np.uint8)
    black_image[:,:] = (0,0,0) #rgb
    print("Resizing img to fit slideshow")
    ii = 0
    jj = len(img)
    frameTime = 0
    timeF = 0
    for i in img:
        print("("+str(ii)+" / "+str(jj-1)+")", end="\r")
        startF = time.time()
        timeF = timeF + 1
        ii = ii + 1
        height, width = i["frame"].shape[:2]
        if width <w or height<h:
            l_img = black_image.copy()
            x_offset = int(w/2 - width/2)
            y_offset = int(h/2 - height/2)
            l_img[y_offset:y_offset+height, x_offset:x_offset+width] = i["frame"].copy()
            ret.append({"frame":l_img,"image_id":i["image_id"]})
        else:
            ret.append(i)
        frameTime = frameTime + time.time() - startF
    print("\n")
    print("Time elapsed per image resized ("+str(frameTime/timeF)+")")
    return ret


def resize2(img, w, h):
    black_image = np.zeros((h,w,3), np.uint8)
    black_image[:,:] = (0,0,0) #rgb
    print("Resizing img to fit slideshow")
    ii = 0
    jj = len(img)
    frameTime = 0
    timeF = 0
    for i in img:
        print("("+str(ii)+" / "+str(jj-1)+")", end="\r")
        startF = time.time()
        timeF = timeF + 1
        ii = ii + 1
        height, width = i["frame"].shape[:2]
        if width <w or height<h:
            l_img = black_image.copy()
            x_offset = int(w/2 - width/2)
            y_offset = int(h/2 - height/2)
            l_img[y_offset:y_offset+height, x_offset:x_offset+width] = i["frame"].copy()
            i["frame"] = l_img
        frameTime = frameTime + time.time() - startF
    print("Time elapsed per image resized ("+str(frameTime/timeF)+")")
    print("\n")
    return img