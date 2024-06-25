import numpy as np
import cv2
import os
import argparse



def closing(img, kernel_size, iterations):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

def fit_ellipse(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (thresh, im_gray) = cv2.threshold(im_gray, 100, 255, 0)
    contours, hierarchy = cv2.findContours(im_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest = 0
    min = 0
    for idx,countour in enumerate(contours):
        if countour.shape[0]>min:
            largest = idx
            min = countour.shape[0]
    empty = np.zeros_like(img)
    if not contours or np.count_nonzero(img) < 400:
        # print(np.count_nonzero(img))
        return empty
    else:
        cnt = contours[largest]
        if cnt.shape[0]<5:
            print('not enough point')
            return img
        ellipse = cv2.fitEllipse(cnt)
        # print("shape: {}".format(empty.shape))
        img_ellipse = cv2.ellipse(empty, ellipse, (255,0,255), thickness=-1)

        return img_ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--seq', default = "S5")
args = parser.parse_args() 
seq = args.seq
filepath = "./output/"+seq

print(len(filepath))
allFileList = os.walk(filepath)
for root, dirs, files in allFileList:
    for file in files:
        if ".png" in file:
            img_path = os.path.join(root,file)
            print("processing... {}".format(img_path))
            img = cv2.imread(img_path)
            img_post = closing(img, kernel_size=5, iterations=5)
            img_post = fit_ellipse(img_post)
            cv2.imwrite(img_path,img_post)
        
            

