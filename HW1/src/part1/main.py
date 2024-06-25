import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)
    
    
def normalize_img(dog_images):
    nor_imgs = []
    for img in dog_images:
        max, min = img.max(), img.min()
        img = 255 * (img - min) / (max - min)
        nor_imgs.append(img.astype(np.uint8))
        
    return nor_imgs


def plot_DoG(imgs):
    num_oct = 2
    num_DoG = 4
    
    cnt = 0
    for i in range(1, num_oct+1):
        for j in range(1, num_DoG+1):
            path = "DoG" + str(i) + "-" + str(j) + ".png"
            cv2.imwrite(path, imgs[cnt])
            cnt += 1
     

def main():
    img1 = cv2.imread("testdata/1.png", 0).astype(np.float32)
    img2 = cv2.imread("testdata/2.png", 0).astype(np.float32)
    
    ### TODO ###
    # Visualize the DoG images for 1.png
    threshold = 5
    DoG = Difference_of_Gaussian(threshold)
    
    H, W = img1.shape
    gaussian_images = DoG.get_Gaussian_imgs(img1)
    resize_img = cv2.resize(gaussian_images[-1], (W // 2, H // 2), interpolation = cv2.INTER_NEAREST)
    gaussian_images += DoG.get_Gaussian_imgs(resize_img)
    
    dog_images = DoG.get_Dog_imgs(gaussian_images[:DoG.num_guassian]) +  \
                 DoG.get_Dog_imgs(gaussian_images[DoG.num_guassian:])
                 
    nor_imgs = normalize_img(dog_images)
    plot_DoG(nor_imgs)
    
    
    # Use three thresholds (2, 5, 7) on 2.png and describe the difference
    thresholds = [2, 5, 7]
    for t in thresholds:
        DoG = Difference_of_Gaussian(t)
        keypoints = DoG.get_keypoints(img2)
        save_path = "treshold" + str(t) + ".png"
        plot_keypoints(img2, keypoints, save_path)
    

if __name__ == '__main__':
    main()