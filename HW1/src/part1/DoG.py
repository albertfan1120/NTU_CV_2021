import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_oct = 2
        self.num_DoG = 4
        self.num_guassian = self.num_DoG + 1


    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        H, W = image.shape
        gaussian_images = self.get_Gaussian_imgs(image)
        resize_img = cv2.resize(gaussian_images[-1], (W // 2, H // 2), interpolation = cv2.INTER_NEAREST)
        gaussian_images += self.get_Gaussian_imgs(resize_img)
        
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = self.get_Dog_imgs(gaussian_images[:self.num_guassian]) +  \
                     self.get_Dog_imgs(gaussian_images[self.num_guassian:])   
        
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        extremum = self.get_extremum(dog_images[:self.num_DoG], 1) +  \
                   self.get_extremum(dog_images[self.num_DoG:], 2)
        
            
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(extremum), axis = 0)
        
        
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints

    
    def get_Gaussian_imgs(self, img):
        gaussian_list = [cv2.GaussianBlur(img, (0, 0), self.sigma ** pow) 
                         for pow in range(1, self.num_guassian)]
        
        return [img] + gaussian_list
    
    
    def get_Dog_imgs(self, img_list):
        dog_list = [cv2.subtract(img_2, img_1) 
                    for img_1, img_2 in zip(img_list[:-1], img_list[1:])]
        
        return dog_list
    
    
    def get_extremum(self, img_list, scale): # scale = 1 for first octave, = 2 for second
        extremum = []
        H, W = img_list[0].shape
        img_tuple = zip(img_list[:self.num_DoG-2], img_list[1:self.num_DoG-1], img_list[2:])
        for img_1, img_2, img_3 in img_tuple:
            for h in range(1, H-1):
                for w in range(1, W-1):
                    window = np.array([img_1[h-1:h+2, w-1:w+2], 
                                       img_2[h-1:h+2, w-1:w+2],
                                       img_3[h-1:h+2, w-1:w+2],])
                    if self.is_extremum(window): extremum.append([h * scale, w * scale])
                        
        return extremum
                        
    
    def is_extremum(self, window):
        center_val = window[1, 1, 1]
        if abs(center_val) > self.threshold:
            if center_val > 0 and window.max() == center_val: return True
            elif center_val < 0 and window.min() == center_val: return True
                
        return False