# official package
from torch.utils.data import Dataset
import os 
from PIL import Image 
import torchvision.transforms as transforms 
import torch
import random
import numpy as np



class PupilDataSet_seg(Dataset):
    def __init__(self, root):
        self.imgs, self.masks = [], []
        self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                            mean = 0.5,
                            std = 0.5),
                    ])
        
        all_img, all_mask = [], []
        
        # read data
        for seq in ['S1', 'S2', 'S3', 'S4']:
            for path, _, files in os.walk(root + "/" + seq):
                for file in files:
                    if file.endswith(".jpg"):
                        all_img.append(os.path.join(path, file))
                    elif file.endswith(".png"):
                        all_mask.append(os.path.join(path, file)) 
                         
                                     
        all_img.sort(), all_mask.sort()
        
        for i, path in enumerate(all_mask): 
            mask_ori = Image.open(path).convert('RGB')
            mask_ori = transforms.ToTensor()(mask_ori)
            if torch.count_nonzero(mask_ori) > 0: 
                self.imgs.append(all_img[i])
                self.masks.append(path) 
            
            
            
    def __getitem__(self, idx):
        img_ori = Image.open(self.imgs[idx])
        img_ori = transforms.functional.adjust_gamma(img_ori, gamma = 0.4) 
        
        mask_ori = Image.open(self.masks[idx]).convert('RGB')
        
        
        img_aug, mask_aug = position_transform(img_ori, mask_ori)
        img_aug = self.trans(image_transfrom(img_aug))
        
        mask_ori = binary_mask(transforms.ToTensor()(mask_ori))
        mask_aug = binary_mask(transforms.ToTensor()(mask_aug))
        img_ori = self.trans(img_ori)
        return (img_aug, mask_aug), (img_ori, mask_ori)
        
        
        
    def __len__(self):
        return len(self.imgs)
    
    
    def get_path(self, idx):
        return self.imgs[idx]



class PupilDataSet_conf(Dataset):
    def __init__(self, root):
        self.imgs, self.masks = [], []
        self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                            mean = 0.5,
                            std = 0.5),
                    ])
        
        # read data
        for seq in ['S1', 'S2', 'S3', 'S4']:
            for path, _, files in os.walk(root + "/" + seq):
                for file in files:
                    if file.endswith(".jpg"):
                        self.imgs.append(os.path.join(path, file))
                    elif file.endswith(".png"):
                        self.masks.append(os.path.join(path, file))
                        
        self.imgs.sort(), self.masks.sort()
        self.len = len(self.imgs)
            
            
    def __getitem__(self, idx):
        img_ori = Image.open(self.imgs[idx]).convert('RGB')
        img_ori = transforms.functional.adjust_gamma(img_ori, gamma=0.4)

        mask_ori = Image.open(self.masks[idx]).convert('RGB')
        
        img_aug, mask_aug = position_transform(img_ori, mask_ori)
        img_aug = self.trans(image_transfrom(img_aug))
        
        mask_ori = binary_mask(transforms.ToTensor()(mask_ori))
        mask_aug = binary_mask(transforms.ToTensor()(mask_aug))
        img_ori = self.trans(img_ori)
        
        conf_aug, conf_ori = mask_aug.any().to(torch.long), mask_ori.any().to(torch.long)
        return (img_aug, conf_aug), (img_ori, conf_ori)
       
        
        
    def __len__(self):
        return self.len
    
    
    def get_path(self, idx):
        return self.imgs[idx]



class PupilDataSet_test(Dataset):
    def __init__(self, root, seq):
        self.imgs = []
        self.seq = seq
        self.trans = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                                mean = 0.5,
                                std = 0.5),
                    ])
        
        allFileList = os.walk(os.path.join(root, seq)) 
        for root, dirs, files in allFileList: 
            dirs.sort()
            files.sort(key=lambda x:int(x[:-4]))
            
            for file in files: 
                img_path = os.path.join(root, file) 
               
                if seq == 'HM' or seq == 'KL':
                    self.imgs.append(img_path)  
                elif file.endswith(".jpg"):
                    self.imgs.append(img_path)                    
                                     
        self.imgs.sort()
             
            
            
    def __getitem__(self, idx): 
        if self.seq == 'HM' or self.seq == 'KL': 
            img = Image.open(self.imgs[idx]).convert('L')        
            img = self.pad_zero(img)
        else:
            img = Image.open(self.imgs[idx]) 
        img = transforms.functional.adjust_gamma(img, gamma = 0.4)
        
        
        return self.trans(img)
        
        
    def __len__(self):
        return len(self.imgs)
    
    
    def get_path(self, idx):
        return self.imgs[idx]  
    
    
    def pad_zero(self, img): 
        tmp = transforms.ToTensor()(img)
        
        new_img = torch.zeros((1, 480, 640)) 
        new_img[:, 80:400, 160:480] = tmp 
        
        return transforms.ToPILImage()(new_img)
    
    
    
    
def binary_mask(mask):
    return mask.any(axis = 0).to(torch.long) 
    
    
    
def position_transform(image,label):
    num = np.random.randint(6)
    if num == 1:
        t1 = transforms.RandomRotation(15)
        state = torch.get_rng_state()
        image = t1(image)
        torch.set_rng_state(state)
        label = t1(label)
    elif num == 2 or num == 3: 
        t1 = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        state = torch.get_rng_state()
        image = t1(image)
        torch.set_rng_state(state)
        label = t1(label) 
    elif num == 4 : 
        t1 = transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1))
        state = torch.get_rng_state()
        image = t1(image)
        torch.set_rng_state(state)
        label = t1(label) 
        
        
        
    t2 = transforms.RandomHorizontalFlip()
    state = torch.get_rng_state()
    image = t2(image)
    torch.set_rng_state(state)
    label = t2(label)

    return image, label


def gaussian_noise(img, mean=0, sigma=0.05):
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    return gaussian_out


def image_transfrom(image):
    trans = transforms.RandomChoice([
        transforms.GaussianBlur(kernel_size=5, sigma=(2)),
        transforms.GaussianBlur(kernel_size=5, sigma=(3)),
        transforms.GaussianBlur(kernel_size=5, sigma=(4)),
        transforms.GaussianBlur(kernel_size=5, sigma=(5)),
        transforms.GaussianBlur(kernel_size=5, sigma=(6)),
        transforms.GaussianBlur(kernel_size=5, sigma=(7)),
    ])
    
    sigma_list = [0.02,0.04,0.06]

    sigma = random.choice(sigma_list)
    img_list = [image, trans(image),gaussian_noise(np.array(image),sigma=sigma)]

    image = random.choice(img_list)

    return image