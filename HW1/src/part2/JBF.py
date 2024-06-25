import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
    
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ## TODO 
        H, W, _ = img.shape
        
        r_table = self.get_range_table()
        Gs = self.get_Gs()
        
        if (guidance.ndim == 3) :
            Tq = np.zeros((H, W, self.wndw_size, self.wndw_size, 3), dtype = np.int32)
        else:
            Tq = np.zeros((H, W, self.wndw_size, self.wndw_size), dtype = np.int32)
        Iq = np.zeros((H, W, self.wndw_size, self.wndw_size, 3))
        
        for h in range(H):
            for w in range(W):    
                Tq[h, w] = padded_guidance[h:h+self.wndw_size, w:w+self.wndw_size]
                Iq[h, w] = padded_img[h:h+self.wndw_size, w:w+self.wndw_size]
        
        Gr = self.get_Gr(guidance, Tq, guidance.ndim, r_table)
        G = np.expand_dims(Gs * Gr, axis = -1)
        output = (G * Iq).sum(axis = 2).sum(axis = 2) / G.sum(axis = 2).sum(axis = 2)
           
        return np.clip(output, 0, 255).astype(np.uint8)
        
        
    def get_Gs(self):
        ran = range(-self.pad_w, self.pad_w + 1)
        x, y = np.meshgrid(ran, ran)
        Gs = np.exp((x ** 2 + y ** 2) / (-2 * self.sigma_s ** 2))
        
        return Gs
    
    
    def get_range_table(self):
        diff = np.arange(0, 256) / 255
        range_table = np.exp(-((diff ** 2) / (2 * self.sigma_r ** 2)))
        
        return range_table
    
    
    def get_Gr(self, Tp, Tq, g_dim, r_table):
        H, W = Tp.shape[0], Tp.shape[1]
        Tp = Tp.reshape((H, W, 1, 1, 3)) if g_dim == 3 else Tp.reshape((H, W, 1, 1))
        diff = abs(Tp - Tq)
        Gr = np.prod(r_table[diff], axis = -1) if g_dim == 3 else r_table[diff]

        return Gr