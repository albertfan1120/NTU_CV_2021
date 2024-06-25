import numpy as np
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, _ = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)


    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    cen_l, cen_r = census_map(Il), census_map(Ir)
    cost_l, cost_r = match_cost(cen_l, cen_r, max_disp)
    
    
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    smooth_l, smooth_r = cost_aggregate(Il, Ir, cost_l, cost_r, max_disp)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    dis_l, dis_r = smooth_l.argmin(axis = -1), smooth_r.argmin(axis = -1)
    
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    hole_val = max_disp + 1 # cannot be same with vaild disparity value
    dis_hole = check_consist(dis_l, dis_r, hole_val)
    labels = hole_filling(dis_hole, max_disp, hole_val)
    labels = xip.weightedMedianFilter(Il.astype('uint8'), labels, 17, 7, xip.WMF_JAC)
    
    
    return labels.astype(np.uint8)


def census_map(img):
    '''
        img -> (H, W, C)
        
        output -> (H, W, C, 8) bool type
    '''
    img_pad = np.pad(img, ((1,1),(1,1),(0,0)), 'constant', constant_values = (0,0))
    ring = np.stack((img_pad[0:-2, 0:-2], 
                     img_pad[0:-2, 1:-1],
                     img_pad[0:-2, 2:  ],
                     img_pad[1:-1, 0:-2],
                     img_pad[1:-1, 2:  ],
                     img_pad[2:  , 0:-2],
                     img_pad[2:  , 1:-1],
                     img_pad[2:  , 2:  ]), axis = -1)
    
    return img[:, :, :, np.newaxis] >= ring


def match_cost(cen_l, cen_r, max_disp):
    '''
        cen_l, cen_r -> (H, W, C, 8) bool type
        max_disp -> scaler
        
        cost_l, cost_r -> (H, W) float32 type
    '''
    H, W, _, _ = cen_l.shape 
    cost_l = np.empty((H, W, max_disp+1), dtype = "float32")
    cost_r = np.empty((H, W, max_disp+1), dtype = "float32")
    
    for d in range(max_disp+1):
        left, right = cen_l[:, d:], cen_r[:, :W-d]        
        ham_map = np.sum(left != right, axis = (2, 3)) # (H, W-d)

        pad_l, pad_r = np.tile(ham_map[:, [0]], (1, d)), np.tile(ham_map[:, [-1]], (1, d))
        cost_l[:, :, d] = np.hstack((pad_l, ham_map))
        cost_r[:, :, d] = np.hstack((ham_map, pad_r))
        
    return cost_l, cost_r
        

def cost_aggregate(Il, Ir, cost_l, cost_r, max_disp):
    '''
        Il, Ir -> (H, W, C) float32 type
        cost_l, cost_r -> (H, W) float32 type
        max_disp -> scaler
        
        smooth_l, smooth_r -> (H, W) float32 type
    '''
    H, W, _ = Il.shape 
    smooth_l = np.empty((H, W, max_disp+1), dtype = "float32")
    smooth_r = np.empty((H, W, max_disp+1), dtype = "float32")
    
    for d in range(max_disp+1):
        smooth_l[:, :, d] = xip.jointBilateralFilter(Il, cost_l[:, :, d], 15, 30, 15)
        smooth_r[:, :, d] = xip.jointBilateralFilter(Ir, cost_r[:, :, d], 15, 30, 15)
    
    return smooth_l, smooth_r


def check_consist(dis_l, dis_r, hole_val):
    '''
        dis_l, dis_r -> (H, W) 
        hole_val -> scaler
        
        output -> (H, W) with hole
    '''
    H, W = dis_l.shape
    for h in range(H):
        for w in range(W):
            d = dis_l[h, w]
            if (w - d >= 0) and d != dis_r[h, w-d]:
                dis_l[h, w] = hole_val
                
    return dis_l            
         

def hole_filling(dis, max_disp, hole_val):
    '''
        dis -> (H, W) with hole
        max_disp, hole_val -> scalers
        
        labels -> (H, W) uint8
    '''
    dis_pad = np.pad(dis, ((0,0), (1,1)), 'constant', constant_values = max_disp)
    Fl, Fr = np.copy(dis_pad), np.copy(dis_pad)
    H, W = dis_pad.shape 
    
    for h in range(H):
        for w in range(1, W-1):
            if dis_pad[h, w] == hole_val:
                w_left = w - 1
                while dis_pad[h, w_left] == hole_val and w_left != 0:
                    w_left = w_left - 1
                
                w_right = w + 1
                while dis_pad[h, w_right] == hole_val and w_right != W:
                    w_right = w_right + 1
                
                Fl[h, w], Fr[h, w] = dis_pad[h, w_left], dis_pad[h, w_right]
                
    labels = np.minimum(Fl, Fr)
    return labels[:, 1:-1].astype('uint8') 