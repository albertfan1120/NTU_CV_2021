# official package
import torch
import numpy as np


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    return device


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path) 
    
    
def get_iou_score(mask1, mask2):
    area1, area2 = np.count_nonzero(mask1), np.count_nonzero(mask2)
    if area1 == 0 or area2 == 0:
        return 0
    
    area_union = np.count_nonzero(mask1 + mask2)
    area_inter = area1 + area2 - area_union
    
    return area_inter / area_union


def get_TNR(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    
    thresholds = np.linspace(0, 1, 1000)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (pred < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - gt) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - gt))
    return np.array(tn_rates).mean()


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)