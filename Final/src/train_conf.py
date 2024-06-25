# official package
import argparse, os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim 
from efficientnet_pytorch import EfficientNet 

# my package
from schedulers import WarmupPolyLR
from dataset import PupilDataSet_conf
from solve import conf_train
from utils import get_device



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "../dataset/public")
    args = parser.parse_args() 
    
    
    data_root = args.data_root 
    
    val_percent = 0.1
    dataset = PupilDataSet_conf(data_root) 
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val


    trainset, validset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1120))
    print('Numbers of images in trainset:', len(trainset)) 
    print('Numbers of images in validset:', len(validset)) 
    

    trainset_loader = DataLoader(trainset, batch_size = 12, shuffle = True, num_workers = 12) 
    validset_loader = DataLoader(validset, batch_size = 12, shuffle = True, num_workers = 12)
    
    device = get_device()
    model = EfficientNet.from_pretrained('efficientnet-b1', 
                                        in_channels = 3, 
                                        num_classes = 2).to(device)
    

    weights = [0.9, 0.1]
    class_weights = torch.FloatTensor(weights).to(device)
    config = {
        "epoch": 30,
        "device": device,
        "criterion": nn.CrossEntropyLoss(weight = class_weights),
        "trainset_loader": trainset_loader,
        "validset_loader": validset_loader,
        'data_root': data_root,
        "model": model,
        "save_path": "./save_model/trained_conf.pth"
    } 
    
    optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.9, 0.999), weight_decay = 0.01)
    scheduler = WarmupPolyLR(optimizer, 0.9, 30, 7, 0.1, warmup='linear')
    
    conf_train(config, optimizer, scheduler)