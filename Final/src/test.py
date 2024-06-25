# official package
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet 
import os
import argparse

# my package
from solve import seg_testing, conf_testing
from model import Segmentor
from utils import get_device, load_checkpoint 
from dataset import PupilDataSet_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "../dataset/public")
    parser.add_argument('--seq', default = "S5") 
    parser.add_argument('--act_num', default = 26)
    args = parser.parse_args() 
    
    data_root = args.data_root #"../dataset/public"
    subjects = [args.seq]
    
    output_path = './output'
    
    
    if args.seq == 'HM' or args.seq == 'KL': 
        save_path = os.path.join(output_path, args.seq)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
    else:
        act_num = int(args.act_num)
        for sub in subjects: 
            for action_number in range(1, act_num+1):
                save_path = os.path.join(output_path, sub, str(action_number).zfill(2))
                if not os.path.isdir(save_path):
                    os.makedirs(save_path) 
    

    # # ## ========================================== Segmentation =============================================== ##
    testset = PupilDataSet_test(data_root, args.seq)
    testset_loader = DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 12)
    device = get_device()
    
    
    model = Segmentor(in_channels=1,out_channels=2,channel_size=48,dropout=True,prob=0.25).to(device)
    load_checkpoint('./save_model/seg.pth', model)
    
    config = {
        "device": device,
        "testset_loader": testset_loader,
        'data_root': data_root,
        'output_path': output_path,
        "model": model,
        'seq': args.seq,
    }
    
    print('\nStart predict segmentation!!!')
    seg_testing(config)
    print('Finish predict segmentation!!!\n')
    
    # # # ============================================ Conf ====================================================== ##
    model = EfficientNet.from_pretrained('efficientnet-b1', 
                                        in_channels = 3, 
                                        num_classes = 2).to(device)
    load_checkpoint('./save_model/conf.pth', model)
    
    
    config = {
        "device": device,
        "testset_loader": testset_loader,
        'data_root': data_root,
        'output_path': output_path,
        "model": model,
        'seq': args.seq,
    } 
    
    print('\nStart predict confidece!!!')
    conf_testing(config)
    print('Finish predict confidece!!!')