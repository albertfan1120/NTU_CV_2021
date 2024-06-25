# official package
import torch
import numpy as np
import cv2, os
import torchvision.transforms as transforms
from PIL import Image 

# my package
from utils import get_iou_score, save_checkpoint, get_TNR


def seg_train(config, optimizer, scheduler):
    epoch = config['epoch']
    device = config['device']
    model = config['model']
    save_path = config["save_path"]
    criterion, trainset_loader = config['criterion'], config['trainset_loader']

    model.train()  
    log_interval = 600
    best_iou = 0
    for ep in range(1, epoch+1):
        iteration = 0
        for batch_idx, ((image, mask), _) in enumerate(trainset_loader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            
            output = model(image.float())
            loss = criterion(output, mask)
            
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(image), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        
        if scheduler is not None:
            scheduler.step()
        
        score = seg_validation(config)
        if score > best_iou:
            best_iou = score
            save_checkpoint(save_path, model, optimizer)
            print("Save best checkpoint sucessfully!!\n")
        

def seg_validation(config):
    validset_loader = config['validset_loader']
    device = config['device']
    model = config['model']
    
    #print(validset_loader.dataset.mode)
    model.eval()  # Important: set evaluation mode
    predList, maskList = [], []
    with torch.no_grad(): 
        for (_, (image, mask)) in validset_loader:
            image, mask = image.to(device), mask.to(device)
            output = model(image.float())
            
            predList += [singleBatch for singleBatch in output.cpu().numpy()]
            maskList += [singleBatch for singleBatch in mask.cpu().numpy()]
    
    
    pred = torch.from_numpy(np.array(predList).argmax(axis = 1))
    gt = torch.from_numpy(np.array(maskList))
    
    
    score = get_iou_score(pred, gt)
    print('\nValidation set:  IoU score = {:.2f}% \n'.format(100. * score))
    
    return score


def conf_train(config, optimizer, scheduler):
    epoch = config['epoch']
    device = config['device']
    model = config['model']
    save_path = config["save_path"]
    criterion, trainset_loader = config['criterion'], config['trainset_loader']

    model.train()  
    log_interval = 200
    best_tnr = 0
    for ep in range(1, epoch+1):
        iteration = 0
        for batch_idx, ((image, confs), _) in enumerate(trainset_loader):
            image, confs = image.to(device), confs.to(device)
            optimizer.zero_grad()
            
            output = model(image.float())
            #print(output.shape, confs.shape)
            loss = criterion(output, confs)
            
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(image), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        
        if scheduler is not None:
            scheduler.step()
        
        score = conf_validation(config)
        if score > best_tnr:
            best_tnr = score
            save_checkpoint(save_path, model, optimizer)
            print("Save best checkpoint sucessfully!!\n")
        

def conf_validation(config):
    validset_loader = config['validset_loader']
    device = config['device']
    model = config['model']
    
    #print(validset_loader.dataset.mode)
    model.eval()  # Important: set evaluation mode
    predList, maskList = [], []
    with torch.no_grad(): 
        for (_, (image, mask)) in validset_loader:
            image, mask = image.to(device), mask.to(device)
            output = model(image.float())
            
            predList += [singleBatch for singleBatch in output.cpu().numpy()]
            maskList += [singleBatch for singleBatch in mask.cpu().numpy()]
    
    
    pred = torch.from_numpy(np.array(predList).argmax(axis = 1))
    gt = torch.from_numpy(np.array(maskList))
    
    N = pred.shape[0]

    GT_nums = N - torch.count_nonzero(gt).item() 
    pred_nums = N - torch.count_nonzero(pred).item()
    print()
    print('GT confs zeros nums:', GT_nums)
    print()
    print('Your pred zeros nums:', pred_nums)


    
    score = get_TNR(pred, gt)
    print('\nValidation set:  TNR score = {:.2f}% '.format(100. * score))
    overall = score - 2 * (abs(GT_nums - pred_nums) / N)
    print('Validation set:  Overall score = {:.2f}%\n'.format(100. * overall))
    
    return overall


def seg_testing(config):
    testset_loader = config['testset_loader']
    device = config['device']
    model = config['model']
    dataset_path = config['data_root']
    output_path = config['output_path'] 
    seq = config['seq'] 
      
    model.eval()  # Important: set evaluation mode
    predList = []
    with torch.no_grad(): 
        for image in testset_loader:
            image = image.to(device)
            output = model(image.float())
            
            predList += [singleBatch for singleBatch in output.cpu().numpy()]
    
    pred = torch.from_numpy(np.array(predList).argmax(axis = 1))
    

    for i in range(len(testset_loader.dataset)):
        path = testset_loader.dataset.get_path(i)
        
        path_list = path.split(dataset_path + '/')[-1].split('/')  
        
        if seq == 'HM' or seq == 'KL': 
            sub, num = path_list[-2], path_list[-1].split('.')[0]
        else: 
            sub, act, num = path_list[-3], path_list[-2], path_list[-1].rstrip('.jpg')
        
        img = pred[i] # (H, W)
        img = torch.unsqueeze(img, -1)
        
        zero = torch.zeros_like(img)
        color = torch.cat((img, zero, img), dim = -1)
        color = np.array(color * 255)

        if seq == 'HM' or seq == 'KL':
            save_path = os.path.join(output_path, sub, num) + '.png' 
            color = color[80:400, 160:480, :] 
        else:
            save_path = os.path.join(output_path, sub, act, num) + '.png' 
        cv2.imwrite(save_path, color)
        print('Save ', save_path)
        
        
        

def conf_testing(config): 
    device = config['device']
    model = config['model'] 
    dataset_path = os.path.join(config['data_root'], config['seq'])
    output_root = config['output_path'] 
    seq = config['seq']
    
    model.eval()  # Important: set evaluation mode 
    
    filepath = dataset_path
    allFileList = os.walk(filepath)
    
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean = 0.5,
                std = 0.5),
    ]) 
    
    if seq == 'HM' or seq == 'KL':
        with torch.no_grad(): 
            zero_cnt = 0
            for root, dirs, files in allFileList: 
                label = []
                dirs.sort()
                files.sort(key=lambda x:int(x[:-4])) 
            
                txt_path = os.path.join(root, 'conf.txt')
                for file in files: 
                    img_path = os.path.join(root,file)
                    img = Image.open(img_path).convert('RGB') 
                    
                    tmp = transforms.ToTensor()(img)
            
                    new_img = torch.zeros((3, 480, 640)) 
                    new_img[:, 80:400, 160:480] = tmp 
                    
                    img = transforms.ToPILImage()(new_img)
                    img = transforms.functional.adjust_gamma(img, gamma=0.4)
                    img = trans(img)[None, :, :, :]
        
                    output = model(img.to(device)).argmax(axis = 1)[0].item()
                    label.append(output)
                    
                    if output == 0: 
                        zero_cnt += 1
        
                #write label to text file 
                out_path = os.path.join(output_root, config['seq'], txt_path.split(dataset_path + '/')[-1])
                print('Save ', out_path)
                textfile = open(out_path, "w+")
                for i in range(len(label)):
                    textfile.write("%s\n" %(label[i]))
                textfile.close()
            
            print('Total zeros =', zero_cnt)
            
    else:
        with torch.no_grad(): 
            zero_cnt = 0
            for root, dirs, files in allFileList:
                label = []
                dirs.sort()
                files.sort(key=lambda x:int(x[:-4])) 
                
                txt_path = os.path.join(root, 'conf.txt')
                if txt_path == os.path.join(filepath, 'conf.txt'): continue
                
                for file in files: 
                    
                    if ".jpg" in file:   
                        img_path = os.path.join(root,file)
                        
                        img = Image.open(img_path).convert('RGB')
                        img = transforms.functional.adjust_gamma(img, gamma=0.4)
                        img = trans(img)[None, :, :, :]
                    
                        output = model(img.to(device)).argmax(axis = 1)[0].item()
                        label.append(output)
                        
                        if output == 0: 
                            zero_cnt += 1

                #write label to text file 
                out_path = os.path.join(output_root, config['seq'], txt_path.split(dataset_path + '/')[-1])
                print('Save ', out_path)
                textfile = open(out_path, "w+")
                for i in range(len(label)):
                    textfile.write("%s\n" %(label[i]))
                textfile.close()
            
            print('Total zeros =', zero_cnt)
            
            
            
            