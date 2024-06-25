# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet
import torch
import torch.nn as nn
import torchvision.models as models

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        
        return out

     
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_out)
        self.model = model
        
        
    def forward(self,x):
        return self.model(x)