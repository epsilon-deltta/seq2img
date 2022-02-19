# preparation

from torch import nn
import torch
from torchvision import transforms
import timm

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.backbone = torchvision.models.resnet101()
        in_nodes = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_nodes,1)

    def forward(self,x):

        x = self.backbone(x)
        return x
    
class ResNext(nn.Module):
    def __init__(self):
        super(ResNext,self).__init__()
        self.backbone = torchvision.models.resnext50_32x4d(pretrained=False)
        in_nodes = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(in_nodes,in_nodes//2),nn.Linear(in_nodes//2,1))
    def forward(self,x):
        x = self.backbone(x)
        return x
    
class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet,self).__init__()
        self.backbone = torchvision.models.shufflenet_v2_x1_5()
        in_nodes = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_nodes,1)
    def forward(self,x):
        x = self.backbone(x)
        return x
    
class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet,self).__init__()
        self.backbone = torchvision.models.squeezenet1_0(pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(256,1)
          )
    def forward(self,x):
        x = self.backbone(x)
        return x

class MNASNet(nn.Module):
    def __init__(self):
        super(MNASNet,self).__init__()
        self.backbone = torchvision.models.mnasnet1_0()
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(1280,640,bias=True),
            nn.ReLU(),
            nn.Linear(640,1)
        )
    def forward(self,x):
        x = self.backbone(x)
        return x

class MobileV3Net(nn.Module):
    def __init__(self):
        super(MobileV3Net,self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small()
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1, bias=True)
          )
    def forward(self,x):
        x = self.backbone(x)
        return x
    
    
class Vit(nn.Module):
    def __init__(self):
        super(Vit,self).__init__()

        self.transform = transforms.Compose([
            transforms.Resize(size = (224,224),interpolation=transforms.InterpolationMode.BICUBIC,max_size=None, antialias=None),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000],std=[0.5000, 0.5000, 0.5000] )
        ])
        self.backbone =  timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)

    def forward(self,x):

        x = self.transform(x)
        x = self.backbone(x)
        return x
    
def get_model(model:str,args):
    
    model = model.lower()
    
    if model == 'resnet':
        md = ResNet()
    elif model == 'resnext':
        md = ResNext()
    elif model == "shufflenet":
        md = ShuffleNet()
    elif model == "squeezenet":
        md = SqueezeNet()
    elif model == "mnasnet":
        md = MNASNet()
    elif model == 'mobilenet':
        md = MobileV3Net()
    elif model == 'vit':
        md = Vit()
    else:
        raise ValueError(f"There is no '{model}' ")
    # config = {'transform':transform,'batch_size':batch_size,'loss':loss,'exist_acc':exist_acc}
    config = args
    return md,config