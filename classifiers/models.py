from torchvision.models import resnet18,vgg13_bn,densenet121,wide_resnet50_2,alexnet,efficientnet_b0,\
    shufflenet_v2_x2_0,mobilenet_v3_large,vit_l_32,resnet50,resnet101
import torch
import torch.nn as nn

class AConvNet(nn.Module):
    def __init__(self, num_class=10, dropout_rate=0.5, ):
        super(AConvNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self._layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(128, self.num_class, kernel_size=(3, 3), stride=(1, 1), padding="valid"),
            nn.Flatten(),
            nn.Linear(6 * 6 * self.num_class, self.num_class),
        )

    def forward(self, x):
        return self._layer(x)

class CNN(nn.Module):
    def __init__(self, num_class=10):
        super(CNN, self).__init__()
        self.num_class = num_class
        self._conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            
        )
        self._liner = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, self.num_class),
        )

    def forward(self, x):
        x = self._conv(x)
        x = self._liner(x)
        return x

def _aconvnet(num_class):
    return AConvNet(num_class=num_class)

def _cnn(num_class):
    return CNN(num_class=num_class)

def _alexnet(num_class):
    model = alexnet(num_classes=num_class)
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    return model

def _vgg(num_class):
    model = vgg13_bn(num_classes=num_class)
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    return model

def _resnet(num_class):
    model = resnet18(num_classes=num_class)
    model.conv1= torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def _resnet50(num_class):
    model = resnet50(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def _resnet101(num_class):
    model = resnet101(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def _wideresnet(num_class):
    model = wide_resnet50_2(num_classes=num_class)
    model.conv1= torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def _densenet(num_class):
    model = densenet121(num_classes=num_class)
    model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def _vit(num_class):
    model = vit_l_32(num_classes=num_class,image_size=128)
    model.conv_proj = torch.nn.Conv2d(1, 1024, kernel_size=(32, 32), stride=(32, 32))
    return model

def _efficientnet(num_class):
    model = efficientnet_b0(num_classes=num_class)
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    return model

def _mobilenet(num_class):
    model = mobilenet_v3_large(num_classes=num_class)
    model.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
    return model

def _shufflenet(num_class):
    model = shufflenet_v2_x2_0(num_classes=num_class)
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)
    return model


def get_model(name,num_class):
    name = '_'+name
    return globals()[name](num_class=num_class)