import torch.nn as nn

# 去掉预训练网络的最后一层
class Net(nn.Module):
    def __init__(self, model, num_class):
        super(Net, self).__init__()
        # ResNet原本接受三通道输入，改为单通道灰度图
        self.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 取掉model的最后一层
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        # self.Linear_layer = nn.Linear(512, num_class) #加上一层参数修改好的全连接层 resnet18 resnet34
        self.Linear_layer1 = nn.Linear(2048, 1024) #加上一层参数修改好的全连接层 resnet50
        self.Linear_layer2 = nn.Linear(1024, 512)
        self.Linear_layer3 = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x) # resnet18 resnet34
        x = self.Linear_layer1(x)
        x = self.Linear_layer2(x)
        x = self.Linear_layer3(x)
        return x
 
