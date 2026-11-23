import torch.nn as nn
import torch
import torch.nn.functional as F


# num_classes = 11
# ResNet {{{
class CNN2D(nn.Module):
    def __init__(self, dataset='128'):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 第一层卷积，输入通道数为1，输出通道数为256，卷积核大小为(1, 3)，填充为(0, 1)
        self.drop1 = nn.Dropout(p=0.5)
        # 第一层dropout，丢弃概率为0.5
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), bias=False)
        # 第二层卷积，输入通道数为256，输出通道数为80，卷积核大小为(2, 3)
        self.drop2 = nn.Dropout(p=0.5)
        # 第二层dropout，丢弃概率为0.5+
        self.dataset = dataset
        # 数据集参数

        # if num_classes == 11:
        if dataset in ['128', 'all_128']:
            num_classes = 11
            self.fc_pool = nn.Linear(126, 128)
            self.dense = nn.Linear(10240, 256)
        elif dataset in ['512', 'all_512']:
            self.fc_pool = nn.Linear(510, 256)
            num_classes = 12
            self.dense = nn.Linear(20480, 256)
        elif dataset == '1024':
            self.fc_pool = nn.Linear(1022, 256)
            num_classes = 24
            self.dense = nn.Linear(20480, 256)
        elif dataset == '3040':
            num_classes = 106
            self.fc_pool = nn.Linear(3038, 128)
            self.dense = nn.Linear(10240, 256)
        # self.dense = nn.Linear(10080, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.classfier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x)).squeeze(dim=2)
        x = self.fc_pool(x)
        x = self.drop2(x).view(x.size(0), -1)
        x = F.relu(self.dense(x))

        x = self.drop3(x)
        x = self.classfier(x)
        return x

##这段代码定义了一个卷积神经网络模型，根据数据集的不同，选择不同的全连接层和类别数来适应不同的任务。
# 通过卷积层、全连接层和分类器层的组合，实现对输入图像进行分类预测。同时，通过使用dropout层，
# 可以在训练过程中随机丢弃部分神经元，以提高模型的泛化能力和防止过拟合。


# def cnn2d(**kwargs):
#     return CNN2D(**kwargs)
# data = torch.randn(10,2,512)
# model = cnn2d()
# out = model(data)
# print(out.shape)
# from torchsummary import summary
# model = cnn2d().cuda()
# summary(model, (2, 128))

# from torchinfo import summary
# model = CNN2D(dataset='512').cuda()
# summary(model, input_size=(128, 2, 512))
#
