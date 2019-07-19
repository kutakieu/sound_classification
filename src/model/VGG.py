import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2D(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias, use_relu=True):
        super(conv2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.BN = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU()
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, dim_in, dim_out, bias, use_relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=bias)
        self.BN = nn.BatchNorm1d(dim_out)
        self.relu = nn.ReLU()
        self.use_relu = use_relu

    def forward(self, x):
        x = self.fc(x)
        x = self.BN(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class Sound_VGG(nn.Module):
    def __init__(self):
        super(Sound_VGG, self).__init__()

        self.maxpool2x2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.maxpool1x2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0)
        # self.global_avg_pool = nn.functional.avg_pool2d()
        # self.dropout = nn.Dropout2d(p)

        self.conv1 = conv2D(1, 64, kernel_size=5, stride=2, padding=2, bias=True)
        self.conv2 = conv2D(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = conv2D(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = conv2D(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5 = conv2D(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = conv2D(256, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = conv2D(384, 384, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv9 = conv2D(384, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv10 = conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv11 = conv2D(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv12 = conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv13 = conv2D(1024, 1024, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv14 = conv2D(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv15 = conv2D(512, 41, kernel_size=1, stride=1, padding=0, bias=True, use_relu=False)
        # self.conv15_2 = conv2D(512, 256, kernel_size=1, stride=1, padding=0, bias=True, use_relu=True)
        self.conv15_2_1 = conv2D(512, 128, kernel_size=1, stride=1, padding=0, bias=True, use_relu=True)
        self.conv15_2_2 = conv2D(128, 41, kernel_size=1, stride=1, padding=0, bias=True, use_relu=False)
        self.fc1 = FC(512, 256, bias=True, use_relu=True)
        self.fc2 = FC(256, 128, bias=True, use_relu=True)
        self.fc3 = FC(128, 41, bias=True, use_relu=False)

    def forward(self, x, is_training):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2x2(x)
        x = F.dropout2d(x, p=0.3, training=is_training)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2x2(x)
        x = F.dropout2d(x, p=0.3, training=is_training)

        x = self.conv5(x)
        x = F.dropout2d(x, p=0.3, training=is_training)
        x = self.conv6(x)
        x = F.dropout2d(x, p=0.3, training=is_training)
        x = self.conv7(x)
        x = F.dropout2d(x, p=0.3, training=is_training)
        x = self.conv8(x)
        x = self.maxpool2x2(x)
        x = F.dropout2d(x, p=0.3, training=is_training)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool1x2(x)
        x = F.dropout2d(x, p=0.3, training=is_training)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool1x2(x)
        x = F.dropout2d(x, p=0.3, training=is_training)

        x = self.conv13(x)
        x = F.dropout2d(x, p=0.5, training=is_training)
        x = self.conv14(x)
        x = F.dropout2d(x, p=0.5, training=is_training)
        x = self.conv15(x)
        # x = self.conv15_2_1(x)
        # x = self.conv15_2(x)
        # print(x.shape)
        x = nn.functional.max_pool2d(x, kernel_size=(6,4), padding=0).squeeze()
        return x

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5, training=is_training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=is_training)
        x = self.fc3(x)

        # x = self.conv15_2_2(x)

        return x
