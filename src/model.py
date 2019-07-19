import torch
import torch.nn as nn
import torch.nn.functional as F


class Sound_VGG(nn.Module):
    def __init__(self):
        super(Sound_VGG, self).__init__()

        self.maxpool2x2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.maxpool1x2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0)
        # self.global_avg_pool = nn.functional.avg_pool2d()
        # self.dropout = nn.Dropout2d(p)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2,
                         padding=2, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                         padding=1, bias=False)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                         padding=1, bias=False)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv7 = nn.Conv2d(256, 384, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv8 = nn.Conv2d(384, 384, kernel_size=3, stride=1,
                         padding=1, bias=False)

        self.conv9 = nn.Conv2d(384, 512, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                         padding=1, bias=False)

        # self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
        #                  padding=1, bias=False)
        # self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
        #                  padding=1, bias=False)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                         padding=1, bias=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                         padding=1, bias=False)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                         padding=0, bias=False)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
                         padding=0, bias=False)
        self.conv15 = nn.Conv2d(512, 41, kernel_size=1, stride=1,
                         padding=0, bias=False)

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
        x = self.conv6(x)
        x = self.conv7(x)
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
        x = nn.functional.avg_pool2d(x, kernel_size=(6,4), padding=0)
        return x
