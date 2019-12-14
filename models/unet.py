import torch
import torch.nn as nn

class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

def UNet_up_conv(input_channel, prev_channel, learned_bilinear=True):

    if learned_bilinear:
        return nn.ConvTranspose2d(input_channel, prev_channel, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, prev_channel, kernel_size=3, padding=1))

class UNet_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_up_block, self).__init__()
        self.up_sampling = UNet_up_conv(input_channel, prev_channel, learned_bilinear)
        self.bn1 = nn.BatchNorm2d(prev_channel)
        self.conv2 = nn.Conv2d(prev_channel+prev_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, pre_feature_map, x):
        x = self.up_sampling(x)
        x = self.relu(self.bn1(x))
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class UNet(nn.Module):
    def __init__(self, colordim=3, n_classes=2, learned_bilinear=False):
        super(UNet, self).__init__()
        self.learned_bilinear = learned_bilinear

        self.down_block1 = UNet_down_block(colordim, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 256, True)
        self.down_block4 = UNet_down_block(256, 512, True)
        self.down_block5 = UNet_down_block(512, 1024, True)

        self.up_block1 = UNet_up_block(1024, 512, 512, self.learned_bilinear)
        self.up_block2 = UNet_up_block(512, 256, 256, self.learned_bilinear)
        self.up_block3 = UNet_up_block(256, 128, 128, self.learned_bilinear)
        self.up_block4 = UNet_up_block(128, 64, 64, self.learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, n_classes, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        x = self.up_block1(self.x4, self.x5)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.last_conv1(x)
        return x
