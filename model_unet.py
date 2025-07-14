import torch
import torch.nn as nn
import torch.nn.functional as F

print("PyTorch version:", torch.__version__)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lrelu2 = nn.LeakyReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.lrelu1(self.bn1(self.conv1(x)))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        return self.dropout(x)

class UNetPP(nn.Module):
    def __init__(self, input_channels=1):
        super(UNetPP, self).__init__()

        # Encoder
        self.x00 = ConvBlock(input_channels, 32)
        self.x10 = ConvBlock(32, 64)
        self.x20 = ConvBlock(64, 128)
        self.x30 = ConvBlock(128, 256)
        self.x40 = ConvBlock(256, 512, dropout_rate=0.5)

        # Decoder
        self.x01 = ConvBlock(32 + 64, 32)
        self.x11 = ConvBlock(64 + 128, 64)
        self.x21 = ConvBlock(128 + 256, 128)
        self.x31 = ConvBlock(256 + 512, 512)

        self.x02 = ConvBlock(32 + 32 + 64, 32)
        self.x12 = ConvBlock(64 + 64 + 128, 64)
        self.x22 = ConvBlock(128 + 128 + 512, 128)

        self.x03 = ConvBlock(32 + 32 + 32 + 64, 32)
        self.x13 = ConvBlock(64 + 64 + 64 + 128, 64)

        self.x04 = ConvBlock(32 + 32 + 32 + 32 + 64, 32)

        # Outputs
        self.out1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Tanh())
        self.out2 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Tanh())
        self.out3 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Tanh())
        self.out4 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Tanh())

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(F.max_pool2d(x00, 2))
        x20 = self.x20(F.max_pool2d(x10, 2))
        x30 = self.x30(F.max_pool2d(x20, 2))
        x40 = self.x40(F.max_pool2d(x30, 2))

        x01 = self.x01(torch.cat([x00, F.interpolate(x10, scale_factor=2, mode='nearest')], dim=1))
        x11 = self.x11(torch.cat([x10, F.interpolate(x20, scale_factor=2, mode='nearest')], dim=1))
        x21 = self.x21(torch.cat([x20, F.interpolate(x30, scale_factor=2, mode='nearest')], dim=1))
        x31 = self.x31(torch.cat([x30, F.interpolate(x40, scale_factor=2, mode='nearest')], dim=1))

        x02 = self.x02(torch.cat([x00, x01, F.interpolate(x11, scale_factor=2, mode='nearest')], dim=1))
        x12 = self.x12(torch.cat([x10, x11, F.interpolate(x21, scale_factor=2, mode='nearest')], dim=1))
        x22 = self.x22(torch.cat([x20, x21, F.interpolate(x31, scale_factor=2, mode='nearest')], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, F.interpolate(x12, scale_factor=2, mode='nearest')], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, F.interpolate(x22, scale_factor=2, mode='nearest')], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, F.interpolate(x13, scale_factor=2, mode='nearest')], dim=1))

        sd1 = self.out1(x01)
        sd2 = self.out2(x02)
        sd3 = self.out3(x03)
        sd4 = self.out4(x04)

        return sd1, sd2, sd3, sd4

def unet(pretrained_weights=None, compile_model=True):
    model = UNetPP(input_channels=1)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))
    return model
