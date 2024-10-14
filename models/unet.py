import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(ConvBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels//2),
            nn.ReLU(),
            nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        res = self.encode(x)
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConvBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=1)
            
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512], bottleneck_channel=1024):
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls, level_4_chnls = level_channels[0], level_channels[1], level_channels[2], level_channels[3]
        self.a_block1 = ConvBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = ConvBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = ConvBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.a_block4 = ConvBlock(in_channels=level_3_chnls, out_channels=level_4_chnls)
        self.bottleNeck = ConvBlock(in_channels=level_4_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block4 = UpConvBlock(in_channels=bottleneck_channel, res_channels=level_4_chnls)
        self.s_block3 = UpConvBlock(in_channels=level_4_chnls, res_channels=level_3_chnls)
        self.s_block2 = UpConvBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConvBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def forward(self, input):
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, residual_level4 = self.a_block4(out)
        out, _ = self.bottleNeck(out)

        out = self.s_block4(out, residual_level4)
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out

class UpBlock_asy(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpBlock_asy, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out

class UNet_asy(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512):
        super(UNet_asy, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = ConvBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = ConvBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = ConvBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = ConvBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        
        self.CatChannels = level_channels[2]
        
        self.s_block3 = UpBlock_asy(in_channels=bottleneck_channel, out_channels=self.CatChannels, res_channels=level_3_chnls)
        self.s_block2 = UpBlock_asy(in_channels=self.CatChannels, out_channels=self.CatChannels, res_channels=level_2_chnls)
        self.s_block1 = UpBlock_asy(in_channels=self.CatChannels, out_channels=self.CatChannels, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def forward(self, input):
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out