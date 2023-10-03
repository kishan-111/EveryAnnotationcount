""" Full assembly of the parts to form the complete network """

from .unet_parts import *

""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, num_in_channels, num_classes, bilinear=False, dropout_probability=0, normalization='batchnorm',
                 num_groups=1):
        super(UNet, self).__init__()

        self.n_channels = num_in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.normalization = normalization
        self.num_groups = num_groups

        self.inc = DoubleConv(self.n_channels, 64, normalization=self.normalization, num_groups=self.num_groups)
        self.down1 = Down(64, 128, normalization=self.normalization, num_groups=self.num_groups)
        self.down2 = Down(128, 256, normalization=self.normalization, num_groups=self.num_groups)
        self.down3 = Down(256, 512, normalization=self.normalization, num_groups=self.num_groups)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor, normalization=self.normalization, num_groups=self.num_groups)
        self.up1 = Up(1024, 512 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up2 = Up(512, 256 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up3 = Up(256, 128 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up4 = Up(128, 64, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.outc = OutConv(64, self.n_classes)
        self.softmax = nn.Softmax2d()
        self.dropout = nn.Dropout2d(p=dropout_probability)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.dropout(x)
        logits = self.outc(x)
        # preds = self.softmax(logits)
        return logits


""" UNet with additional intermediate output-heads """

class HierarchicalUNet(nn.Module):
    def __init__(self, num_in_channels, num_classes, bilinear=False, dropout_probability=0.0, normalization='batchnorm',
                 num_groups=1):
        super(HierarchicalUNet, self).__init__()

        # Basic configuration parameters
        self.n_channels = num_in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.normalization = normalization
        self.num_groups = num_groups
        self.dropout = nn.Dropout2d(p=dropout_probability)

        # Standard UNet layer definitions
        self.inc = DoubleConv(self.n_channels, 64, normalization=self.normalization, num_groups=self.num_groups)
        self.down1 = Down(64, 128, normalization=self.normalization, num_groups=self.num_groups)
        self.down2 = Down(128, 256, normalization=self.normalization, num_groups=self.num_groups)
        self.down3 = Down(256, 512, normalization=self.normalization, num_groups=self.num_groups)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor, normalization=self.normalization, num_groups=self.num_groups)
        self.up1 = Up(1024, 512 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up2 = Up(512, 256 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up3 = Up(256, 128 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up4 = Up(128, 64, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)

        # Output head on the outermost layer (full-scale output)
        self.outc_mask = torch.nn.Sequential(OutConv(64, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                             OutConv(64, self.n_classes))

        # Hierarchically integrated intermediate output-heads
        self.outc_x5 = torch.nn.Sequential(OutConv(1024 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                           OutConv(64, self.n_classes))
        self.outc_up1 = torch.nn.Sequential(OutConv(512 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up2 = torch.nn.Sequential(OutConv(256 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up3 = torch.nn.Sequential(OutConv(128 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up4 = torch.nn.Sequential(OutConv(64, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))

    def forward(self, x):
        # UNet encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # UNet decoder + intermediate output-heads
        x5_logit = self.outc_x5(x5)

        x = self.up1(x5, x4)
        up1_logit = self.outc_up1(x)

        x = self.up2(x, x3)
        up2_logit = self.outc_up2(x)

        x = self.up3(x, x2)
        up3_logit = self.outc_up3(x)

        x = self.up4(x, x1)
        up4_logit = self.outc_up4(x)

        x = self.dropout(x)

        # Full-scale output conv
        logit_mask_ce = self.outc_mask(x)

        return logit_mask_ce, [x5_logit, up1_logit, up2_logit, up3_logit, up4_logit]