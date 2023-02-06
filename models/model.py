import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, dropout_rate,normalizationMode):
        """ This function instantiates all the model layers """

        super(Net, self).__init__()
        
        if normalizationMode=="BN":
            norm1=nn.BatchNorm2d(32)
            norm2=nn.BatchNorm2d(64)
            norm3=nn.BatchNorm2d(32)
            norm4=nn.BatchNorm2d(64)
            norm5=nn.BatchNorm2d(32)
            norm6=nn.BatchNorm2d(64)
            norm7=nn.BatchNorm2d(32)
            norm8=nn.BatchNorm2d(64)
          
        elif normalizationMode=="LN":
            norm1=nn.LayerNorm([14,26,26])
            norm2=nn.LayerNorm([14,26,26])
            norm3=nn.LayerNorm([14,26,26])
            norm4=nn.LayerNorm([14,26,26])
            norm5=nn.LayerNorm([14,26,26])
            norm6=nn.LayerNorm([14,26,26])
            norm7=nn.LayerNorm([14,26,26])
            norm8=nn.LayerNorm([14,26,26])


        elif normalizationMode=="GN":
            norm1=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm2=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm3=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm4=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm5=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm6=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm7=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
            norm8=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
           

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            norm1,
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.ReLU(),
            norm2,
            #nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 32x32x64 | Output: 16x16x64 | RF: 6x6
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=,stride=2),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 16x16x64 | Output: 16x16x32 | RF: 6x6
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 16x16x32 | Output: 16x16x32 | RF: 10x10
            nn.ReLU(),
            norm3,
            #nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Input: 16x16x32 | Output: 16x16x64 | RF: 14x14
            nn.ReLU(),
            norm4,
            #nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock2 = nn.Sequential(
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=2),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=2),
            nn.MaxPool2d(2, 2),  # Input: 16x16x64 | Output: 8x8x64 | RF: 16x16
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 8x8x64 | Output: 8x8x32 | RF: 16x16
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 24x24
            nn.ReLU(),
            norm5,
            #nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Depthwise separable convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 32x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),  # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32
            nn.ReLU(),
            norm6,
            #nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 8x8x64 | Output: 4x4x64 | RF: 36x36
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=2),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 4x4x64 | Output: 4x4x32 | RF: 36x36
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Input: 4x4x32 | Output: 4x4x32 | RF: 52x52
            nn.ReLU(),
            norm7,
            #nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Dilated convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=2),  # Input: 4x4x32 | Output: 4x4x64 | RF: 84x84
            nn.ReLU(),
            norm8,
            #nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 4x4x64 | Output: 1x1x64 | RF: 108x108

        # self.fc = nn.Sequential(
            # nn.Linear(64, 10)
        # )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        """ This function defines the network structure """

        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.transblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        #x = self.fc(x)
        #return x
