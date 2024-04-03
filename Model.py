import torch
import torch.nn as nn

class ourModel(nn.Module): # ver 1
    def __init__(self,num_class):
        super(ourModel, self).__init__()
        self.conv1 = nn.Conv2d(24, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 22 * 22, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.bn1(self.pool1(torch.relu(self.conv1(x))))
        print(x.shape)
        x = self.bn2(self.pool2(torch.relu(self.conv2(x))))
        print(x.shape)
        x = x.view(-1, 128 * 22 * 22)
        print(x.shape)
        x = torch.relu(self.fc1(x))
        #print(x.shape)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)
        return x

class ourConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, cancelReLu = False):
        super(ourConv2d, self).__init__()
        self.cancelReLu = cancelReLu
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.cancelReLu:
            x = self.bn(self.conv(x))
        else:
            x = self.relu(self.bn(self.conv(x)))
        return x

class Inception_Block_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inception_Block_1, self).__init__()
        self.branch_1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_1_conv2d = ourConv2d(in_channel, out_channel, kernel_size=1)
        self.branch_2_conv2d_1x1 = ourConv2d(in_channel, 48, kernel_size=1)
        self.branch_2_conv2d_3x3_1 = ourConv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.branch_2_conv2d_3x3_2 = ourConv2d(96, out_channel, kernel_size=3, stride=1, padding=1)
        self.branch_3_conv2d_1x1 = ourConv2d(in_channel, 48, kernel_size=1)
        self.branch_3_conv2d_3x3 = ourConv2d(48, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch_1 = self.branch_1_conv2d(self.branch_1_pool(x))
        branch_2 = self.branch_2_conv2d_3x3_2(self.branch_2_conv2d_3x3_1(self.branch_2_conv2d_1x1(x)))
        branch_3 = self.branch_3_conv2d_3x3(self.branch_3_conv2d_1x1(x))
        outputs = torch.cat([branch_1, branch_2, branch_3], 1)
        return outputs

class Inception_Block_2(nn.Module):
    def __init__(self, in_channel):
        super(Inception_Block_2, self).__init__()
        self.branch_1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_1_conv2d = ourConv2d(in_channel, 96, kernel_size=1)
        self.branch_2_conv2d_1x1 = ourConv2d(in_channel, 96, kernel_size=1)
        self.branch_2_conv2d_3x3_1 = ourConv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.branch_2_conv2d_3x3_2 = ourConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.branch_3_conv2d_1x1 = ourConv2d(in_channel, 48, kernel_size=1)
        self.branch_3_conv2d_3x3 = ourConv2d(48, 72, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch_1 = self.branch_1_conv2d(self.branch_1_pool(x))
        branch_2 = self.branch_2_conv2d_3x3_2(self.branch_2_conv2d_3x3_1(self.branch_2_conv2d_1x1(x)))
        branch_3 = self.branch_3_conv2d_3x3(self.branch_3_conv2d_1x1(x))
        outputs = torch.cat([branch_1, branch_2, branch_3], 1)
        return outputs

class Inception_Block_3(nn.Module):
    def __init__(self, in_channel):
        super(Inception_Block_3, self).__init__()
        self.branch_1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_1_conv2d = ourConv2d(in_channel, 224, kernel_size=1)
        self.branch_2_conv2d_1x1 = ourConv2d(in_channel, 224, kernel_size=1)
        self.branch_2_conv2d_3x3 = ourConv2d(224, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch_1 = self.branch_1_conv2d(self.branch_1_pool(x))
        branch_2 = self.branch_2_conv2d_3x3(self.branch_2_conv2d_1x1(x))
        outputs = torch.cat([branch_1, branch_2], 1)
        return outputs

class Inception(nn.Module):
    def __init__(self,num_class,dropout_value=0):
        super(Inception, self).__init__()
        self.conv1 = ourConv2d(24, 64, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = ourConv2d(64, 96, kernel_size=3, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block1 = Inception_Block_1(96,96)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2 = Inception_Block_2(288)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block3 = Inception_Block_3(296)
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(480 * 1 * 1, 196)
        self.dropout = nn.Dropout(p=dropout_value)
        self.fc2 = nn.Linear(196, num_class)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        #print("1 : ",x.shape)
        x = self.pool2(self.conv2(x))
        #print("2 : ",x.shape)
        x = self.block1(x)
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.pool4(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        x = self.gap(x)
        #print(x.shape)
        x = x.view(-1, 480 * 1 * 1)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        #print(x.shape)
        x = torch.relu(self.fc2(x))
        #print(x.shape)
        x = nn.functional.softmax(x, dim=1)
        #print(x.shape)
        return x

class Skip_Inception_Block_1(nn.Module):
    def __init__(self, in_channel, out_channel, scale=1.0):
        super(Skip_Inception_Block_1, self).__init__()
        self.scale = scale
        self.branch_1 = ourConv2d(in_channel, out_channel, kernel_size=1)# i : 96, o : 96
        self.branch_2 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, 96, kernel_size=3, stride=1, padding=1),
            ourConv2d(96, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d((3*out_channel), in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        integrad_1 = torch.cat([branch_1, branch_2, branch_3], 1)
        integrad_2 = self.conv(integrad_1)
        outputs = torch.relu(x + self.gamma * integrad_2)
        return outputs

class Skip_Inception_Block_3(nn.Module):
    def __init__(self, in_channel, scale=1.0):
        super(Skip_Inception_Block_3, self).__init__()
        self.scale = scale
        self.branch_1 = ourConv2d(in_channel, 96, kernel_size=1)# i : 288 other : 1152 192 128 160 192 1152
        self.branch_2 = nn.Sequential(
            ourConv2d(in_channel, 96, kernel_size=1),
            ourConv2d(96, 128, kernel_size=3, stride=1, padding=1),
            ourConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, 72, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d((96+128+72), in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        integrad_1 = torch.cat([branch_1, branch_2, branch_3], 1)
        integrad_2 = self.conv(integrad_1)
        outputs = torch.relu(x + self.gamma * integrad_2)
        return outputs

class Skip_Inception_Block_5(nn.Module):
    def __init__(self, in_channel, scale=1.0):
        super(Skip_Inception_Block_5, self).__init__()
        self.scale = scale
        self.branch_1 = ourConv2d(in_channel, 224, kernel_size=1)
        self.branch_2 = nn.Sequential(
            ourConv2d(in_channel, 224, kernel_size=1),
            ourConv2d(224, 256, kernel_size=3, stride=1, padding=1),
            ourConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ourConv2d(in_channel, 224, kernel_size=1),
            ourConv2d(224, 256, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d((224+256+256), in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        integrad_1 = torch.cat([branch_1, branch_2, branch_3], 1)
        integrad_2 = self.conv(integrad_1)
        outputs = torch.relu(x + self.gamma * integrad_2)
        return outputs

class Reduce_Inception_Block_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduce_Inception_Block_2, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ourConv2d(in_channel, 96, kernel_size=1)
            )
        self.branch_2 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, 96, kernel_size=3, stride=1, padding=1),
            ourConv2d(96, out_channel, kernel_size=3, stride=2, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, out_channel, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        outputs = torch.cat([branch_1, branch_2, branch_3], 1)
        return outputs

class Reduce_Inception_Block_4(nn.Module):
    def __init__(self, in_channel):
        super(Reduce_Inception_Block_4, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ourConv2d(in_channel, 96, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            ourConv2d(in_channel, 96, kernel_size=1),
            ourConv2d(96, 128, kernel_size=3, stride=1, padding=1),
            ourConv2d(128, 128, kernel_size=3, stride=2, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ourConv2d(in_channel, 48, kernel_size=1),
            ourConv2d(48, 72, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        outputs = torch.cat([branch_1, branch_2, branch_3], 1)
        return outputs

class Skip_Inception(nn.Module):
    def __init__(self,num_class,dropout_value=0):
        super(Skip_Inception, self).__init__()
        self.conv1 = ourConv2d(24, 64, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = ourConv2d(64, 96, kernel_size=3, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Sblock1 = Skip_Inception_Block_1(96,96)
        self.Rblock2 = Reduce_Inception_Block_2(96,96)
        self.Sblock3 = Skip_Inception_Block_3(288)
        self.Rblock4 = Reduce_Inception_Block_4(288)
        self.Sblock5 = Skip_Inception_Block_5(296)
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(296 * 1 * 1, 148)
        self.dropout = nn.Dropout(p=dropout_value)
        self.fc2 = nn.Linear(148, num_class)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        #print("1 : ",x.shape)
        x = self.pool2(self.conv2(x))
        #print("2 : ",x.shape)
        x = self.Sblock1(x)
        #print("3 : ",x.shape)
        x = self.Rblock2(x)
        #print("4 : ",x.shape)
        x = self.Sblock3(x)
        #print("5 : ",x.shape)
        x = self.Rblock4(x)
        #print("6 : ",x.shape)
        x = self.Sblock5(x)
        #print("7 : ",x.shape)
        x = self.gap(x)
        #print("8 : ",x.shape)
        x = x.view(-1, 296 * 1 * 1)
        #print("9 : ",x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        #print("10 : ",x.shape)
        x = torch.relu(self.fc2(x))
        #print(x.shape)
        x = nn.functional.softmax(x, dim=1)
        #print(x.shape)
        return x

class CBAM_Inception(nn.Module):
    def __init__(self,num_class,dropout_value=0):
        super(CBAM_Inception, self).__init__()
        self.conv1 = ourConv2d(24, 64, kernel_size=5, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = ourConv2d(64, 96, kernel_size=3, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ca2 = ChannelAttention(96)
        self.sa2 = SpatialAttention()
        self.block1 = Inception_Block_1(96,96)
        self.ca3 = ChannelAttention(288)
        self.sa3 = SpatialAttention()
        self.bn3 = nn.BatchNorm2d(288)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2 = Inception_Block_2(288)
        self.ca4 = ChannelAttention(296)
        self.sa4 = SpatialAttention()
        self.bn4 = nn.BatchNorm2d(296)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block3 = Inception_Block_3(296)
        self.ca5 = ChannelAttention(480)
        self.sa5 = SpatialAttention()
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(480 * 1 * 1, 196)
        self.dropout = nn.Dropout(p=dropout_value)
        self.fc2 = nn.Linear(196, num_class)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        #print("1 : ",x.shape)
        x = self.pool2(self.conv2(x))
        #print("2 : ",x.shape)
        x = self.block1(x)
        #print("3 : ",x.shape)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)
        #print("4 : ",x.shape)
        x = self.block2(x)
        #print("5 : ",x.shape)
        x = self.ca4(x) * x
        x = self.sa4(x) * x
        x = torch.relu(x)
        x = self.bn4(x)
        x = self.pool4(x)
        #print("6 : ",x.shape)
        x = self.block3(x)
        #print("7 : ",x.shape)
        x = self.gap(x)
        #print("8 : ",x.shape)
        x = x.view(-1, 480 * 1 * 1)
        #print("9 : ",x.shape)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        #print("10 : ",x.shape)
        x = torch.relu(self.fc2(x))
        #print("11 : ",x.shape)
        x = nn.functional.softmax(x, dim=1)
        #print("12 : ",x.shape)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #(1) mean let output dimension become 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes//16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes//16,in_planes,1,bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2,1,kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x , dim=1, keepdim=True)
        max_out = torch.mean(x , dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x
