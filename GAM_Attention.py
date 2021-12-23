import torch.nn as nn
import torch

# class GAM_Attention(nn.Module):
#     def __init__(self, in_channels, out_channels, rate=4):
#         super(GAM_Attention, self).__init__()
#
#         self.channels_attnetion = nn.Sequential(
#             nn.Linear(in_channels, int(in_channels / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(in_channels / rate), in_channels)
#         )
#
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
#             nn.BatchNorm2d(int(in_channels / rate)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm2d(out_channels)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channels_attnetion(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#
#         x = x * x_channel_att
#
#         x_spatial_att = self.spatial_attention(x).sigmoid()
#         out = x * x_spatial_att
#
#         return out
#
# if __name__ == '__main__':
#     x = torch.rand(1, 16, 300, 300)
#     b, c, h, w = x.shape
#     net = GAM_Attention(in_channels=c, out_channels=c)
#     y = net(x)
#     print(y)
class ChannelsAttentionModul(nn.Module):
    def __init__(self, in_channel, rate=0.5):
        super(ChannelsAttentionModul, self).__init__()

        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * rate)),
            nn.ReLU(),
            nn.Linear(int(in_channel * rate), in_channel),
            nn.Sigmoid()
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * rate)),
            nn.ReLU(),
            nn.Linear(int(in_channel * rate), in_channel),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_branch = self.MaxPool(x)
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x

        return x

class SpatialAttentionModul(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x
        return x

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelsAttentionModul(in_channel=in_channel)
        self.Sam = SpatialAttentionModul(in_channel=in_channel)

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn(10, 100, 224, 224)
    model = CBAM(in_channel=100)
    print(model)
    outputs = model(inputs)
    print(inputs.shape)
    print(outputs.shape)

