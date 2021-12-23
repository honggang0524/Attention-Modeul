import torch
import torch.nn as nn

class Bneck(nn.Module):
    """
    input_size: input Dim
    operator_kernel: Dpth_conv kernel
    exp_size: expand dim
    out_size: out dim
    """
    def __init__(self, input_size, operator_kernel, exp_size, out_size, NL, s, SE=False, skip_connection=False):
        super(Bneck, self).__init__()

        self.conv_1_1_up = nn.Conv2d(input_size, exp_size, 1)
        if NL == 'RE':
            self.nl1 = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(exp_size)
            )
        elif NL == 'HS':
            self.nl1 = nn.Sequential(
                nn.Hardswish(),
                nn.BatchNorm2d(exp_size)
            )

        self.depth_conv = nn.Conv2d(exp_size, exp_size, kernel_size=operator_kernel, stride=s, groups=exp_size,
                                    padding=(operator_kernel - 1) // 2)

        self.nl2 = nn.Sequential(
            self.nl1,
            nn.BatchNorm2d(exp_size)
        )

        self.conv_1_1_down = nn.Conv2d(exp_size, out_size, 1)

        self.se = SE
        if SE:
            self.se_block = SEblock(exp_size)

        self.skip = skip_connection

    def forward (self, x):
        x1 = self.conv_1_1_up(x)
        x1 = self.nl1(x)

        x2 = self.depth_conv(x1)
        x2 = self.nl2(x2)

        if self.se:
            x2 = self.se_block(x2)

        x3 = self.conv_1_1_down(x2)


        if self.skip:
            x3 = x3 + x

        return x3

class SEblock(nn.Module):
    def __init__(self, in_channel, rate = 0.25):
        super(SEblock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * rate)),
            nn.ReLU(),
            nn.Linear(int(in_channel * rate), in_channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), 1)
        weight = self.fc(branch)

        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))
        scale = weight * x
        return scale

class MobileNet_v3(nn.Module):
    def __init__(self, k):
        super(MobileNet_v3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 1, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.layer2 = Bneck(input_size=16, operator_kernel=3, exp_size=16, out_size=16, NL='RE', s=1,
                            SE=False, skip_connection=True)
        self.layer3 = Bneck(input_size=16, operator_kernel=3, exp_size=64, out_size=24, NL='RE', s=2,
                            SE=False, skip_connection=False)

        self.layer4 = Bneck(input_size=24, operator_kernel=3, exp_size=72, out_size=24, NL='RE', s=1,
                            SE=False, skip_connection=True)

        self.layer5 = Bneck(input_size=24, operator_kernel=5, exp_size=72, out_size=40, NL='RE', s=2,
                            SE=True, skip_connection=False)

        self.layer6 = Bneck(input_size=40, operator_kernel=5, exp_size=120, out_size=40, NL='RE', s=1,
                            SE=True, skip_connection=True)

        self.layer7 = Bneck(input_size=40, operator_kernel=5, exp_size=120, out_size=40, NL='RE', s=1,
                            SE=True, skip_connection=True)

        self.layer8 = Bneck(input_size=40, operator_kernel=3, exp_size=240, out_size=80, NL='HS', s=2,
                            SE=False, skip_connection=False)

        self.layer9 = Bneck(input_size=80, operator_kernel=3, exp_size=200, out_size=80, NL='HS', s=1,
                            SE=False, skip_connection=True)

        self.layer10 = Bneck(input_size=80, operator_kernel=3, exp_size=184, out_size=80, NL='HS', s=1,
                             SE=False, skip_connection=True)

        self.layer11 = Bneck(input_size=80, operator_kernel=3, exp_size=184, out_size=80, NL='HS', s=1,
                             SE=False, skip_connection=True)

        self.layer12 = Bneck(input_size=80, operator_kernel=3, exp_size=480, out_size=112, NL='HS', s=1,
                             SE=False, skip_connection=False)

        self.layer13 = Bneck(input_size=112, operator_kernel=3, exp_size=672, out_size=112, NL='HS', s=1,
                             SE=True, skip_connection=True)

        self.layer14 = Bneck(input_size=112, operator_kernel=5, exp_size=672, out_size=160, NL='HS', s=2,
                             SE=True, skip_connection=False)

        self.layer15 = Bneck(input_size=160, operator_kernel=5, exp_size=960, out_size=160, NL='HS', s=1,
                             SE=True, skip_connection=True)

        self.layer16 = Bneck(input_size=160, operator_kernel=5, exp_size=960, out_size=160, NL='HS', s=1,
                             SE=True, skip_connection=True)
        self.layer17 = nn.Sequential(
            nn.Conv2d(160, 960, 1, stride=1),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )
        self.layer18_pool = nn.AvgPool2d((7,7), stride=1)
        self.layer19 = nn.Sequential(
            nn.Conv2d(960, 1280, 1, stride=1),
            nn.Hardswish()
        )
        self.layer20 = nn.Sequential(
            nn.Conv2d(1280, k, 1, stride=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18_pool(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = torch.squeeze(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn(10, 3, 224, 224)
    model = MobileNet_v3(k=21)
    print(model)


