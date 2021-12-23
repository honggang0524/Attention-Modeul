import torch
import torch.nn as nn
from MobileNet_v3_large import Bneck
class MobileNet_v3_small(nn.Module):
    def __init__(self, k):
        super(MobileNet_v3_small, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.layer2 = Bneck(input_size=16, operator_kernel=3, exp_size=16, out_size=16, NL='RE',
                            s=2, SE=True, skip_connection=False)
        self.layer3 = Bneck(input_size=16, operator_kernel=3, exp_size=72, out_size=24, NL='RE',
                            s=2, SE=False, skip_connection=False)
        self.layer4 = Bneck(input_size=24, operator_kernel=3, exp_size=88, out_size=24, NL='RE',
                            s=1, SE=False, skip_connection=True)
        self.layer5 = Bneck(input_size=24, operator_kernel=5, exp_size=96, out_size=40, NL='HS',
                            s=2, SE=True, skip_connection=False)
        self.layer6 = Bneck(input_size=40, operator_kernel=5, exp_size=240, out_size=40, NL='HS',
                            s=1, SE=True, skip_connection=True)
        self.layer7 = Bneck(input_size=40, operator_kernel=5, exp_size=240, out_size=40, NL='HS',
                            s=1, SE=True, skip_connection=True)
        self.layer8 = Bneck(input_size=40, operator_kernel=5, exp_size=120, out_size=48, NL='HS',
                            s=1, SE=True, skip_connection=False)
        self.layer9 = Bneck(input_size=48, operator_kernel=5, exp_size=144, out_size=48, NL='HS',
                            s=1, SE=True, skip_connection=True)
        self.layer10 = Bneck(input_size=48, operator_kernel=5, exp_size=288, out_size=96, NL='HS',
                            s=2, SE=True, skip_connection=False)
        self.layer11 = Bneck(input_size=96, operator_kernel=5, exp_size=576, out_size=96, NL='HS',
                            s=1, SE=True, skip_connection=False)
        self.layer12 = Bneck(input_size=96, operator_kernel=5, exp_size=576, out_size=96, NL='HS',
                            s=1, SE=True, skip_connection=True)

        self.layer13 = nn.Sequential(
            nn.Conv2d(96, 576, 1, stride=1),
            nn.BatchNorm2d(576),
            nn.Hardswish()
        )

        self.layer14_pool = nn.AvgPool2d((7,7), stride=1)

        self.layer15 = nn.Sequential(
            nn.Conv2d(576, 1024, 1, stride=1),
            nn.Hardswish()
        )

        self.layer16 = nn.Sequential(
            nn.Conv2d(1024, k, 1, stride=1)
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
        x = self.layer14_pool(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = torch.squeeze(x)
        return x
if __name__ == '__main__':
    inputs = torch.randn(10, 3, 300, 300)
    model = MobileNet_v3_small(k=21)
    print(model)




