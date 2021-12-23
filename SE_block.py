import torch.nn as nn
import torch
class seBlock(nn.Module):
    def __init__(self, channels, r=0.5):
        super(seBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, int(channels * r)),
            nn.ReLU(),
            nn.Linear(int(channels * r), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        weight = self.fc(branch)

        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        scale = weight * x
        return scale

if __name__ == '__main__':
    inputs = torch.randn(10, 3, 64, 64)
    se = seBlock(inputs)
    print(se)

