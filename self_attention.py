import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()

        self.in_channel = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward (self, x):
        b, c, w, h = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, w * h )

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, w, h)

        out = self.gamma*out + x
        return out,attention

if __name__ == '__main__':
    x = torch.randn(10, 3, 300, 300)
    model = SelfAttention(x, activation=nn.Softmax)
    print(model)