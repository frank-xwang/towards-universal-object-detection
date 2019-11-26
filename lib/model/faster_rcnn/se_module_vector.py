from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=True):
        super(SELayer, self).__init__()
        self.with_sigmoid = with_sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_sigmoid:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y