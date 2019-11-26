from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, se_loss=False, nclass=21):
        super(SELayer, self).__init__()
        self.se_loss = se_loss
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.se_loss:
            self.linear1 = nn.Linear(channel, channel // reduction)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(channel // reduction, channel)
            self.sigmoid = nn.Sigmoid()
            self.seloss = nn.Linear(channel // reduction, nclass)
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        if self.se_loss:
            y = self.linear1(y)
            cross_road = self.relu(y)
            y = self.linear2(cross_road)
            y = self.sigmoid(y).view(b, c, 1, 1)
            se_pred = self.seloss(cross_road)
            return x*y, se_pred
        else:
            y = self.fc(y).view(b, c, 1, 1)
            return x * y

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channel, channel//reduction, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
#         self.fc2 = nn.Conv2d(channel//reduction, channel, kernel_size=1)

#     def forward(self, out):
#         # Squeeze
#         w = self.avg_pool(out)
#         w = F.relu(self.fc1(w))
#         w = F.sigmoid(self.fc2(w))
#         # Excitation
#         out = out * w  # New broadcasting feature from v0.2!
#         #print(out.size())
#         return out