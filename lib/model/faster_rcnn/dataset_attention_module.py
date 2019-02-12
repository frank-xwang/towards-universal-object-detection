import torch
from torch import nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.faster_rcnn.se_module_vector import SELayer

class DatasetsAttention(nn.Module):
    def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None, fixed_block=False, num_se=0):
        super(DatasetsAttention, self).__init__()
        self.planes = planes
        if num_se == 0:
            self.n_datasets = len(nclass_list)
        else:
            self.n_datasets = num_se
        self.fixed_block = fixed_block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cfg.less_blocks = False
        if not self.fixed_block and cfg.less_blocks:
            if cfg.layer_index %2 == 0:
                self.fixed_block = True
                # print(cfg.layer_index)
        if self.fixed_block:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class, with_sigmoid=False) for num_class in range(1)])
        elif num_se == 0:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class, with_sigmoid=False) for num_class in nclass_list])
        else:
            self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class, with_sigmoid=False) for num_class in range(num_se)])
        self.fc_1 = nn.Linear(planes, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()

        if self.fixed_block:
            SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            weight = self.fc_1(self.avg_pool(x).view(b, c))
            weight = self.softmax(weight).view(b, self.n_datasets, 1)
            for i, SE_Layer in enumerate(self.SE_Layers):
                if i == 0:
                    SELayers_Matrix = SE_Layer(x).view(b, c, 1)
                else:
                    SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
            SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        return x*SELayers_Matrix

# ## 88 use this module 
# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.planes = planes
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc_1 = nn.Linear((self.ndatasets + 1) * self.planes, self.planes//reduction)        
#         self.relu = nn.ReLU(inplace=True)
#         self.fc_2 = nn.Linear(self.planes//reduction, self.planes)                
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         SELayers_Matrix_ = self.avg_pool(x).view(b, c, 1)
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             if i == 0:
#                 SELayers_Matrix = SE_Layer(x).view(b, c, 1)
#             else:
#                 SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
#         # print('0',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_1(SELayers_Matrix.view(b,(self.ndatasets + 1) * self.planes))
#         # print('1',SELayers_Matrix.size())
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         # print('2',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_2(SELayers_Matrix).view(b, c, 1, 1)
#         # print('3',SELayers_Matrix.size())
#         SELayers_Matrix += SELayers_Matrix_
#         SELayers_Matrix = self.sigmoid(SELayers_Matrix)
#         return x*SELayers_Matrix

# # 91 use it
# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.planes = planes
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc_1 = nn.Linear(self.ndatasets + 1, self.ndatasets + 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc_2 = nn.Linear(self.ndatasets + 1, 1)          
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         SELayers_Matrix = self.avg_pool(x).view(b, c, 1)
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
#         # print('0',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_1(SELayers_Matrix.view(b*self.planes, self.ndatasets + 1))
#         # print('1',SELayers_Matrix.size())
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         # print('2',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_2(SELayers_Matrix).view(b, c, 1, 1)
#         # print('3',SELayers_Matrix.size())
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         return x*SELayers_Matrix

# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.planes = planes
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc_1 = nn.Linear(self.ndatasets + 1, self.ndatasets + 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc_2 = nn.Linear(self.ndatasets + 1, 1)                
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         SELayers_Matrix = self.avg_pool(x).view(b, c, 1)
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
#         # print('0',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_1(SELayers_Matrix.view(b*self.planes, self.ndatasets + 1))
#         # print('1',SELayers_Matrix.size())
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         # print('2',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_2(SELayers_Matrix).view(b, c, 1, 1)
#         # print('3',SELayers_Matrix.size())
#         SELayers_Matrix = self.sigmoid(SELayers_Matrix)
#         return x*SELayers_Matrix

# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.planes = planes
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc_1 = nn.Linear((self.ndatasets + 1) * self.planes, self.planes//reduction)        
#         self.relu = nn.ReLU(inplace=True)
#         self.fc_2 = nn.Linear(self.planes//reduction, self.planes)                
#         self.sigmoid = nn.Sigmoid()
        
#         #self.downsample = torch.nn.functional.interpolate(size=(7,7), mode='bilinear')
#         self.fc_1 = nn.Linear(49, 1)
#         self.fc_2 = nn.Linear(self.planes, 3)

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         Scaler = torch.nn.functional.interpolate(x, size=(7,7), mode='bilinear').view(b, c, 7*7)
#         Scaler = self.fc_1(Scaler)
#         Scaler = self.relu(Scaler).view(b, self.planes)
#         Scaler = self.fc_2(Scaler).view(b, 3, 1)
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             if i == 0:
#                 SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1)
#             else:
#                 SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)),2)
#         # print('0',SELayers_Matrix.size())
#         SELayers_Matrix = SELayers_Matrix.bmm(Scaler)
#         SELayers_Matrix = self.sigmoid(SELayers_Matrix).view(b, c, 1, 1)
#         return x*SELayers_Matrix

## 30 and 31 use this module 

# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)

#         self.weight_1 = nn.Parameter(torch.ones(self.ndatasets, self.ndatasets))
#         #self.bias_1 = nn.Parameter(torch.zeros(self.ndatasets, self.ndatasets))
#         self.relu = nn.ReLU(inplace=True)
#         self.weight_2 = nn.Parameter(torch.ones(self.ndatasets, 1))
#         #self.bias_2 = nn.Parameter(torch.zeros(self.ndatasets, 1))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             if i == 0:
#                 SELayers_Matrix = self.SE_Layers[0](x).view(b, c, 1)
#             else:
#                 SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)),2)
#         #SELayers_Matrix = SELayers_Matrix @ self.weight_1 + self.bias_1
#         SELayers_Matrix = torch.mm(SELayers_Matrix.view(b*c, self.ndatasets), self.weight_1)
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         #SELayers_Matrix = SELayers_Matrix @ self.weight_2 + self.bias_2
#         SELayers_Matrix = torch.mm(SELayers_Matrix.view(b*c, self.ndatasets), self.weight_2).view(b, c, 1, 1)
#         SELayers_Matrix = self.sigmoid(SELayers_Matrix)
#         return x*SELayers_Matrix

# ## 32, 34 use this module 
# class DatasetsAttention(nn.Module):
#     def __init__(self, planes, reduction=16, se_loss=False, nclass_list=None):
#         super(DatasetsAttention, self).__init__()
#         self.planes = planes
#         self.SE_Layers = nn.ModuleList([SELayer(planes, reduction, se_loss=se_loss, nclass=num_class) for num_class in nclass_list])
#         self.ndatasets = len(nclass_list)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc_1 = nn.Linear((self.ndatasets + 1) * self.planes, self.planes//reduction)        
#         self.relu = nn.ReLU(inplace=True)
#         self.fc_2 = nn.Linear(self.planes//reduction, self.planes)                
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         SELayers_Matrix = self.avg_pool(x).view(b, c, 1)
#         for i, SE_Layer in enumerate(self.SE_Layers):
#             SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)
#         # print('0',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_1(SELayers_Matrix.view(b,(self.ndatasets + 1) * self.planes))
#         # print('1',SELayers_Matrix.size())
#         SELayers_Matrix = self.relu(SELayers_Matrix)
#         # print('2',SELayers_Matrix.size())
#         SELayers_Matrix = self.fc_2(SELayers_Matrix).view(b, c, 1, 1)
#         # print('3',SELayers_Matrix.size())
#         SELayers_Matrix = self.sigmoid(SELayers_Matrix)
#         return x*SELayers_Matrix