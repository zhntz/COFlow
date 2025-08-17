import torch
from torch import nn

#
# a = torch.randn(436,1024,2)
#
# print(a[180:, :, :].shape)

a = torch.ones(1,3,4)
criterionL1 = nn.Dropout(1)
a = criterionL1(a)
print(a)