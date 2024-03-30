#!/usr/bin/env python
# coding: utf-8

# In[30]:


#get_ipython().system('jupyter nbconvert --to python Decoder.ipynb')


# In[29]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    
class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, in_channels // 2),
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels // 2, out_channels),
            )
        else:
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                DoubleConv(in_channels, in_channels // 2),
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 2, out_channels // 2, kernel_size=2, stride=2),
                DoubleConv(in_channels // 2, out_channels),
            )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return x




# In[ ]:





# In[ ]:





# In[ ]:




