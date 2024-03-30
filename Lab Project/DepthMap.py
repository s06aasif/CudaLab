#!/usr/bin/env python
# coding: utf-8

# In[6]:


#get_ipython().system('jupyter nbconvert --to python DepthMap.ipynb')


# In[5]:


import torch.nn as nn
import torch.nn.functional as F
class DepthDecoder(nn.Module):
    def __init__(self, input_channels, depth_channels=1):
        super(DepthDecoder, self).__init__()

        self.upsample1 = nn.ConvTranspose2d(input_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample2 = nn.ConvTranspose2d(32, depth_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.depth_conv = nn.Conv2d(depth_channels, depth_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(x)
        x = self.upsample2(x)
        x = F.relu(x)
        x = self.depth_conv(x)
        return x


# In[ ]:




