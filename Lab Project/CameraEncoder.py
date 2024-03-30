#!/usr/bin/env python
# coding: utf-8

# In[22]:


#get_ipython().system('jupyter nbconvert --to python CameraEncoder.ipynb')


# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraEncoder(nn.Module):
    def __init__(self, input_channels, output_channels=6):
        super(CameraEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, stride=2, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.adjust_pool = nn.AdaptiveAvgPool2d((4, 4))  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Downsampling
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Further downsampling
        x = F.relu(self.conv3(x))
        x = self.adjust_pool(x)  # Directly achieve 4x4 output, focusing on adjusting the width.
        return x


# In[ ]:




