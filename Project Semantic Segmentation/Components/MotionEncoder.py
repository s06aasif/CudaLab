#!/usr/bin/env python
# coding: utf-8

# In[7]:


#get_ipython().system('jupyter nbconvert --to python MotionEncoder.ipynb')


# In[9]:


import torch
import torch.nn as nn
class MotionEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size = 3):
        super(MotionEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, current_features, optical_flow):
        _, _, H, W = current_features.shape
        optical_flow_resized = F.interpolate(optical_flow, size=(H, W), mode='bilinear', align_corners=False)
        # Concatenate the features along the channel dimension with the optical flow
        combined_features = torch.cat([current_features, optical_flow_resized], dim=1)
        return self.conv(combined_features)


# In[ ]:





# In[ ]:




