#!/usr/bin/env python
# coding: utf-8

# In[9]:


#get_ipython().system('jupyter nbconvert --to python EncoderCNN.ipynb')


# In[10]:


import torch.nn as nn


# In[11]:


import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Downnsampling Images
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        
    def forward(self, x):
        x = self.features(x)
        return x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




