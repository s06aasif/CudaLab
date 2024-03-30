#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('jupyter nbconvert --to python CameraDecoder.ipynb')


# In[1]:


import torch.nn as nn
class CameraDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CameraDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 96) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 6, 4, 4)
        return x


# In[ ]:





# In[ ]:




