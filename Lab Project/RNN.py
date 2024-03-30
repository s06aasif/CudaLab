#!/usr/bin/env python
# coding: utf-8

# In[7]:


#get_ipython().system('jupyter nbconvert --to python RNN.ipynb')


# In[5]:





# In[6]:


import torch
import torch.nn as nn
class EgoMotionRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EgoMotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, combined_input):
        outputs, (hn, cn) = self.rnn(combined_input)
        output_sequence = self.fc(outputs)
        return output_sequence


# In[ ]:




