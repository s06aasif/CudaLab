#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('jupyter nbconvert --to python ConvRNN.ipynb')


# In[2]:


import torch
import torch.nn as nn


# In[4]:


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride=1):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True
        )
        
        
    def forward(self, x, hidden=None):
        """
        x: tensor of shape (batch_size, seq_len, channels, height, width)
        hidden: a tuple of the hidden and cell states, each of shape (batch_size, hidden_channels, height, width)
        """
        batch_size, seq_len, _, height, width = x.size()

        if hidden is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device),
                        torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device))
        else:
            h_t, c_t = hidden

        output_inner = []
        depth_maps_inner = []  
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            combined = torch.cat((x_t, h_t), dim=1)  
            gates = self.conv(combined)
            i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            output_inner.append(h_t)
            
        output = torch.stack(output_inner, dim=1)  # [batch_size, seq_len, hidden_channels, height, width]
        
        return output, (h_t, c_t)
    
    
    def init_hidden(self, batch_size, height, width):
        """
        Initializes hidden and cell states.
        """
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


# In[ ]:





# In[ ]:




